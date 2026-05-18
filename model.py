# model.py 
import math
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEFAULT_CONFIG, HeadTypes



### backbones

class TIMMBackbone(nn.Module):
    """Backbone using TIMM library models"""
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
    def forward(self, x):
        return self.model(x)


### heads

class RegressionHead(nn.Module):
    """Head for direct regression. Outputs a single continuous value"""
    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            
            nn.Linear(embedding_dim, 512), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            
            nn.Linear(512, 128), 
            nn.ReLU(),
            
            nn.Linear(128, 1),
        )

    def forward(self, features):
        return self.head(features)
    
class RegressionSigmoidHead(nn.Module):
    """Head for direct regression. Outputs a single continuous value"""
    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            
            nn.Linear(embedding_dim, 512), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            
            nn.Linear(512, 128), 
            nn.ReLU(),
            
            nn.Linear(128, 1),
            nn.Sigmoid() # restricts output to [0, 1] for stability
        )

    def forward(self, features):
        # scale output to [0, 100]
        return self.head(features) * 100 

class BinnedRegressionHead(nn.Module):
    """Head for binned regression. Outputs logits for bins spanning exactly [min_val, max_val]"""
    def __init__(self, embedding_dim: int, min_val: int = -5, max_val: int = 105, temperature: float = 0.75, **kwargs):
        super().__init__()
        self.temperature = temperature 
        self.min_val = min_val
        self.max_val = max_val
        self.num_bins = int(abs(max_val) + abs(min_val)) + 1

        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            
            nn.Linear(embedding_dim, 512), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            
            nn.Linear(512, 128), 
            nn.ReLU(),
            
            nn.Linear(128, self.num_bins)
        )

    def forward(self, features):
        return self.head(features)

class ClassificationHead(nn.Module):
    """Head for classification. Outputs logits for N classes"""
    def __init__(self, embedding_dim: int, num_classes: int = 100, **kwargs):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            
            nn.Linear(embedding_dim, 512), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            
            nn.Linear(512, 128), 
            nn.ReLU(),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, features):
        return self.head(features)

class DeepRegressionHead(nn.Module):
    """Deeper MLP for more complex feature extraction"""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            
            nn.Linear(embedding_dim, 1024), 
            nn.GELU(), 
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512), 
            nn.GELU(), 
            nn.Dropout(0.2),
            
            nn.Linear(512, 256), 
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128), 
            nn.GELU(),
            
            nn.Linear(128, 1),
            
            nn.Sigmoid()
        )
    def forward(self, x): return self.head(x) * 100.0

class MultiTaskHead(nn.Module):
    """Predicts value and glyph class."""
    def __init__(self, embedding_dim: int, num_classes: int = 100):
        super().__init__()
        self.shared = nn.Linear(embedding_dim, 512)
        self.value_branch = nn.Sequential(            
            nn.Linear(512, 128), 
            nn.ReLU(),
            
            nn.Linear(128, 1),
        )
        self.class_branch = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = F.relu(self.shared(x))
        val = torch.sigmoid(self.value_branch(x)) * 100.0
        logits = self.class_branch(x)
        return val, logits


### main glyph model

class GlyphModel(nn.Module):
    def __init__(self, backbone, value_head, decoder=None, resolution=(224,224), 
                 use_vae=False, latent_dim=128, value_dims=8):
        super().__init__()
        self.backbone = backbone
        self.value_head = value_head
        self.decoder = decoder
        self.use_vae = use_vae
        self.resolution = resolution
        self.latent_dim = latent_dim
        self.value_dims = value_dims

        # auto-detect base grid: 7x7 for 7 / 14 / 28 / 56 / 112 / 224, else 4x4
        self.base_size = 7 if resolution[0] % 7 == 0 else 4

        with torch.no_grad():
            dummy = torch.zeros(1, 3, resolution[0], resolution[1])
            self.feat_dim = backbone(dummy).shape[1]

        if self.use_vae:
            self.fc_mu = nn.Linear(self.feat_dim, latent_dim)
            self.fc_logvar = nn.Linear(self.feat_dim, latent_dim)
            # decoder starts at 256 channels on the base grid
            self.decoder_input = nn.Linear(latent_dim, 256 * self.base_size * self.base_size) 
        else:
            self.neck = nn.Linear(self.feat_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * torch.clamp(logvar, min=-30.0, max=20.0))
        return mu + torch.randn_like(std) * std

    def _format_pred(self, value_pred):
        """Safely formats the prediction if Tensor or multitask tuple"""
        if isinstance(value_pred, tuple):
            val, logits = value_pred
            return (val.squeeze(-1), logits)
        return value_pred.squeeze(-1)

    def forward_with_z(self, z):
        """Generate prediction and image from a latent vector."""
        if not self.use_vae:
            value_pred = self.value_head(z)
            return self._format_pred(value_pred), None, None, None, z
            
        value_pred = self.value_head(z[:, :self.value_dims])
        d_in = self.decoder_input(z)
        d_out = self.decoder(d_in)
        recon_img = F.interpolate(d_out, size=self.resolution, mode='bilinear', align_corners=False)
        return self._format_pred(value_pred), recon_img, None, None, z
    
    def forward(self, x):
        features = self.backbone(x)
        
        if self.use_vae:
            mu, logvar = self.fc_mu(features), self.fc_logvar(features)
            z = self.reparameterize(mu, logvar) if self.training else mu
            value_pred = self.value_head(z[:, :self.value_dims])
            
            # reconstruct
            d_in = self.decoder_input(z)
            d_out = self.decoder(d_in)
            recon_img = F.interpolate(d_out, size=self.resolution, mode='bilinear', align_corners=False)
            return self._format_pred(value_pred), recon_img, mu, logvar, z
        
        z = self.neck(features)
        value_pred = self.value_head(z)
        return self._format_pred(value_pred), None, None, None, z

def build_dynamic_decoder(target_res):

    channel_map = { 7: 256, 14: 256, 28: 128, 56: 64, 112: 32, 224: 16 }

    res = target_res[0]
    base = 7 if res % 7 == 0 else 4
    num_upsamples = int(math.log2(res // base))
    
    # Start at the Base (7x7)
    curr_ch = channel_map.get(base, 256)
    layers = [nn.Unflatten(1, (curr_ch, base, base))]
    
    current_spatial_size = base
    
    for i in range(num_upsamples):
        # determine next spatial size
        next_spatial_size = current_spatial_size * 2
        # determine next channel count from our fixed map
        next_ch = channel_map.get(next_spatial_size, 16)
        
        layers.extend([
            nn.ConvTranspose2d(curr_ch, next_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(next_ch),
            nn.LeakyReLU(0.2)
        ])
        
        curr_ch = next_ch
        current_spatial_size = next_spatial_size
        
    # final layer to RGB
    layers.append(nn.Conv2d(curr_ch, 3, kernel_size=3, padding=1))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

def get_model(cfg, resolution=DEFAULT_CONFIG['resolution']):
    backbone = TIMMBackbone(cfg['backbone']['model_name'], pretrained=True)
    
    use_vae = cfg.get('use_vae', DEFAULT_CONFIG['use_vae'])
    latent_dim = cfg.get('latent_dim', DEFAULT_CONFIG['latent_dim'])
    value_dims = cfg.get('value_dims', DEFAULT_CONFIG['value_dims'])
    
    # VAE uses part of the dimensions for value estimation and full space for reconstruction
    # regression task uses full latent dimension (latent_dim == value_dims)
    head_input_dim = value_dims if use_vae else latent_dim
    
    # dynamic Head creation based on config head type
    match cfg['head']['type']:
        case HeadTypes.DIRECT:
            value_head = RegressionHead(head_input_dim)
            
        case HeadTypes.SIGMOID:
            value_head = RegressionSigmoidHead(head_input_dim)
            
        case HeadTypes.BINNED:
            temp = cfg['head'].get('temperature', 0.75)
            min_v = cfg['head'].get('min_val', -5)
            max_v = cfg['head'].get('max_val', 105)
            value_head = BinnedRegressionHead(head_input_dim, min_val=min_v, max_val=max_v, temperature=temp)
        
        case HeadTypes.CLASSIFICATION:
            num_classes = cfg['head'].get('num_classes', 100)
            value_head = ClassificationHead(head_input_dim, num_classes)
        
        case HeadTypes.MULTITASK:
            value_head = MultiTaskHead(head_input_dim)
        
        case HeadTypes.DEEP:
            value_head = DeepRegressionHead(head_input_dim)
        
        case _:
            value_head = RegressionHead(head_input_dim)
    
    # decoder    
    decoder = build_dynamic_decoder(resolution) if use_vae else None
    
    return GlyphModel(backbone, value_head, decoder, resolution, use_vae, latent_dim, value_dims)
    



### parameter counts

def count_params(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    groups = {
        "ViT Family": [
            'vit_tiny_patch16_224.augreg_in21k',
            'vit_tiny_r_s16_p8_224.augreg_in21k',
            'vit_small_patch16_dinov3.lvd1689m',
            'vit_small_patch16_224.augreg_in21k',
            'vit_base_patch16_224.dino',
            'vit_base_patch16_224.mae',
        ],
        "Hierarchical Transformers": [
            'deit_tiny_patch16_224',
            'swin_tiny_patch4_window7_224',
            'pvt_v2_b0',
            'poolformer_s12',
            
        ],
        "Hybrid Architectures": [
            'maxvit_tiny_tf_224',
            'coatnet_0_rw_224',
        ],
        "CNNs & Mobile Models": [
            'resnet18', 'resnet50', 'efficientnet_b0', 
            'convnext_nano', 'convnext_tiny', 'regnety_004',
            'mobilenetv3_large_100.miil_in21k_ft_in1k',
        ]
    }

    test_cfg = {
        'use_vae': True,
        'latent_dim': 128,
        'value_dims': 8,
        'head': {'type': 'sigmoid_regression'},
        'backbone': {'model_name': ''}
    }

    print(f"{'Group / Backbone':<50} | {'Backbone':<10}| {'Head+Neck':<10}| {'Decoder':<10}| {'Total':<10}")

    for group_name, models in groups.items():
        print(f"\n# {group_name}")
        for model_name in models:
            test_cfg['backbone']['model_name'] = model_name
            try:
                m = get_model(test_cfg, (224, 224))
                # sum all
                backbone_p = count_params(m.backbone)
                head_p = count_params(m.value_head)
                decoder_p = count_params(m.decoder) + count_params(m.decoder_input)
                neck_p = count_params(m.fc_mu) + count_params(m.fc_logvar)
                total_p = backbone_p + head_p + decoder_p + neck_p
                
                print(f"{model_name:<50} & {backbone_p/1e6:>8.2f}M & {(head_p+neck_p)/1e6:>8.2f}M & {decoder_p/1e6:>8.2f}M & {total_p/1e6:>8.2f}M")
            except Exception as e:
                print(f"{model_name:<50} | Error: {str(e)}")