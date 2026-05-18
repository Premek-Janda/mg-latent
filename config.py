import torch
import copy
import collections.abc

### ClearML project settings
PROJECT_NAME = "Glyph Research"
USE_CLEARML = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Architecture settings
class HeadTypes:
    DIRECT = "direct_regression"
    SIGMOID = "sigmoid_regression"
    BINNED = "binned_regression"
    CLASSIFICATION = "classification"
    MULTITASK = "multitask"
    DEEP = "deep_regression"

RESOLUTION = (56, 56)
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 15
WORST_SAMPLES_COUNT = 10

DEFAULT_CONFIG = {
    # general config
    'resolution': (112, 112),
    'batch_size': 256,
    'lr': 1e-3,
    'augment': False,
    'custom_weights': None,
    
    # VAE specific
    'use_vae': True,
    'latent_dim': 64,
    'value_dims': 4,
    'vae_alpha': 8.785,
    'vae_beta': 0.064,
    'vae_gamma': 0.012,
    
    # backbone
    'freeze_backbone': False,
    'backbone': {
        'name': 'timm',
        'model_name': 'resnet18',
        'pretrained': True
    },
    
    # binned head
    # 'head': {
    #     'type': 'binned_regression',
    #     'classes': [],
    #     'min_val': -10,
    #     'max_val': 110,
    #     'temperature': 0.35,
    # }
    
    # multitask head
    'head': {
        'type': 'multitask',
        'cls_weight': 10.0,
    }
}

DEFAULT_HISTORY = {
    # train loss
    'train_total_loss': [], 'val_total_loss': [],
    'train_task_loss': [], 'val_task_loss': [],
    # original vae losses
    'train_recon_loss': [], 'val_recon_loss': [],
    'train_kl_loss': [], 'val_kl_loss': [],
    'train_ortho_loss': [], 'val_ortho_loss': [],
    # weighted vae losses
    'train_w_recon_loss': [], 'val_w_recon_loss': [],
    'train_w_kl_loss': [], 'val_w_kl_loss': [],
    'train_w_ortho_loss': [], 'val_w_ortho_loss': [],
    # multihead losses
    'train_val_task_loss': [], 'val_val_task_loss': [],
    'train_cls_task_loss': [], 'val_cls_task_loss': [],
    'train_w_cls_task_loss': [], 'val_w_cls_task_loss': [],
    # metric
    'train_metric': [], 'val_metric': [],
    # train targets
    'all_targets': []
}


def merge_configs(default_dict, user_dict):
    """Deep merges user_dict into default_dict"""
    d = copy.deepcopy(default_dict)
    for k, v in user_dict.items():
        if isinstance(v, collections.abc.Mapping):
            target = d.get(k, {})
            if not isinstance(target, collections.abc.Mapping):
                target = {}
            d[k] = merge_configs(target, v)
        else:
            d[k] = v
    return d


# ploting 

RC_PARAMS = {
    "text.usetex": False,
    "font.family": "serif",
    'font.size': 13,
    "mathtext.fontset": "cm",
    'figure.dpi': 300.0,
    'axes.titlepad': 20.0,
    'axes.titlesize': 18,
    'axes.titlecolor': 'auto',
    'axes.labelsize': 13,
    'axes.edgecolor': '#333333',
    'axes.spines.bottom': True,
    'axes.spines.left': True,
    'axes.spines.right': True,
    'axes.spines.top': True,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'legend.facecolor': '#eeeeee',
    'legend.fancybox': True,
    'legend.edgecolor': "0.4",
    'legend.framealpha': 0.75,
    'legend.frameon': True,
    'axes.grid': True,
    'axes.grid.which': 'both',
    'axes.grid.axis': 'both',
    'grid.alpha': 1.0,
    'grid.color': '#b0b0b0',
    'grid.linestyle': '-',
    'grid.linewidth': 0.7,
}

RC_MARKERS = [".",",","o","v","^","<",">","8","s","p","P","*","h","H","X","D","d"]
