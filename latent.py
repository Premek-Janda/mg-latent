import torch
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from torch.utils.data import DataLoader

class LatentManifold:
    def __init__(self, vae_model, device, resolution):
        self.vae = vae_model
        self.device = device
        self.resolution = resolution

    def extract_latents(self, dataset):
        """Passes a dataset through the VAE and extracts all latent vectors."""
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        all_z, all_true, all_labels = [], [], []
        
        with torch.no_grad():
            for images, values, labels in loader:
                # Forward pass
                _, _, _, _, z = self.vae(images.to(self.device))
                all_z.append(z.cpu().numpy())
                all_true.extend(values.numpy())
                # Handle possible empty labels or 'unknown' gracefully
                all_labels.extend([str(l) if l and l != 'unknown' else 'default_label' for l in labels])
                
        return np.concatenate(all_z, axis=0), np.array(all_true), np.array(all_labels)

    def build_splines(self, all_z, all_true, all_labels):
        """Creates smooth B-splines mapping 'scale value' -> 'Latent Vector' per class."""
        splines = {}
        unique_labels = sorted(list(set(all_labels)))
        
        for label in unique_labels:
            mask = (all_labels == label)
            Z_c, V_c = all_z[mask], all_true[mask]
            
            df = pd.DataFrame(Z_c)
            df['val'] = V_c
            grouped = df.groupby('val').mean().sort_index()
            u_vals, m_zs = grouped.index.values, grouped.values
            
            if len(u_vals) > 2:
                k = min(len(u_vals) - 1, 2)
                spl_model = make_interp_spline(u_vals, m_zs, k=k)
                x_min, x_max = u_vals.min(), u_vals.max()
                splines[label] = lambda v, s=spl_model, mn=x_min, mx=x_max: s(np.clip(v, mn, mx))
            else:
                splines[label] = lambda v, mz=m_zs[0]: mz
                
        return splines, unique_labels

    @staticmethod
    def slerp(val, low, high):
        """Spherical Linear Interpolation for properly traversing a VAE Gaussian prior space."""
        low_norm = low / np.linalg.norm(low)
        high_norm = high / np.linalg.norm(high)
        omega = np.arccos(np.clip(np.dot(low_norm, high_norm), -1, 1))
        so = np.sin(omega)
        
        if so == 0:
            return (1.0 - val) * low + val * high
            
        return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high