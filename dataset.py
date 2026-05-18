# dataset.py

import os
import zipfile
import string
import json
import io
import random
import glob
import zlib
import numpy as np
from datetime import datetime
import mglyph as mg
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from glyphs import (Glyph, Star, Circle, Square, Blob, VUT, Flower, Ripple, 
                    Flame, Battery, Running, Eye, Target, Eclipse, Fingerprint, 
                    Bucket, Gym, Grin, Starsky, Paw, CloudSun, Hourglass, 
                    Book, Wind, Cookie, Rainbow, Water, Letter)

# definitions
ALL_GLYPHS = [
    "Star", "Square", "Circle", "Flower", "VUT", 
    "Ripple", "Blob", 
    "Flame", "Battery", "Running", "Eye", "Target", "Eclipse", "Fingerprint", "Bucket", 
    "Gym", "Grin", "Starsky", "Paw", "CloudSun", "Hourglass", 
    "Book", "Wind", "Cookie", "Rainbow", "Water", "Letter"
]

# base glyphs
BASE_GLYPHS = [
    "Star", "Square", "Circle", "Flower", "VUT", "Ripple", "Blob", 
]

# test glyphs for ood 
TEST_GLYPHS = [
    "Starsky", "Water", "Grin"
]


GLYPH_NAME_MAP = {g.lower(): g for g in ALL_GLYPHS}

# variations
ALL_STYLES = [
    'default', 
    'size', 'displacement', 'rotation', 
    'border', 'thin', 'thick', 'constant_border', 'random_border',  
    'opacity', 'hue', 'saturation', 'grayscale', 'random_bg', 'random_color',
    'scaled',
    'combined', 
    'variational', 
]

# group definitions for merging
STYLE_GROUPS = {
    "Default":      ['default'],
    "Geometry":     ['size', 'displacement', 'rotation'],
    "Border":       ['border', 'thin', 'thick', 'constant_border', 'random_border'],
    "Color":        ['opacity', 'hue', 'saturation', 'grayscale', 'random_bg', 'random_color'],
    "Scaled":       ['scaled'],
    "Combined":     ['combined'],
    "Variational":  ['variational'],
}

# config
PROJECT_NAME = "Glyph variations"
SIZES = [5, 100, 200]


class DatasetGenerator:
    def __init__(self, filepath: str, dataset_name: str):
        self.filepath = filepath
        self.metadata = {
            "name": dataset_name,
            "created": datetime.now().isoformat(),
            "samples": {},
        }
        self.data = []

    def create(self, glyph_obj: Glyph, style: str, mode: str, draw_args: dict = None, **kwargs):
        draw_args = draw_args or {}
        
        # dynamic method call: e.g. thin_star(), variational_letter()
        method_name = f"{style}_{glyph_obj.__class__.__name__.lower()}"
        
        if not hasattr(glyph_obj, method_name):
            print(f"Warning: {method_name} not found. Skipping.")
            return

        draw_func = getattr(glyph_obj, method_name)(**draw_args)
        
        # generate values
        samples_map = self._generate_values(mode, **kwargs)

        for split, xvalues in samples_map.items():
            if len(xvalues) == 0: continue
            blob = mg.export(
                draw_func, path=None, name="glyph", short_name="glyph", silent=True,
                version="1.0.0", xvalues=xvalues
            )
            self.data.append((split, blob))

        self._save_zip()

    def _generate_values(self, mode: str, **kwargs):
        # 20% test size
        train_n = kwargs.get("train_samples", 1000)
        test_n = kwargs.get("test_samples", int(train_n * 0.2))
        
        def _gap(n):
            res = []
            while len(res) < n:
                c = np.random.uniform(0, 100)
                if not any(s <= c <= e for s, e in gaps): 
                    res.append(c)
            return np.array(res)
        
        def _boundary(boundaries, eps, n_samples):
            if not boundaries: return np.array([])
            res =[]
            samples_per_cluster = n_samples // len(boundaries)
            for b in boundaries:
                res.extend(np.random.uniform(b-eps, b+eps, samples_per_cluster))
            random.shuffle(res)
            return np.array(res)
            

        if mode == "random":
            return {
                "train": np.random.uniform(0, 100, train_n),
                "test": np.random.uniform(0, 100, test_n),
            }
        elif mode == "gapped":
            gaps = kwargs.get("omitted_intervals", [])
            return {
                "train": _gap(train_n), 
                "test": _gap(test_n)
            }
        
        elif mode == "fixed":
            step = kwargs.get("step", 10)
            samples = 100 // step + 1
            return {
                "train": np.linspace(0, 100, samples).tolist() * (train_n // samples), 
                "test": np.linspace(0, 100, samples).tolist() * (test_n // samples)
            }
        
        elif mode == "boundary":
            boundaries = kwargs.get("boundaries", [])
            eps = kwargs.get("eps", 0.5)
            return {
                "train": _boundary(boundaries, eps, train_n),
                "test": _boundary(boundaries, eps, test_n)
            }
        
        return {"train": np.array([]), "test": np.array([])}

    def _save_zip(self):
        folder = os.path.dirname(self.filepath)
        if folder and not os.path.exists(folder): os.makedirs(folder)
        
        with zipfile.ZipFile(self.filepath, "w", zipfile.ZIP_DEFLATED) as zf:
            for idx, (split, blob) in enumerate(self.data):
                with zipfile.ZipFile(blob) as gz:
                    for fname in gz.namelist():
                        data = gz.read(fname)
                        if fname.endswith(".png"):
                            zf.writestr(f"{split}-{idx}-{fname.split('-')[-1]}", data)
                        elif fname.endswith(".json"):
                            meta = json.loads(data.decode())
                            if split not in self.metadata["samples"]: self.metadata["samples"][split] = []
                            self.metadata["samples"][split].extend(
                                [{"value": i[1], "file": f"{split}-{idx}-{i[0]}"} for i in meta["images"]]
                            )
            zf.writestr("dataset.json", json.dumps(self.metadata, indent=2))
        print(f"Created: {self.filepath}")

class MultiDatasetMerger:
    def __init__(self, output_path: str, dataset_name: str):
        self.output_path = output_path
        self.metadata = {
            "name": dataset_name,
            "created": datetime.now().isoformat(),
            "samples": {"train": [], "test": []},
            "composition": []
        }

    def merge_datasets(self, sources: list):
        out_dir = os.path.dirname(self.output_path)
        if out_dir and not os.path.exists(out_dir): os.makedirs(out_dir)

        print(f"Merging {len(sources)} datasets → {self.output_path}")

        with zipfile.ZipFile(self.output_path, "w", compression=zipfile.ZIP_DEFLATED) as out_zip:
            for src_path in sources:
                if not os.path.exists(src_path): continue
                
                # filename as unique prefix
                prefix = os.path.basename(src_path).replace(".zip", "")
                self.metadata["composition"].append(prefix)

                with zipfile.ZipFile(src_path, "r") as src_zip:
                    try:
                        src_meta = json.load(src_zip.open("dataset.json"))
                    except KeyError: continue

                    for split in ["train", "test"]:
                        if split not in src_meta["samples"]: continue
                        for sample in src_meta["samples"][split]:
                            orig_file = sample["file"]
                            new_file = f"{prefix}_{orig_file}"
                            try:
                                img_data = src_zip.read(orig_file)
                                out_zip.writestr(new_file, img_data)
                                new_sample = sample.copy()
                                new_sample["file"] = new_file
                                new_sample["source"] = prefix
                                self.metadata["samples"][split].append(new_sample)
                            except KeyError: pass

            out_zip.writestr("dataset.json", json.dumps(self.metadata, indent=2))

class GlyphDataset(Dataset):
    def __init__(self, zip_path, split="train", resize=(64, 64), augment=False, is_cached=True):
        self.zip_path = zip_path
        self.resize = resize
        self.augment = augment
        self.zf = None 
        self.cache = []
        self.is_cached = is_cached
        
        # load the JSON metadata into RAM
        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open('dataset.json') as f:
                metadata = json.load(f)
                
        # depending on JSON structure, extract the list of files
        if split == "all":
            train_s = metadata.get("samples", {}).get("train", [])
            test_s = metadata.get("samples", {}).get("test",[])
            self.samples = train_s + test_s
        else:
            self.samples = metadata.get("samples", {}).get(split,[])
        
        # possibly add augmentation
        self.transform = self._build_transforms()
        
        # takes exactly the size of the ZIP file
        if self.is_cached:
            print(f"Loading {len(self.samples)} compressed images into RAM...")
            with zipfile.ZipFile(self.zip_path, 'r') as zf:
                for sample in tqdm(self.samples, desc=f"Caching {split} to RAM", leave=True):
                    try:
                        self.cache.append(zf.read(sample['file']))
                    except Exception as e:
                        print(f"Failed caching {sample['file']}: {e}")
                        self.cache.append(None)
    
    def _build_transforms(self):
        """
        Dynamically builds the transform pipeline.
        `self.augment` can be:
          - bool (True defaults all to level 1)
          - int (1, 2, or 3 sets all categories to that level)
          - dict (determines rate 1-3 for each category, e.g. {'geometric': 1, 'blur': 3})
        """
        base_transforms = [transforms.Resize(self.resize)]
        
        if not self.augment:
            base_transforms.append(transforms.ToTensor())
            return transforms.Compose(base_transforms)
            
        # parse augmentation configuration
        config = {'geometric': 1, 'photometric': 1, 'blur': 1, 'noise': 1}
        
        if isinstance(self.augment, int) and not isinstance(self.augment, bool):
            lvl = max(1, min(3, self.augment))
            config = {k: lvl for k in config}
        elif isinstance(self.augment, dict):
            config.update(self.augment)
            
        # geometric transformations (Flips, Rotations, Translations, Scales)
        lvl_g = config.get('geometric', 0)
        if lvl_g == 1:
            base_transforms.append(transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), fill=255))
        elif lvl_g == 2:
            base_transforms.append(transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.96, 1.04), fill=255))
        elif lvl_g >= 3:
            base_transforms.append(transforms.RandomAffine(degrees=15, translate=(0.06, 0.06), scale=(0.90, 1.10), fill=255))
            base_transforms.append(transforms.RandomHorizontalFlip(p=0.3)) 
            
        # photometric Distortions (Brightness, Contrast, Saturation)
        lvl_p = config.get('photometric', 0)
        if lvl_p == 1:
            base_transforms.append(transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05))
        elif lvl_p == 2:
            base_transforms.append(transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02))
        elif lvl_p >= 3:
            base_transforms.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05))

        # blurry edges
        lvl_b = config.get('blur', 0)
        if lvl_b == 1:
            base_transforms.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.2))
        elif lvl_b == 2:
            base_transforms.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 1.0))], p=0.4))
        elif lvl_b >= 3:
            base_transforms.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 1.5))], p=0.6))

        # core tensor transformation
        base_transforms.append(transforms.ToTensor())

        # noise Injection 
        lvl_n = config.get('noise', 0)
        if lvl_n == 1:
            base_transforms.append(AddGaussianNoise(std=0.01))
        elif lvl_n == 2:
            base_transforms.append(AddGaussianNoise(std=0.03))
        elif lvl_n >= 3:
            base_transforms.append(AddGaussianNoise(std=0.06))

        return transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):        
        sample = self.samples[idx]
        source = sample.get("source", "unknown")
        
        if source == "unknown" and self.zip_path:
            source = os.path.basename(self.zip_path).replace('.zip', '')

        # handle class or char type            
        char = sample.get("char")
        if char is None:
            parts = source.split('_')
            if parts[0] in['glyph', 'letter', 'number', 'temp'] and len(parts) > 1:
                if parts[0] == 'temp' and len(parts) > 2:
                    char = parts[2]
                else:
                    char = parts[1]
            else:
                char = parts[0]            
            char = GLYPH_NAME_MAP.get(char.lower(), char)
        
        # cache datasets
        try:
            if self.is_cached:
                img_bytes = self.cache[idx]
                if img_bytes is None: raise ValueError("Corrupt cached image")
            else:
                if self.zf is None:
                    self.zf = zipfile.ZipFile(self.zip_path, 'r')
                img_bytes = self.zf.read(sample['file'])
            
            # decode the PNG/JPEG from the RAM bytes on the fly
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_tensor = self.transform(img)
            
            return img_tensor, sample['value'], char
            
        except Exception as e:
            print(f"\nCorrupted image: {sample['file']}")
            safe_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(safe_idx)

class AddGaussianNoise(object):
    """Custom transform to inject Gaussian noise into image tensors."""
    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, tensor):
        # add noise and clamp values to valid image range [0.0, 1.0]
        noise = torch.randn(tensor.size()) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)

