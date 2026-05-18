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

from visualize import VisualizeDataset
from glyphs import (Glyph, Star, Circle, Square, Blob, VUT, Flower, Ripple, 
                    Flame, Battery, Running, Eye, Target, Eclipse, Fingerprint, 
                    Bucket, Gym, Grin, Starsky, Paw, CloudSun, Hourglass, 
                    Book, Wind, Cookie, Rainbow, Water, Letter)
from manager import ExperimentManager 

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




# visualize functions    

def visualize_augmentation():
    auto_dataset(VUT)
    viz = VisualizeDataset(plots_dir="augmentation", title_prefix="Augmentation")
    viz.show_augmentation_levels("data/vut.zip", GlyphDataset, count=5)

def visualize_inline(zip_path, manager: ExperimentManager):
    if not os.path.exists(zip_path): 
        return

    logger = None
    if manager and manager.active:
        logger = manager.task.get_logger()

    viz = VisualizeDataset(plots_dir="data_visualization", logger=logger, task_type="dataset_check")
    
    filename = os.path.basename(zip_path).replace(".zip", "")
    title = filename
    parts = filename.split('_')
    if len(parts) > 1:
        title = f"{parts[0].capitalize()} ({', '.join(parts[1:-1])}) - Size {parts[-1]}"
    
    try:
        viz.show_samples(zip_path, custom_title=title)
    except Exception as e:
        print(f"Visualization failed for {zip_path}: {e}")



# ensure that dataset exists if not create one

def auto_dataset(glyph_class, style="default", mode="random", train_samples=1000, test_samples=None, **kwargs):
    glyph_name = glyph_class.__name__.lower()
    
    mode_str = ""
    if mode == "gapped" and "omitted_intervals" in kwargs:
        gaps = "_".join([f"{s}-{e}" for s, e in kwargs["omitted_intervals"]])
        mode_str = f"_gapped_{gaps}"
    elif mode != "random":
        mode_str = f"_{mode}"
        
    kw_str = ""
    if kwargs:
        clean_kwargs = {k: v for k, v in kwargs.items() if k != "omitted_intervals"}
        if clean_kwargs:
            kw_str = "_" + "_".join([f"{k}-{v}" for k, v in clean_kwargs.items()])
            kw_str = kw_str.replace("#", "").replace(" ", "")

    folder = f"data_{glyph_name}"
    filename = f"{glyph_name}_{style}{mode_str}{kw_str}_{train_samples}.zip"
    filepath = os.path.join(folder, filename)
    
    if test_samples is None:
        test_samples = int(train_samples * 0.2)
        
    ensure_dataset(
        filename=filepath, 
        glyph_class=glyph_class, 
        style=style, 
        mode=mode, 
        train_samples=train_samples, 
        test_samples=test_samples,
        visualize=False, 
        **kwargs
    )
    return filepath


def ensure_dataset(filename, glyph_class, style, mode, train_samples=1000, manager: ExperimentManager = None, visualize: bool = True, **kwargs):
    if os.path.exists(filename): 
        return filename
    
    # check cloud
    if manager:
        cloud_path = manager.get_dataset_path(os.path.basename(filename), filename)
        if cloud_path != filename and os.path.exists(cloud_path): 
            return cloud_path

    # generate
    gen = DatasetGenerator(filepath=filename, dataset_name=f"{style}-{mode}-{train_samples}")
    gen.create(glyph_class(), style=style, mode=mode, train_samples=train_samples, **kwargs)
    
    # upload to ClearML
    if manager: 
        manager.upload_dataset(os.path.basename(filename), filename)
    
    # visualize
    if visualize:
        visualize_inline(filename, manager)
    
    return filename



# base glyphs

def generate_base_datasets(manager: ExperimentManager = None):
    """Generate all Glyphs with all variations"""
    for size in SIZES:
        for glyph_name in ALL_GLYPHS:
            for style in ALL_STYLES:
                g_lower = glyph_name.lower()
                filename = f"data_{g_lower}/{g_lower}_{style}_{size}.zip"
                
                # ensure_dataset now handles the visualization call internally
                ensure_dataset(
                    filename=filename,
                    glyph_class=globals()[glyph_name],
                    style=style,
                    mode='random',
                    train_samples=size,
                    manager=manager
                )
                
    if manager: manager.close() 
    

# letters
    
def generate_letter_datasets(manager: ExperimentManager = None):
    """Generate all letters and different splits"""
    # individual letters
    for char in string.ascii_letters:
        for size in SIZES:
            ensure_dataset(
                filename=f"data_letters/letter_{char}_{size}.zip",
                glyph_class=Letter,
                style="default",
                mode="random",
                draw_args={'char': char},
                train_samples=size,
                manager=manager
            )

    # splits
    for chars in [string.ascii_lowercase, string.ascii_uppercase]:
        all_chars = list(chars)
        merge_letters(all_chars[:-1], manager) # A-Y
        merge_letters(all_chars[:-2], manager) # A-X
        merge_letters(all_chars[:-3], manager) # A-W
        merge_letters(all_chars[:-4], manager) # A-V
        merge_letters(all_chars[:-5], manager) # A-U
        
        merge_letters(all_chars[-2:], manager) # Y-Z
        merge_letters(all_chars[-3:], manager) # X-Z
        merge_letters(all_chars[-4:], manager) # W-Z
        merge_letters(all_chars[-5:], manager) # V-Z
        
        merge_letters(all_chars[:13], manager) # A-M
        merge_letters(all_chars[13:], manager) # N-Z
        
        merge_letters(all_chars, manager)      # A-Z
    
    all_chars = list(string.ascii_letters) # a-Z
    merge_letters(all_chars, manager) # A-Z
    
    # words
    for size in SIZES:
        for rotation in [False, True, 90]:
            for word in ['alice', 'bob', 'longSequenceOfLetters']:
                name = ''
                if rotation:
                    if rotation == 90:
                        name = 'rot_90_'
                    else:
                        name = 'scaled_rot_'
                
                filename = f"data_strings/{'_'.join(word.split())}_{name}{size}.zip"
                ensure_dataset(filename, Letter, 'default', 'random', draw_args={'char': f"{word}", 'rotation': rotation}, train_samples=size, manager=manager)
    

def merge_letters(letters: list = None, manager: ExperimentManager = None):
    if letters is None:
        letters = ["A", "B"]
        
    output_dir = "data_merged_letters"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_char = letters[0]
    end_char = letters[-1]
    # name for the group (e.g., "A_M" or "A_Z")
    group_name = f"letters_{start_char}_{end_char}"

    for size in SIZES:
        sources = []
        for char in letters:
            src_path = f"data_letters/letter_{char}_{size}.zip"
            if os.path.exists(src_path):
                sources.append(src_path)
            else:
                print(f"Warning: Missing source {src_path}")

        if not sources:
            print(f"No sources found for size {size}. Skipping.")
            continue

        target = f"{output_dir}/{group_name}_{size}.zip"
        
        # merge
        if not os.path.exists(target):
            MultiDatasetMerger(target, f"{group_name}_{size}").merge_datasets(sources)
            if manager: 
                manager.upload_dataset(os.path.basename(target), target)
                visualize_inline(target, manager)



# numbers

def generate_number_datasets(manager: ExperimentManager = None):
    """Generates datasets for numbers 0-9 and 0-100 for 100 and 1000 samples"""
    output_dir = "data_numbers"
    output_dir_merged = "data_merged_numbers"
    
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    if not os.path.exists(output_dir_merged):
        os.makedirs(output_dir_merged)
        
    for size in SIZES:
        sources_0_100 = []
        
        # generate individual numbers from 0 to 100
        for i in range(101):
            filepath = f"{output_dir}/number_{i}_{size}.zip"
            ensure_dataset(
                filename=filepath,
                glyph_class=Letter,
                style="default",
                mode="random",
                train_samples=size,
                draw_args={'char': str(i)}, 
                manager=manager,
                visualize=False
            )
            sources_0_100.append(filepath)
        
        # merge 0-9
        sources_0_9 = sources_0_100[:10]
        target_0_9 = f"{output_dir_merged}/numbers_0-9_{size}.zip"
        if not os.path.exists(target_0_9):
            MultiDatasetMerger(target_0_9, f"Numbers_0-9_{size}").merge_datasets(sources_0_9)
            if manager: manager.upload_dataset(os.path.basename(target_0_9), target_0_9)

        # merge 0-100
        target_0_100 = f"{output_dir_merged}/numbers_0-100_{size}.zip"
        if not os.path.exists(target_0_100):
            MultiDatasetMerger(target_0_100, f"Numbers_0-100_{size}").merge_datasets(sources_0_100)
            if manager: manager.upload_dataset(os.path.basename(target_0_100), target_0_100)



# ood

def generate_ood_datasets(manager: ExperimentManager = None):
    """Generates Out-of-Distribution (OOD) interpolation and extrapolation datasets"""
    ood_glyphs =[Star, Circle, Square, Blob, VUT, Flower, Ripple, 
                Flame, Battery, Running, Eye, Target, Eclipse, Fingerprint, 
                Bucket, Gym, Grin, Starsky, Paw, CloudSun, Hourglass, 
                Book, Wind, Cookie, Rainbow, Water, Letter]
    
    # Define the OOD scenarios
    scenarios = {
        # Interpolation Gap: Train missing middle, Test ONLY middle
        "interp_train": {"mode": "gapped", "omitted_intervals": [(30, 70)]},
        "interp_test":  {"mode": "gapped", "omitted_intervals": [(0, 30), (70, 100)]},
        
        # Extrapolation Gap: Train missing tails, Test ONLY tails
        "extrap_train": {"mode": "gapped", "omitted_intervals": [(0, 20), (80, 100)]},
        "extrap_test":  {"mode": "gapped", "omitted_intervals": [(20, 80)]}
    }

    out_dir = "data_ood"
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    for size in SIZES:
        for glyph_class in ood_glyphs:
            g_name = glyph_class.__name__.lower()
            
            # Handle standard shapes
            if g_name != 'letter':
                for s_name, kwargs in scenarios.items():
                    ensure_dataset(
                        filename=f"{out_dir}/{g_name}_ood_{s_name}_{size}.zip",
                        glyph_class=glyph_class, style="default", 
                        train_samples=size, manager=manager, visualize=False, **kwargs
                    )
            
            # Handle specific letter sets
            else:
                letter_cases = {
                    "lower": list(string.ascii_lowercase),
                    "upper": list(string.ascii_uppercase),
                    "all": list(string.ascii_letters)
                }
                
                for case_name, chars in letter_cases.items():
                    for s_name, kwargs in scenarios.items():
                        
                        # Generate individual letters
                        sources =[]
                        for char in chars:
                            char_file = f"{out_dir}/temp_{case_name}_{char}_{s_name}_{size}.zip"
                            ensure_dataset(
                                filename=char_file, glyph_class=Letter, style="default",
                                draw_args={'char': char}, train_samples=size, 
                                manager=manager, visualize=False, **kwargs
                            )
                            sources.append(char_file)
                        
                        # Merge the letters into a single case dataset
                        target = f"{out_dir}/letters_{case_name}_ood_{s_name}_{size}.zip"
                        MultiDatasetMerger(target, f"OOD_{case_name}_{s_name}").merge_datasets(sources)
                        
                        # Clean up temp files
                        for src in sources:
                            if os.path.exists(src): os.remove(src)



# styles

def generate_style_glyphs(sizes = SIZES, styles = ALL_STYLES, glyphs = ALL_GLYPHS, iterations=20, manager: ExperimentManager = None):
    for size in sizes:
        for style in styles:
            for glyph in glyphs:    
                for i in range(iterations):
                    filename = f"data_style_{style}/{iterations}x{size}/{glyph.lower()}_{style}_{i+1}_{size}.zip"
                    ensure_dataset(filename, globals()[glyph], style, 'random', train_samples=size, visualize=False, manager=manager)

def merge_by_style(sizes = SIZES, styles = ALL_STYLES, glyphs = ALL_GLYPHS, iterations=20, manager: ExperimentManager = None):
    output_dir = "data_merged_styles"
    
    for size in sizes:
        for style in styles:
            for glyph in glyphs:
                glyph_name = glyph.lower()
                sources = [f"data_style_{style}/{iterations}x{size}/{glyph_name}_{style}_{i+1}_{size}.zip" for i in range(iterations)]
                target = f"{output_dir}/all_{glyph_name}_{style}_{iterations}x{size}.zip"
                
                if not os.path.exists(target):
                    MultiDatasetMerger(target, f"All_{glyph}_{style}_{iterations}x{size}").merge_datasets(sources)
                    if manager: 
                        manager.upload_dataset(os.path.basename(target), target)
                    visualize_inline(target, manager)


def merge_by_style_across_glyphs(sizes = SIZES, styles = ALL_STYLES, glyphs = ALL_GLYPHS, manager: ExperimentManager = None):
    output_dir = "data_merged_styles"
    
    for size in sizes:
        for style in styles:
            sources = [f"data_{g.lower()}/{g.lower()}_{style}_{size}.zip" for g in glyphs]
            target = f"{output_dir}/all_{style}_{size}.zip"
            
            if not os.path.exists(target):
                MultiDatasetMerger(target, f"All_{style}_{size}").merge_datasets(sources)
                if manager: 
                    manager.upload_dataset(os.path.basename(target), target)
                visualize_inline(target, manager)



# groups

def generate_grouped_datasets(manager: ExperimentManager = None):
    """Creates grouped datasets strictly according to STYLE_GROUPS definitions"""
    output_dir = "data_groups"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    for size in SIZES:
        group_paths = {}
        
        # base Groups defined in STYLE_GROUPS
        for group_name, style_list in STYLE_GROUPS.items():
            sources = []
            for style in style_list:
                for glyph in ALL_GLYPHS:
                    # look for the individually generated style files
                    src = f"data_{glyph.lower()}/{glyph.lower()}_{style}_{size}.zip"
                    if os.path.exists(src):
                        sources.append(src)
                        
            target = f"{output_dir}/{group_name}_{size}.zip"
            if sources and not os.path.exists(target):
                MultiDatasetMerger(target, group_name).merge_datasets(sources)
                if manager: manager.upload_dataset(os.path.basename(target), target)
                visualize_inline(target, manager)
            
            group_paths[group_name] = target
            


# style


def generate_style_datasets(manager: ExperimentManager = None):
    """Creates merged datasets for each style containing all glyphs, saved to data_styles."""
    output_dir = "data_styles"
    
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir, exist_ok=True)

    for size in SIZES:
        for style in ALL_STYLES:
            sources =[]
            
            for glyph_name in ALL_GLYPHS:
                g_lower = glyph_name.lower()
                src_path = f"data_{g_lower}/{g_lower}_{style}_{size}.zip"
                
                ensure_dataset(
                    filename=src_path,
                    glyph_class=globals()[glyph_name],
                    style=style,
                    mode='random',
                    train_samples=size,
                    manager=manager,
                    visualize=False
                )
                sources.append(src_path)
            
            target = f"{output_dir}/all_glyphs_{style}_{size}.zip"
            if not os.path.exists(target):
                MultiDatasetMerger(target, f"All_Glyphs_{style}_{size}").merge_datasets(sources)
                if manager: 
                    manager.upload_dataset(os.path.basename(target), target)
                visualize_inline(target, manager)


# meging functions

def merge_everything_together(size, glyph = "", manager: ExperimentManager = None, test = ""):
    output_dir = "data_merged_glyphs" + test
    glyph = glyph.lower()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # all zip files in that directory
    all_sources = glob.glob(f"data_merged_styles/all_{glyph}_*x{size}.zip") 

    # merge All by glyph
    if all_sources:
        final = glyph if glyph != '' else 'all'
        target = f"{output_dir}/glyph_{final}_{size}.zip"
        MultiDatasetMerger(target, f"Merged {final} dataset").merge_datasets(all_sources)
        if manager:
            manager.upload_dataset(os.path.basename(target), target)
            
def merge_final(size, manager: ExperimentManager = None, test=""):
    output_dir = "data_merged_glyphs" + test
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # all zip files in that directory
    all_sources = glob.glob(f"{output_dir}/glyph_*_{size}.zip") 
    all_sources = [src for src in all_sources if not src.endswith(f"glyph_all_{size}.zip")]

    # merge all by glyph
    if all_sources:
        target = f"{output_dir}/glyph_all_{size}.zip"
        MultiDatasetMerger(target, f"Merged all datasets").merge_datasets(all_sources)
        if manager:
            manager.upload_dataset(os.path.basename(target), target)



# generate variety of different iterations of glyphs by importance not to overfit eg. on color

def create_train_datasets(manager: ExperimentManager = None):
    generate_style_glyphs(SIZES, ['default', 'size', 'hue', 'rotation'], ALL_GLYPHS, iterations=1)
    generate_style_glyphs(SIZES, ['saturation', 'displacement', 'opacity', 'border', 'constant_border'], ALL_GLYPHS, iterations=3)
    generate_style_glyphs(SIZES, ['grayscale', 'random_border'], ALL_GLYPHS, iterations=5)
    generate_style_glyphs(SIZES, ['variational', 'combined', 'scaled', 'thin', 'thick'], ALL_GLYPHS, iterations=10)
    generate_style_glyphs(SIZES, ['random_color', 'random_bg'], ALL_GLYPHS, iterations=20)
    
    merge_by_style(SIZES, ['default', 'size', 'hue', 'rotation'], ALL_GLYPHS, iterations=1)
    merge_by_style(SIZES, ['saturation', 'displacement', 'opacity', 'border', 'constant_border'], ALL_GLYPHS, iterations=3)
    merge_by_style(SIZES, ['grayscale', 'random_border'], ALL_GLYPHS, iterations=5)
    merge_by_style(SIZES, ['variational', 'combined', 'scaled', 'thin', 'thick'], ALL_GLYPHS, iterations=10)
    merge_by_style(SIZES, ['random_color', 'random_bg'], ALL_GLYPHS, iterations=20)
    
    for s in SIZES:
        for g in ALL_GLYPHS:
            merge_everything_together(s, g, manager=manager)
        merge_final(s)

def create_test_datasets(manager: ExperimentManager = None):
    generate_style_glyphs([5], iterations=1)
    merge_by_style([5], iterations=1)
    
    for g in ALL_GLYPHS:
        merge_everything_together(5, g, manager=manager, test="_test")
    merge_final(5, test="_test")
        

    
# one dataset

def generate_one_dataset():
    manager = ExperimentManager(PROJECT_NAME, "random border", tags=["trash"])
    for size in [200]:
        for glyph_name in ["Circle"]:
            for style in ['default']:
                g_lower = glyph_name.lower()
                filename = f"data_{g_lower}/{g_lower}_{style}_{size}.zip"
                
                # ensure_dataset now handles the visualization call internally
                ensure_dataset(
                    filename=filename,
                    glyph_class=globals()[glyph_name],
                    style=style,
                    mode='random',
                    train_samples=size,
                    manager=manager
                )
    manager.close()

    


# all datasets

def generate_all_datasets(manager: ExperimentManager = None):
    # note that this run creates around 60 GB of data
    generate_base_datasets(manager)
    generate_letter_datasets(manager)
    generate_number_datasets(manager)
    generate_ood_datasets(manager)
    generate_grouped_datasets(manager)
    generate_style_datasets(manager)
    create_train_datasets()
    create_test_datasets()


if __name__ == "__main__":
    # generate_all_datasets()
    generate_style_datasets()
    
    
    # generate datasets for ploting a distribution of the data
    
    # kwargs = { 'boundaries': [20, 30, 55, 75, 90], 'eps': 1 }
    # ensure_dataset(filename="data/stars_boundary_2k.zip", glyph_class=Star, style="default", mode='boundary', train_samples=2000,**kwargs)
    
    # kwargs = { 'omitted_intervals': [(25,75)] }
    # ensure_dataset(filename="data/stars_extrapolate_2k.zip", glyph_class=Star, style="default", mode='gapped', train_samples=2000,**kwargs)
    
    # kwargs = { 'omitted_intervals': [(0,25),(75,100)] }
    # ensure_dataset(filename="data/stars_interpolate_2k.zip", glyph_class=Star, style="default", mode='gapped', train_samples=2000,**kwargs)
    
    # ensure_dataset(filename="data/stars_fixed_2k.zip", glyph_class=Star, style="default", mode='fixed', train_samples=2000)
    
    # ensure_dataset(filename="data/stars_random_2k.zip", glyph_class=Star, style="default", mode='random', train_samples=2000)
    
    pass
