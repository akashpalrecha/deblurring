from pdb import set_trace
import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import itertools
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import kornia
import kornia.augmentation as K
import pytorch_lightning as pl
import os
import mimetypes

try:
    get_ipython().__class__.__name__
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm

class DeblurDataset(Dataset):
    def __init__(self, files, size=None, stats=None):
        self.files    = files
        self.toTensor = transforms.ToTensor()
        self.stats = stats
        if self.stats is not None:
            self.normalize  = kornia.augmentation.Normalize(*self.stats)
            self.denormaize = kornia.augmentation.Denormalize(*self.stats)
        else:
            self.normalize  = Noop()
            self.denormaize = Noop()
        # set_trace()
        if size == "auto": # Do not resize for test images
            self.size = size
            self.im_reader = self.readTestImage
        else: 
            self.im_reader = self.readImage
            if size is None:
                size = tuple(reversed(Image.open(str(self.files[0][0])).size))
            self.size = tuple(reversed(size))
    
    def __len__(self):
        return len(self.files)
    
    def readTestImage(self, path:str):
        return self.toTensor(Image.open(str(path)))
    
    def readImage(self, path:str):
        return self.toTensor(Image.open(str(path)).resize(self.size))
    
    def __getitem__(self, idx):
        blurred = self.im_reader(self.files[idx][0])
        sharp   = self.im_reader(self.files[idx][1])
        return self.normalize(blurred), self.normalize(sharp)


class DeblurDataModule(pl.LightningDataModule):
    def __init__(self, train:Path, test:Path=None, batch_size=1, valid_batch_size=1, transforms=None, stats=None, name='GOPRO', 
                 size=(256,256), train_files=None, valid_files=None, val_pct=0.2):
        super(DeblurDataModule, self).__init__()
        self.train      = train
        self.test       = test
        self.batch_size = batch_size
        self.val_pct    = val_pct
        self.name       = name
        # Set valid_batch_size to batch_size if not provided explicitly
        self.valid_batch_size = self.batch_size if valid_batch_size == 0 else valid_batch_size
        
        if train_files is not None:
            if valid_files is None: 
                raise Exception("Please provide the valid_files argument")
            else:
                self.train_files, self.valid_files = train_files, valid_files
        else:
            self.train_files, self.valid_files = self.get_dataset_files(self.train, self.test)
        
        # Setting size properly
        if size == 0: 
            # This means we take the native resolution of input images
            # But we still add a resize transform in the dataset 
            # in case a few random images have slightly different sizes
            width, height = Image.open(str(self.train_files[0][0])).size
            size = (height, width)
        self.transforms       = make_listy(transforms) if transforms is not None else self.get_transforms()[0]
        self.valid_transforms = self.get_transforms()[1]
        self.size  = size
        self.stats = stats
        
        # Adding Transform for Normalizing Data
        tmp_ds = DeblurDataset(self.train_files, self.size)
        if self.stats is None:
            means = torch.zeros_like(tmp_ds[0][0].mean((1,2)))
            stds  = torch.zeros_like(tmp_ds[0][0].std((1,2)))
            
            factor = 0.5
            print(f"Calculating dataset stats with a {int(factor * 100)}% random subset of training data")
            for i in tqdm(torch.randint(0, len(tmp_ds), (int(factor*len(tmp_ds)),))):
                im = tmp_ds[i][0]
                means += im.mean((1,2))
                stds  += im.std((1,2))
                
            means /= factor * len(tmp_ds)
            stds  /= factor * len(tmp_ds)
            self.stats = (means, stds)
        elif self.stats == "auto":
            means = torch.zeros_like(tmp_ds[0][0].mean((1,2))).float()
            stds  = torch.ones_like(tmp_ds[0][0].std((1,2))).float()
            self.stats = (means, stds)
        else: pass
        
        self.setup()
        self.normalize_func   = self.train_ds.normalize
        self.denormalize_func = self.train_ds.denormaize
        
    def prepare_data(self):
        pass

    def setup(self, stage:str = None):
        if stage == 'fit' or stage is None:
            self.train_ds = DeblurDataset(self.train_files, self.size, self.stats)
            self.valid_ds = DeblurDataset(self.valid_files, self.size, self.stats)
            self.dims = self.train_ds[0][0].shape
            
        if stage == 'test' or stage is None:
            self.test_ds  = DeblurDataset(self.valid_files, size='auto', stats=self.stats)
            self.dims     = getattr(self, 'dims', self.test_ds[0][0].shape)
            
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.valid_batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    def test_dataloader(self):
        return self.val_dataloader()
        # return DataLoader(self.valid_ds, batch_size=self.valid_batch_size, shuffle=False, num_workers=8, pin_memory=True)
            
    def get_dataset_files(self, train:Path, test:Path=None):
        train = Path(train)
        files = get_image_files(train)
        if test is not None:
            test = Path(test)
            test_files = get_image_files(test)
            files = itertools.chain(files, test_files)
        
        files       = list(files)
        train_files =        list(filter(lambda x: "train" in str(x), files))
        test_files  =        list(filter(lambda x: "test"  in str(x), files))
        train_x     = sorted(list(filter(lambda x: "blur"  in str(x), train_files)))
        train_y     = sorted(list(filter(lambda x: "sharp" in str(x), train_files)))
        test_x      = sorted(list(filter(lambda x: "blur"  in str(x), test_files)))
        test_y      = sorted(list(filter(lambda x: "sharp" in str(x), test_files)))
        
        if len(test_x) == 0 and len(test_y) == 0:
            # transfer self.val_pct fraction of samples from train to test data
            # in case that no test data is found automatically
            import random
            test_x, test_y = [], []
            total = len(train_x)
            idxs  = random.sample(range(total), int(total*self.val_pct))
            for idx in idxs:
                test_x.append(train_x.pop(idx))
                test_y.append(train_y.pop(idx))

        return list(zip(train_x, train_y)), list(zip(test_x, test_y))
      
    def get_transforms(self):
        # TODO: Add RandomAffine, Rotate, Scale, Etc.
        tfms = [K.RandomHorizontalFlip(), K.RandomVerticalFlip()]
        tfms.append(K.RandomAffine(degrees=30, scale=(1.0, 1.3)))
        valid_transforms = []
        return tfms, valid_transforms
    
    def set_size(self, size):
        """ 
        (re)sets the image size for train and valid datasets
        size: (height, width) 
        """
        self.size = size
        self.train_ds.size = tuple(reversed(size))
        self.valid_ds.size = tuple(reversed(size))
        # self.transforms[0] = kornia.Resize(size=size, interpolation='bicubic')
    
    def decode(self, batch, return_torch_type=False):
        batch = make_listy(batch)
        images = [[] for i in range(len(batch))]
        for i, b in enumerate(batch): # One i for blurred images, one for sharp images
            # denormalize a batch with clamping to avoid errors during plotting
            # set_trace()
            if len(b.shape) == 3: # If passed a single image, convert to batch of size 1
                b.unsqueeze_(0)
            b = self.denormalize_func(b).clamp(0.0, 1.0)
            for j in range(b.shape[0]):
                if return_torch_type: image = b[j].detach().cpu()
                else: image = b[j].detach().cpu().numpy().transpose(1,2,0)
                images[i].append(image)
        return images            
        
    def show_batch(self, n=2):
        """ `n` is the number of batches to view """
        # TODO: use torch.utils.makegrid to fix bug here
        dl = iter(self.train_dataloader())
        batches = []
        try:
            for i in range(n): batches.append(next(dl))
        except StopIteration: pass # If asked for more items than in dataset, leave out the rest
        
        total_images = batches[0][0].shape[0] * len(batches)
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, total_images))
        row = 0
        blurred = []
        sharp   = []
        # set_trace()
        for i in range(len(batches)):
            b = augment_image_pair(batches[i], self.transforms)
            blur_ims, sharp_ims = self.decode(b, return_torch_type=True)
            blurred.extend(blur_ims)
            sharp.extend(sharp_ims)
        blurred = make_grid(blurred, nrow=1, range=(0.0, 1.0)).cpu().numpy().transpose(1, 2, 0)
        sharp   = make_grid(sharp  , nrow=1, range=(0.0, 1.0)).cpu().numpy().transpose(1, 2, 0)
        axes[0].imshow(blurred)
        axes[1].imshow(sharp)
        # TODO: Adjust figure size for better viz
        # fig, axes = plt.subplots(nrows=total_images, ncols=2, figsize=(8, 2*total_images))
        # row = 0
        # for i in range(len(batches)):
        #     b = augment_image_pair(batches[i], self.transforms)
        #     blurred, sharp = self.decode(b) # Decode from normalized torch tensors to np images
        #     for j in range(len(blurred)):
        #         axes[row][0].imshow(blurred[j])
        #         axes[row][1].imshow(sharp[j])
        #         row += 1
        return fig


def augment_image_pair(items:list, transforms:list):
    """ 
    Augments each batch items with the same random seed without
    disturbing the global seed
    If transforms is empty, returns items unchanged.
    """
    transforms = make_listy(transforms)
    items = make_listy(items)
    seed = torch.randint(0, 100000, (1,))
    with torch.random.fork_rng():
        for tfm in transforms:
            torch.manual_seed(seed)
            items[0] = tfm(items[0])
            torch.manual_seed(seed)
            items[1] = tfm(items[1])
            seed = torch.randint(0, 100, (1,))
    return items


def make_listy(x):
    if type(x) != list and type(x) != tuple: 
        return [x]
    else: 
        return x
    
    
def get_full_test_out_path(test_out:Path, test_file:Path, test_folder:Path=None, makedirs=True):
    # set_trace()
    test_out, test_file = Path(test_out), Path(test_file)
    str_path = test_file.as_posix()
    if test_folder is not None:
        test_folder = Path(test_folder)
        tail  = test_file.relative_to(test_folder)
    elif 'blur' in str_path:
        tail = str_path[str_path.find("blur") + len("blur")+1:]
    else:
        tail  = test_file.name
    out = test_out/tail
    if makedirs: os.makedirs(out.parent.as_posix(), exist_ok=True)
    
    return out

def image_filter(path:str):
    "Returns true if `path` has the extension of an image file"
    ftype = mimetypes.guess_type(str(path))[0]
    if type(ftype) is str: 
        return 'image' in ftype
    else: return False
    # elif ftype is None: return False
    
def get_image_files(path:Path):
    "Returns iterator over all images found in `path` recursively"
    files = filter(image_filter, Path(path).rglob("*.*"))
    return files


class Noop:
    """ Object form of a function that does nothing """
    def __call__(self, x): return x