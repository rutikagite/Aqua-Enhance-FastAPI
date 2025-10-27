import os
import random
from PIL import Image
from typing import Dict
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as ttf
import torchvision.transforms as T
from torchvision.transforms import (RandomCrop, Pad, RandomHorizontalFlip,
                                    RandomVerticalFlip, Resize, ToTensor, Normalize,
                                    RandomResizedCrop, ColorJitter)


def uieb_transforms(size):
    """Enhanced transforms with color jitter and random resized crop"""
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.9, 1.0)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor()
    ])


class UIEBTrain(Dataset):
    _INPUT_ = 'input'
    _TARGET_ = 'target'

    def __init__(self, folder: str, size: int):
        super(UIEBTrain, self).__init__()
        self._size = size
        self._root = folder
        self._filenames = os.listdir(os.path.join(self._root, self._INPUT_))

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, item):
        input_img = Image.open(os.path.join(self._root, self._INPUT_, self._filenames[item]))
        target_img = Image.open(os.path.join(self._root, self._TARGET_, self._filenames[item]))
        input_img, target_img = self._aug_data(input_img, target_img)
        return input_img, target_img

    def _aug_data(self, input_img, target_img):
        # padding to ensure minimum size
        pad_w = self._size - input_img.width if input_img.width < self._size else 0
        pad_h = self._size - input_img.height if input_img.height < self._size else 0
        input_img = Pad(padding=(0, 0, pad_w, pad_h), padding_mode='reflect')(input_img)
        target_img = Pad(padding=(0, 0, pad_w, pad_h), padding_mode='reflect')(target_img)
        
        # random resized crop (instead of regular random crop)
        i, j, h, w = RandomResizedCrop.get_params(input_img, scale=(0.9, 1.0), 
                                                   ratio=(0.95, 1.05))
        input_img = ttf.resized_crop(input_img, i, j, h, w, (self._size, self._size))
        target_img = ttf.resized_crop(target_img, i, j, h, w, (self._size, self._size))
        
        # color jitter (apply same transformation to both images)
        color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        # Get the same random parameters for both images
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            color_jitter.get_params(color_jitter.brightness, color_jitter.contrast,
                                   color_jitter.saturation, color_jitter.hue)
        
        for fn_id in fn_idx:
            if fn_id == 0:
                input_img = ttf.adjust_brightness(input_img, brightness_factor)
                target_img = ttf.adjust_brightness(target_img, brightness_factor)
            elif fn_id == 1:
                input_img = ttf.adjust_contrast(input_img, contrast_factor)
                target_img = ttf.adjust_contrast(target_img, contrast_factor)
            elif fn_id == 2:
                input_img = ttf.adjust_saturation(input_img, saturation_factor)
                target_img = ttf.adjust_saturation(target_img, saturation_factor)
            elif fn_id == 3:
                input_img = ttf.adjust_hue(input_img, hue_factor)
                target_img = ttf.adjust_hue(target_img, hue_factor)
        
        # random flip
        vertical_flip_seed = random.random()
        horizontal_flip_seed = random.random()
        if vertical_flip_seed > 0.5:
            input_img = ttf.vflip(input_img)
            target_img = ttf.vflip(target_img)
        if horizontal_flip_seed > 0.5:
            input_img = ttf.hflip(input_img)
            target_img = ttf.hflip(target_img)
        
        # random rotate
        rand_rotate = random.randint(0, 3)
        input_img = ttf.rotate(input_img, 90 * rand_rotate)
        target_img = ttf.rotate(target_img, 90 * rand_rotate)
        
        # to tensor
        input_img = ToTensor()(input_img)
        target_img = ToTensor()(target_img)
        
        # TODO mix up
        # TODO norm
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return input_img, target_img


class UIEBValid(Dataset):
    _INPUT_ = 'input'
    _TARGET_ = 'target'

    def __init__(self, folder: str, size: int):
        super(UIEBValid, self).__init__()
        self._size = size
        self._root = folder
        self._filenames = os.listdir(os.path.join(self._root, self._INPUT_))
        self._transform = Resize((self._size, self._size))

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, item):
        input_img = Image.open(os.path.join(self._root, self._INPUT_, self._filenames[item]))
        target_img = Image.open(os.path.join(self._root, self._TARGET_, self._filenames[item]))
        input_img, target_img = self._aug_data(input_img, target_img)
        return input_img, target_img

    def _aug_data(self, input_img, target_img):
        # padding
        pad_w = self._size - input_img.width if input_img.width < self._size else 0
        pad_h = self._size - input_img.height if input_img.height < self._size else 0
        input_img = Pad(padding=(0, 0, pad_w, pad_h), padding_mode='reflect')(input_img)
        target_img = Pad(padding=(0, 0, pad_w, pad_h), padding_mode='reflect')(target_img)
        
        # resize (no augmentation for validation)
        input_img = self._transform(input_img)
        target_img = self._transform(target_img)
        
        # to tensor
        input_img = ToTensor()(input_img)
        target_img = ToTensor()(target_img)
        
        return input_img, target_img