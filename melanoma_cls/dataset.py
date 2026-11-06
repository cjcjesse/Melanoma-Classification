import csv
import random
import pandas as pd
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F

from PIL import Image
from torch.utils.data import Dataset

def create_ph2_metadata(ph2_metadata_path, ph2_txt_path):
    if not ph2_metadata_path.exists():
        with open(ph2_txt_path, encoding='utf-8') as f:
            lines = f.read().splitlines()

        rows = []
        for line in lines:
            line = line.strip()
            # Keep only table rows; this skips the legend text
            if not line.startswith('|'):
                continue
            # Replace the double-pipe delimiters, then split on single pipes
            cells = [c.strip() for c in line.strip('|').replace('||', '|').split('|')]
            # Drop empty cells created by leading/trailing/double pipes
            if cells and cells[0] == '':
                cells = cells[1:]
            if cells and cells[-1] == '':
                cells = cells[:-1]
            rows.append(cells)

        with open(ph2_metadata_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(rows)

    ph2_meta = pd.read_csv(ph2_metadata_path, header=0, sep=',')
    print(ph2_meta.shape) # (200, 10); only need cols: ['Name', 'Clinical Diagnosis'], collapse labels into binary labels and assign to new column 'labels' resulting in new cols ['Name', 'labels']
    ph2_meta.head(10)

    return ph2_meta

class RandomRotation90: # custom transformation class that randomly rotates in 90 degree increments
    def __call__(self, img):
        return F.rotate(img, 90 * random.randint(0, 3))

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    RandomRotation90(), # only multiples of 90 degree rotations ensure no introduction of black edges after rotation
    transforms.ColorJitter(brightness=0.15, contrast=0.15), # no saturation/hue
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet mean and std
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) # ImageNet mean and std

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) # ImageNet mean and std

class ISICDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df # cols: ['isic_id', 'patient_id'. 'target'] <- might have 'weights' 
        self.root_dir = root_dir # expect to pass in train_images = data_cache / 'train-image' / 'image'
        self.transform = transform
        self.targets = df['target']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = f"{self.df['isic_id'].iloc[idx]}.jpg"
        img_path = self.root_dir / img_name

        with Image.open(img_path) as img:
            image = img.convert('RGB')
            label = self.targets.iloc[idx]

            if self.transform:
                image = self.transform(image)

        return image, label

class PH2Dataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df # cols: ['Name', 'labels']
        self.root_dir = root_dir # expect to pass ph2_img_path = ph2_cache / 'PH2Dataset/PH2 Dataset images'
        self.transform = transform
        self.targets = df['labels']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df['Name'].iloc[idx]
        img_path = self.root_dir / img_name / f'{img_name}_Dermoscopic_Image' / f'{img_name}.bmp'

        with Image.open(img_path) as img:
            image = img.convert('RGB') # standardize to RGB
            label = self.targets.iloc[idx]

            if self.transform:
                image = self.transform(image)

        return image, label