import math
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from config import *
import pandas as pd
import numpy as np
from glob import glob
import os
from PIL import Image
import imageio

# Celebrity Dataset custom class
# Climate Dataset Custom Class
class CelebrityDataset(Dataset):
    def __init__(self, images):
        self.images = images


    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = self.images[i] / 255.0
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)
        return image

def preprocess_data():
    df_attrs = pd.read_csv(ATTRIBUTES_PATH, sep='\t', skiprows=1)
    df_attrs = pd.DataFrame(df_attrs.iloc[:,:-1].values, columns = df_attrs.columns[1:])

    photo_ids = []
    for dirpath, _, filenames in os.walk(DATASET_PATH):
        for fname in filenames:
            if fname.endswith(".jpg"):
                fpath = os.path.join(dirpath,fname)
                photo_id = fname[:-4].replace('_',' ').split()
                person_id = ' '.join(photo_id[:-1])
                photo_number = int(photo_id[-1])
                photo_ids.append({'person':person_id, 'imagenum':photo_number, 'photo_path':fpath})

    photo_ids = pd.DataFrame(photo_ids)
    df = pd.merge(df_attrs, photo_ids, on=('person', 'imagenum'))

    assert len(df)==len(df_attrs),"lost some data when merging dataframes"
    
    images = df['photo_path'].apply(imageio.imread)\
                                .apply(lambda img: img[50:-50, 50:-50])\
                                .apply(lambda img: np.array(Image.fromarray(img).resize([128, 128])) )

    images = np.stack(images.values).astype('uint8')
    attributes = df.drop(["photo_path", "imagenum"],axis=1)
    np.save('processed_dataset/images.npy', images)
    attributes.to_csv('processed_dataset/attributes.csv', index=False)

    return images, attributes

def fetch_dataset():
    # images, attributes = preprocess_data()
    images = np.load('processed_dataset/images.npy')
    attributes = pd.read_csv('processed_dataset/attributes.csv')

    train_val_split = math.floor(len(images)*0.8)
    train_images = images[:train_val_split]
    val_images = images[train_val_split:]
    

    train_ds, val_ds = CelebrityDataset(train_images), CelebrityDataset(val_images)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_ds, train_loader, val_ds, val_loader, attributes

def fetch_data():
    images = np.load('processed_dataset/images.npy')
    attributes = pd.read_csv('processed_dataset/attributes.csv')
    
    return images, attributes

