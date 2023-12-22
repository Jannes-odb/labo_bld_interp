from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


IGNORE_IMAGES = [
    'c1f3f2cb1463bbfa905ccaff484cd668.png',  # truncated image
]


# Ignore 'DataFrame.swapaxes' is deprecated warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Ignore palette images with transparency expressed in bytes warnings
warnings.simplefilter(action='ignore', category=UserWarning)

class DimSumDataset(Dataset):
    def __init__(self, data_path, subset, k=5, val_fold=0, transform=None):
        data_path = Path(data_path)
        df = pd.DataFrame([
            {
                'image': str(img_path),
                'label': img_path.parent.name
            }
            for img_path in data_path.glob('data/*/*')
        ])
        
        # create the label-to-integer dictionary
        label_to_integer = {label: i for i, label in enumerate(sorted(df['label'].unique()))}

        # split dataset into train, test and val
        train_val, test = train_test_split(df, train_size = 0.8, random_state = 42)
        folds = np.array_split(train_val, k)

        val = folds[val_fold]
        train_folds = [fold for i, fold in enumerate(folds)
               if i != val_fold]
        train = pd.concat(train_folds)
        
        self.data_path = data_path
        self.transform = transform
        self.label_to_integer = label_to_integer
        
        if subset == 'train':
            self.df = train
        elif subset == 'test':
            self.df = test
        elif subset == 'val':
            self.df = val
        else:
            raise ValueError('wrong subset')

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image']
        label = row['label']
        int_label = self.label_to_integer[label]
        img = Image.open(image_path)
        if self.transform is not None:
            img = self.transform(img)
        return (img, int_label)
        
    def __len__(self):
        return len(self.df)
    
