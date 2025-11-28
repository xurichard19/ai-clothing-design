"""
utilities file for generating pytorch compatible datasets
"""

import csv
import os
from PIL import Image
import torch
import trainingdata

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split: str):
        labels = trainingdata.Labels()
        if not os.path.exists(f"./outputs/{split}.csv"):
            trainingdata.Split(labels, split)

        self.data = []
        file = f"./outputs/{split}.csv"
        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            self.data = [{"image": row["image"], "caption": row["caption"]} for row in reader]

    def __getitem__(self, index):
        sample = self.data[index]
        image = Image.open('./data/df-training-data/' + sample['image'])
        return image, sample['caption']

    def __len__(self):
        return len(self.data)