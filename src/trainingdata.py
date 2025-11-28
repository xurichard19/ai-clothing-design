"""
utilities file for training data compilation, for use with dataset.py and train.py
"""

import pandas as pd

class Labels:
    """
    store label annotations in object (only one instance required).
    """

    @staticmethod
    def load_file(file_path: str) -> list:
        try:
            with open(file_path, 'r') as f:
                lines = [line.strip().split()[0] for line in f.readlines()][2:]
                print(f"loaded {file_path}")
                return lines
        except FileNotFoundError:
            print(f"could not retrive file {file_path}")
            return []

    # Class variables for label storage
    attr_labels = load_file.__func__("./data/df-training-data/list_attr_cloth.txt")
    cat_labels = load_file.__func__("./data/df-training-data/list_category_cloth.txt")

    def __init__(self) -> None:
        pass

class Split:
    """
    store captions and label data for any given split (train/val/test)
    """

    def load_file(self) -> list:
        file_path = f"./data/df-training-data/{self.split}.txt"
        try:
            with open(file_path, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
                print(f"loaded {file_path}")
                return lines
        except FileNotFoundError:
            print(f"could not retrive file {file_path}")
            return []

    def load_attr_labels(self, labels: Labels) -> list:
        file_path = f"./data/df-training-data/{self.split}_attr.txt"
        attr = []
        try:
            with open(file_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip().split()
                    attr.append([labels.attr_labels[i] for i,v in enumerate(line) if v == "1"])
                print(f"loaded {file_path}")
                return attr
        except FileNotFoundError:
            print(f"could not retrive file {file_path}")
            return []

    def load_cat_labels(self, labels: Labels) -> list:
        file_path = f"./data/df-training-data/{self.split}_cate.txt"
        try:
            with open(file_path, 'r') as f:
                lines = [labels.cat_labels[int(line.strip()) - 1] for line in f.readlines()]
                print(f"loaded {file_path}")
                return lines
        except FileNotFoundError:
            print(f"could not retrive file {file_path}")
            return []

    def __init__(self, labels: Labels, split: str) -> None:
        self.split = split
        self.images = self.load_file()
        self.attr_labels = self.load_attr_labels(labels)
        self.cat_labels = self.load_cat_labels(labels)

        self.captions = []
        for image, attr, cat in zip(self.images, self.attr_labels, self.cat_labels):
            if attr:
                caption = f"a {' '.join(attr)} {cat}"
            else:
                caption = f"a {cat}"
            self.captions.append({"image": image, "caption": caption})

        df = pd.DataFrame(self.captions)
        df.to_csv(f"./outputs/{split}.csv", index=False)
