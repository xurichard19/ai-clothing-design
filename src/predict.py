"""
utilities file for trend analysis
"""

from dotenv import load_dotenv
import numpy as np
import os
from PIL import Image
from sklearn.cluster import KMeans
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

def get_tensor(model: CLIPModel, processor: CLIPProcessor, device: torch.device, path: str) -> list[float]:
    sample = Image.open(path)
    inputs = processor(images=sample, return_tensors="pt").to(device)
    with torch.no_grad():
        tensor = model.get_image_features(**inputs)
    tensor = tensor / tensor.norm(2, -1, True)
    return tensor[0]

def get_representative(results: dict, device: torch.device) -> torch.Tensor:
    # find kmeans clusters for trend analysis
    tensors = np.array(list(results.values()))
    kmeans = KMeans(n_clusters=5)
    labels = kmeans.fit_predict(tensors)
    centroids = kmeans.cluster_centers_

    # normalize
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    # find largest cluster by size (representative)
    unique, count = np.unique(labels, return_counts=True)
    largest_index = unique[np.argmax(count)]
    cluster = torch.tensor(centroids[largest_index], dtype=torch.float32).to(device)
    return cluster

def generate_caption_tensors(model: CLIPModel, processor: CLIPProcessor, device: torch.device, path: str) -> dict:
    captions = {}

    with open(path, "r") as file:
        embed_captions = tqdm(file.readlines(), desc=f"gathering {path[16:-4]} tensors")
        for caption in embed_captions:
            caption = caption.strip()
            input = processor(text=[caption], return_tensors="pt").to(device)
            with torch.no_grad():
                tensor = model.get_text_features(**input)
            tensor = tensor / tensor.norm(2, -1, True)
            captions[caption] = tensor[0]
    return captions

def generate_similarity(model: CLIPModel, processor: CLIPProcessor, device: torch.device, tensor: torch.Tensor, captions: dict) -> dict:
    # calculate cosine similarity between image and captions to select best fitting captions

    candidate_captions = {}
    for caption in captions.keys():
        similarity = torch.nn.functional.cosine_similarity(tensor.squeeze(), captions[caption].squeeze(), dim=0).item()
        candidate_captions[caption] = similarity
    return candidate_captions