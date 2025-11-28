from dotenv import load_dotenv
import json
import numpy as np
import os
from openai import OpenAI
from PIL import Image
from sklearn.cluster import KMeans
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import predict

def main():
    model = CLIPModel.from_pretrained("./models")
    processor = CLIPProcessor.from_pretrained("./models/processors")

    # for computing cluster
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")
    model.to(device)
    model.eval()

    results = {}
    samples = [sample for sample in os.listdir("./data/custom-data") if sample != ".DS_Store"]
    print(f"\npredicting {len(samples)} images...\n")

    predict_loop = tqdm(samples, desc="prediction")
    for sample in predict_loop:
        tensor = predict.get_tensor(model, processor, device, "./data/custom-data/" + sample)
        results[sample] = tensor

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

    captions = {}
    for file in os.listdir("./data/captions"):
        captions[file[:-4]] = predict.generate_caption_tensors(model, processor, device, "./data/captions/" + file)
    image_tensor = predict.get_tensor(model, processor, device, "./data/sample.png")

    final = []
    for i in captions.values():
        final+=predict.select_top_captions(model, processor, device, cluster, i)

if __name__ == "__main__":
    main()