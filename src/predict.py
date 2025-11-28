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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    load_dotenv()

    # load pretrained model and processor from train.py
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
        tensor = get_tensor(model, processor, device, "./data/custom-data/" + sample)
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

    candidate_captions = select_top_captions(model, processor, device, cluster)
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": (
                    ""
                )
            },
            {
                "role": "user",
                "content": json.dumps(candidate_captions)
            }
        ]
    )
    print("\nOPENAI RESPONSE\n")
    print(response.output_text)

def get_tensor(model: CLIPModel, processor: CLIPProcessor, device: torch.device, path: str) -> list[float]:
    sample = Image.open(path)
    inputs = processor(images=sample, return_tensors="pt").to(device)
    with torch.no_grad():
        tensor = model.get_image_features(**inputs)
    tensor = tensor / tensor.norm(2, -1, True)
    return tensor[0]

def generate_caption_tensors(model: CLIPModel, processor: CLIPProcessor, device: torch.device, path: str) -> dict:
    captions = {}

    with open(path, "r") as file:
        embed_captions = tqdm(file.readlines(), desc="gathering text features")
        for caption in embed_captions:
            caption = caption.strip()
            input = processor(text=[caption], return_tensors="pt").to(device)
            with torch.no_grad():
                tensor = model.get_text_features(**input)
            tensor = tensor / tensor.norm(2, -1, True)
            captions[caption] = tensor[0]
    return captions

def select_top_captions(model: CLIPModel, processor: CLIPProcessor, device: torch.device, tensor: torch.Tensor, captions: dict) -> list:
    # select top captions based on cosine similarity between tensors

    candidate_captions = []
    similarity_loop = tqdm(captions.keys(), desc = "finding cosine similarity")
    for caption in similarity_loop:
        similarity = torch.nn.functional.cosine_similarity(tensor.squeeze(), captions[caption].squeeze(), dim=0).item()
        # select captions which exceed similarity threshold
        if similarity >= 0.28: candidate_captions.append(caption)
        print(f"{caption}: {similarity}")
    return candidate_captions

if __name__ == "__main__":
    main()