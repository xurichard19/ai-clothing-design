import base64
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

    # find kmeans clusters to avoid sending length json queries
    tensors = np.array(list(results.values()))
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(tensors)
    centroids = kmeans.cluster_centers_
    # normalize
    centroids /= np.linalg.norm(centroids, keepdims=True)
    cluster_results = centroids.tolist()
    print(f"\ncompleted kmeans clustering on {len(results)} samples\n")

    # IMPLEMENT:
    # RUN get_text_features ON EVERY TEXT IN CAPTION DATASET
    # STORE DICT OF FEATURES
    # COMPUTE COSINE SIMILARITY ON LARGEST CLUSTER AND CAPTION
    # RETURN CAPTIONS PASSING A THRESHOLD

    # generate tensors and base64 for sample image to give baseline comparison NO NEED
    sample_image = 0 #placeholder
    sample_tensor = get_tensor(model, processor, device, "./data/sample.png")

    # compile into single json input
    input = {
        "clusters": cluster_results,
        "sample image": {
            "tensor": sample_tensor,
            "image description": sample_image
        }
    }
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "system",
                "content": (
                    ""
                )
            },
            {
                "role": "user",
                "content": json.dumps(input)
            }
        ]
    )
    print("\nOPENAI RESPONSE\n")
    print(response.output_text)

def get_tensor(model: CLIPModel, processor: CLIPProcessor, device: torch.device, path: str) -> list[float]:
    sample = Image.open(path)
    inputs = processor(images=sample, text=[""], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        tensor = model.get_image_features(**inputs)

    # normalize
    tensor = tensor / tensor.norm(2, -1, True)
    return tensor.cpu().tolist()[0]

if __name__ == "__main__":
    main()