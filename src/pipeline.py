from dotenv import load_dotenv
import json
import os
from openai import OpenAI
from PIL import Image
import predict
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

    predict_loop = tqdm(samples, desc="prediction")
    for sample in predict_loop:
        tensor = predict.get_tensor(model, processor, device, "./data/custom-data/" + sample)
        results[sample] = tensor

    representative = predict.get_representative(results, device)
    captions = {}
    for file in os.listdir("./data/captions"):
        captions[file[:-4]] = predict.generate_caption_tensors(model, processor, device, "./data/captions/" + file)

    final = []
    for i in captions.values():
        final += predict.select_top_captions(model, processor, device, representative, i)

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
                "content": json.dumps({"captions": final})
            }
        ]
    )
    print("\nOPENAI RESPONSE\n")
    print(response.output_text)