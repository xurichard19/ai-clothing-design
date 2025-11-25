from dotenv import load_dotenv
import json
import os
from openai import OpenAI
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

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
    print(f"predicting {len(samples)} images...")

    for sample in samples:
        tensor = get_tensor(model, processor, device, "./data/custom-data/" + sample)
        results[sample] = tensor
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "You are a fashion analyst. Given CLIP embeddings of some sample fashion "
                    "images, identify patterns like silhouettes, textures, fits, colors, and "
                    "overall themes. You may use fashion specific terms including but not limited to "
                    "deconstruction, trompe de l'oeil, futurism, relaxed fit, etc. to understand the trends. "
                    
                    "Then, generate a prompt for a diffusion model to generate a clothing "
                    "design based on the observed trends in the sample fashion images. For "
                    "example, you could say something akin to 'Generate a clothing design for "
                    "a pair of denim jeans with a baggy silhouette and a vintage theme that "
                    "combines streetwear and avant-garde aesthetics through a distressed finish "
                    "and a mud wash'. You should describe key features of the new design and "
                    "the overall theme of the piece."
                )
            },
            {
                "role": "user",
                "content": json.dumps({"image_embeddings": results})
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