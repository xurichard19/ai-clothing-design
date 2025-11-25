from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

def main():
    # load pretrained model and processor from train.py
    model = CLIPModel.from_pretrained("./models")
    processor = CLIPProcessor.from_pretrained("./models/processors")

    # for computing cluster
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")
    model.to(device)
    model.eval()

    sample = Image.open("./data/custom-data/test.jpg")
    inputs = processor(images=sample, text=[""], return_tensors="pt").to(device)
    outputs = model.get_image_features(**inputs)
    print(outputs)

# generate preds for sample images

if __name__ == "__main__":
    main()