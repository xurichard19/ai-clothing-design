from diffusers import DiffusionPipeline
from dotenv import load_dotenv
import json
import os
from openai import OpenAI
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

    # generate tensors for each sample image
    results = {}
    samples = [sample for sample in os.listdir("./data/custom-data") if sample != ".DS_Store"]
    predict_loop = tqdm(samples, desc="gathering sample image tensors")
    for sample in predict_loop:
        tensor = predict.get_tensor(model, processor, device, "./data/custom-data/" + sample)
        results[sample] = tensor

    # get largest kmeans cluster
    representative = predict.get_representative(results, device)

    # generate tensors for captions
    caption_sets = {}
    for file in os.listdir("./data/captions"):
        if file == ".DS_Store": continue
        caption_sets[file[:-4]] = predict.generate_caption_tensors(model, processor, device, "./data/captions/" + file)
    
    # calculate cosine similarity for each caption
    similarity_dict = {}
    for captions in caption_sets.values():
        similarity_dict = similarity_dict | predict.generate_similarity(model, processor, device, representative, captions)

    # select top captions
    sorted_similarity = sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True)
    top_captions = [caption[0] for caption in sorted_similarity[:len(similarity_dict) // 9]]
    print(f"\nselected {len(top_captions)} captions with similarity scores of {similarity_dict[top_captions[0]]} to {similarity_dict[top_captions[-1]]}")

    # send top captions to OpenAI for a natural language prompt
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "You are a fashion designer who utilizes diffusion models to generate quality designs. "
                    "Given a list of captions describing a set of fashion images, generate a design for a single "
                    "piece of clothing you feel best encapsulates the overall theme of the images. You are "
                    "welcome to use fashion specific terms to describe your design."

                    "Your prompt will be sent to a diffusion model to generate the actual design. Only respond "
                    "with the design idea and key features that you would like to see in the final product. "
                    "In the explanation of the design idea, you should summarize key information and explicitly "
                    "state that you are prompting the model to generate a clothing design for a singular piece. "
                    "You should format the response as if you were prompting the diffusion model yourself. Do "
                    "not include any pleasantries or anything of the sort."

                    "Your prompt should be strictly limited to 77 tokens."
                )
            },
            {
                "role": "user",
                "content": json.dumps({"captions": top_captions})
            }
        ]
    )
    with open("./outputs/design.txt", "w") as f:
        f.write(response.output_text)

    # send prompt to image generation model
    pipe = DiffusionPipeline.from_pretrained(
        "OFA-Sys/small-stable-diffusion-v0",
        dtype=torch.float16
    ).to(device)
    image = pipe(response.output_text).images[0]
    image.save("./outputs/design.png")
    print("\nsuccessfully generated image in /outputs/design.png")

if __name__ == "__main__":
    main()