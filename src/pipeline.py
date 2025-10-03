# TIMELINE:
# 1) data preprocessing and selection (DeepFashion data set + gathered)
# 2) fine tuned CLIP (not zero shot)
# 3) LLM
# 4) diffusion

import torch
from transformers import pipeline

clip = pipeline(
   task="zero-shot-image-classification",
   model="openai/clip-vit-base-patch32",
   dtype=torch.bfloat16,
   device=0
)
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
result = clip("http://images.cocodataset.org/val2017/000000039769.jpg", candidate_labels=labels)
print(result)
