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
labels = ["balenciaga", "acne studios", "our legacy", "givenchy", "supreme", "celine"]
url = "https://scontent-atl3-1.cdninstagram.com/v/t51.82787-15/550217220_18080737532496810_6198471278833621107_n.jpg?stp=dst-jpg_e35_p750x750_sh0.08_tt6&_nc_cat=106&ig_cache_key=MzcyNDQwNjk1ODU4Njk1Njg5MQ%3D%3D.3-ccb1-7&ccb=1-7&_nc_sid=58cdad&efg=eyJ2ZW5jb2RlX3RhZyI6InhwaWRzLjExNzl4MTQ5MS5zZHIuQzMifQ%3D%3D&_nc_ohc=ojcQfzMi1l0Q7kNvwH7oMmc&_nc_oc=AdkCYtRF90hYhDAFKHUfKejUG-95U66LADpibPn_PwJTonlX3sg9tWqSeAGBgA4q9EI&_nc_ad=z-m&_nc_cid=0&_nc_zt=23&_nc_ht=scontent-atl3-1.cdninstagram.com&_nc_gid=PSjKzTGMn8bOPNchDcHnNA&oh=00_Affwv3XCnZBdFs_2pjhyzH4uFmxQPjKUEGAXDpokqrkv1w&oe=68E60A50"
result = clip(url, candidate_labels=labels)
print("")
print(result)
