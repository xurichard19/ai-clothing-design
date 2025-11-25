AI Clothing Trend Analysis and Designs


• train contrastive language-image pretraining (CLIP) model using DeepFashion data set to recognize common clothing traits

• collect 100+ custom images of current fashion trends (from #fashion on Instagram)

• extract common labels from provided images

• send labels to LLM to produce prompt for diffusion model


• uses PyTorch and Hugging Face


1) install necessary libraries and datasets (DeepFashion)
2) run train.py to train CLIP model
3) run predict.py on sample images to extract common features and trends