## AI Clothing Trend Analysis and Designs

This project aims to analyse current fashion trends and generate 'novel' designs based on emerging trends with the help of artificial intelligence. We first train a contrastive language-image pre-training (CLIP) AI model to recognize fashion-specific features through fine-tuning its abstract visual layers. We then use our trained model to generate embeddings for a sample set of trending fashion images and discover the overarching themes using K-means clustering to select the largest centroid (features) by membership. After finding a single representative embedding for the set, we compare the embedding to a series of captions (using cosine similarity) to find the most fitting captions. Finally, we send this to an LLM model (GPT4o mini) to generate a natural language prompt for the diffusion model (small SD v0).

We focused on having a diverse tech stack for this project, using technologies such as PyTorch, Transformers (HuggingFace), NumPy, scikit-learn, OpenAI API, etc.

### Usage Instructions

1) create necessary folders for data, structured like so

```
  > ai clothing design
    > models
      > processors
    > data
      > captions
      > custom-data (any selection of contemporary fashion images, more data stronger correlations)
      > df-training-data (DeepFashion dataset)
    > outputs
    > src...
```

2) install necessary dependencies and data (DeepFashion, custom dataset, OpenAI API key, etc.)
3) run train.py to train CLIP visual encoder (use computing cluster or GPU for best performance)
4) run pipeline.py to generate a design

### The Process

Implementing the actual code for this pipeline was a demanding task. Particularly, our team faced several performance bottlenecks and implementation issues, most notably training the model, the original plan for the encoder to LLM, and the diffusion model. Firstly, training the CLIP encoder had severe performance bottlenecks, even with the help of a computing cluster. Any inefficient code reflected in testing and was a big focus with the initial effort. Although we could have likely optimized the training loop further, the performance reached a satisfactory level. Another issue was the token limit to OpenAI queries. The original plan was to send all embeddings to the LLM for analysis, but soon realized we had to calculate the trends ourselves, settling on K-means instead of computing the average of all features (noisy and could be nonsensical). Finally, it was found that loading an entire diffusion model into memory demands hardware beyond what we had access to, thus bringing us to use a smaller scale model (ideally, the original StableDiffusion would be able to generate more intricate and coherent designs).

### Authors
Richard Xu

Leona Jiang
