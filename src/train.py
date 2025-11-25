import os
import csv
import trainingdata
import dataset
import torch
from transformers import CLIPProcessor, CLIPModel, get_scheduler
from PIL import Image
from tqdm import tqdm

def main():
    train_data = dataset.Dataset("train")
    val_data = dataset.Dataset("val")

    model, processor = load_model()

    NUM_EPOCHS = 10
    BATCH_SIZE = 100

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda examples: collate(examples, processor)
        )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        collate_fn=lambda examples: collate(examples, processor)
        )

    optimizer, lr_scheduler = setup_optimizer(train_loader, model, NUM_EPOCHS)

    # for computing cluster
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")
    model.to(device)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"epoch {epoch+1} training")
        for batch in train_loop:
            batch = {k: v.to(device) for k,v in batch.items()}
            outputs = model(**batch, return_loss=True)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if train_loop.n > 0:
                avg_train_loss = total_train_loss / train_loop.n
                train_loop.set_postfix(loss=avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():

            val_loop = tqdm(train_loader, desc=f"epoch {epoch+1} validation")
            for batch in val_loop:
                batch = {k: v.to(device) for k,v in batch.items()}
                outputs = model(**batch, return_loss=True)
                total_val_loss += outputs.loss.item()

                if val_loop.n > 0:
                    avg_val_loss = total_val_loss / val_loop.n
                    val_loop.set_postfix(loss=avg_val_loss)

        print(f'completed epoch {epoch+1}')

    model.save_pretrained("./models/")
    processor.save_pretrained("./models/processors/")
    print("saved model")

def load_model(model_name="openai/clip-vit-base-patch32") -> list[CLIPModel, CLIPProcessor]:
    # load model and processor
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # freeze parameters
    for name, param in model.named_parameters():
        # unfreeze final layers
        if name in {"text_projection.weight", "visual_projection.weight", "logit_scale"}: param.requires_grad = True
        # unfreeze abstract visual layers
        elif "vision_model.encoder.layers.11" in name or "vision_model.encoder.layers.10" in name: param.requires_grad = True
        else: param.requires_grad = False

    return model, processor

def setup_optimizer(train_loader: torch.utils.data.DataLoader, model: CLIPModel, epochs: int):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()))
    training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer,
        0,
        training_steps
    )
    return optimizer, lr_scheduler

def collate(samples, processor: CLIPProcessor):
    images = [sample[0] for sample in samples]
    texts = [sample[1] for sample in samples]
    return processor(images, texts, return_tensors="pt", padding=True, truncation=True)

if __name__ == "__main__":
    main()