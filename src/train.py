import os
import csv
import file_operations
import torch
from transformers import CLIPProcessor, CLIPModel, get_scheduler
from PIL import Image

def main():
    # generate split files if not available
    labels = file_operations.Labels()
    if not os.path.exists("./outputs/train.csv"):
        file_operations.Split(labels, "train")
    if not os.path.exists("./outputs/val.csv"):
        file_operations.Split(labels, "val")

    train_data = load_csv("train")
    val_data = load_csv("val")

    model, processor = load_model()

    NUM_EPOCHS = 10
    BATCH_SIZE = 50

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

        for batch in train_loader:
            batch = {k: v.to(device) for k,v in batch.items()}
            outputs = model(**batch, return_loss=True)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k,v in batch.items()}
                outputs = model(**batch, return_loss=True)
                total_val_loss += outputs.loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f'epoch {epoch} with train loss {avg_train_loss} and val loss {avg_val_loss}')

    model.save_pretrained("./models/")
    processor.save_pretrained("./models/processors/")
    
def load_csv(split: str) -> list[dict]:
    file = f"./outputs/{split}.csv"
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        data = [{"image": row["image"], "caption": row["caption"]} for row in reader]
    return data

def load_model(model_name="openai/clip-vit-base-patch32") -> list[CLIPModel, CLIPProcessor]:
    # load model and processor
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # freeze parameters
    for name, param in model.named_parameters():
        if name in {"text_projection", "visual_projection", "logit_scale"}: param.requires_grad = True
        else: param.requires_grad = False

    return model, processor

def setup_optimizer(train_loader: torch.utils.data.DataLoader, model: CLIPModel, epochs: int):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6)
    training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer,
        0,
        training_steps
    )
    return optimizer, lr_scheduler

def collate(examples, processor: CLIPProcessor):
    texts = [example["caption"] for example in examples]
    images = [Image.open('./data/df-training-data/' + example["image"]) for example in examples]
    return processor(images, texts, return_tensors="pt", padding=True, truncation=True)

if __name__ == "__main__":
    main()