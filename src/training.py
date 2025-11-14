import os
import csv
import file_operations
import torch
from transformers import CLIPProcessor, CLIPModel, AdamW, get_scheduler

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

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=lambda examples: collate(examples, processor))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=lambda examples: collate(examples, processor))
    
    NUM_EPOCHS = 20
    optimizer, lr_scheduler = setup_optimizer(train_loader, model, NUM_EPOCHS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")
    model.to(device)

    model.train()
    for epoch in range(NUM_EPOCHS):
        total_train_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k,v in batch.items()}
            outputs = model(**batch)
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
                pass
            # finish validation step

        print() # PRINT ACCURACY
    
    model.save_pretrained("./models/")
    processor.save_pretrained("./models/processors/")
    
def load_csv(split: str) -> list[dict]:
    file = f"./outputs/{split}.csv"
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        data = [{"image": row["image"], "caption": row["caption"]} for row in reader]
    return data

def load_model(model_name="openai/clip-vit-base-patch32"):
    # load model and processor
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # freeze parameters
    for name, param in model.named_parameters():
        if name in {"text_projection", "visual_projection", "logit_scale"}: param.requires_grad = True
        else: param.requires_grad = False

    return model, processor

def setup_optimizer(train_loader: torch.utils.data.DataLoader, model: CLIPModel, epochs: int):
    optimizer = torch.optim.Adam() # USE ADAM INSTEAD OF ADAMW
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6)
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
    images = [example["image"] for example in examples]
    return processor(images, texts, return_tensors="pt", padding=True, truncation=True)

if __name__ == "__main__":
    main()