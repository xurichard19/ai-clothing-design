import dataset
import torch
from transformers import CLIPProcessor, CLIPModel, get_scheduler
from tqdm import tqdm

def main():
    # load data
    train_data = dataset.Dataset("train")
    val_data = dataset.Dataset("val")

    # load model and processor
    MODEL_NAME = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # freeze parameters
    for name, param in model.named_parameters():
        # unfreeze final layers
        if name in {"text_projection.weight", "visual_projection.weight", "logit_scale"}: param.requires_grad = True
        # unfreeze abstract visual layer
        elif "vision_model.encoder.layers.11" in name: param.requires_grad = True
        else: param.requires_grad = False

    NUM_EPOCHS = 12
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

    LEARNING_RATE = 1e-6
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=0.001)
    training_steps = NUM_EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer,
        0,
        training_steps
    )

    # for computing cluster
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")
    model.to(device)

    # TRAINING LOOP
    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        print(f"epoch {epoch+1}")
        model.train()
        total_train_loss = 0.0
        train_loop = tqdm(train_loader, desc="training")
        for batch in train_loop:
            batch = {k: v.to(device) for k,v in batch.items()}
            outputs = model(**batch, return_loss=True)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            avg_train_loss = total_train_loss / (train_loop.n + 1)
            train_loop.set_postfix(loss=f"{avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc="validation")
            for batch in val_loop:
                batch = {k: v.to(device) for k,v in batch.items()}
                outputs = model(**batch, return_loss=True)
                total_val_loss += outputs.loss.item()

                avg_val_loss = total_val_loss / (val_loop.n + 1)
                val_loop.set_postfix(loss=f"{avg_val_loss:.4f}")

        avg_val_loss = total_val_loss / len(val_loop)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            model.save_pretrained("./models/")
            processor.save_pretrained("./models/processors/")
    print(f"completed training with best validation loss of {best_loss}")

def collate(samples, processor: CLIPProcessor):
    images = [sample[0] for sample in samples]
    texts = [sample[1] for sample in samples]
    return processor(images, texts, return_tensors="pt", padding=True, truncation=True)

if __name__ == "__main__":
    main()