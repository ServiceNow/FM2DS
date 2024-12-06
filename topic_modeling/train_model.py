import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from multimodal_contrast import MultimodalContrastModel
from transformers import CLIPProcessor

def preprocess_data(batch, processor):
    """Preprocess the batch data using CLIPProcessor."""
    inputs = processor(
        text=batch['text'],
        images=batch['image'],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return inputs

def train_model(
    dataset_name="flickr30k",
    split="train",
    epochs=10,
    batch_size=32,
    learning_rate=1e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(dataset_name, split=split)

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def collate_fn(batch):
        processed_batch = preprocess_data(batch, processor)
        return processed_batch

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = MultimodalContrastModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            text_inputs = batch['input_ids'].to(device)
            image_inputs = batch['pixel_values'].to(device)

            # Forward pass
            text_topics, image_topics = model(text_inputs, image_inputs)
            loss = model.contrastive_loss(text_topics, image_topics)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

    # Save the trained model
    torch.save(model.state_dict(), "multimodal_contrast_model.pth")
    print("Model training complete and saved.")

if __name__ == "__main__":
    train_model(dataset_name="flickr30k", split="train")
