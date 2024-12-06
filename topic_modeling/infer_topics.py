import torch
from transformers import CLIPProcessor
from FM2DS.topic_modeling.multimodal_contrast import MultimodalContrastModel

def preprocess_document(document, processor, device):
    """Preprocess a document for the model."""
    text_inputs = []
    image_inputs = []

    for obj in document:
        if obj["type"] == "text":
            text_inputs.append(obj["value"])
        elif obj["type"] == "image":
            image_inputs.append(obj["value"])

    inputs = processor(
        text=text_inputs if text_inputs else None,
        images=image_inputs if image_inputs else None,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    if "input_ids" in inputs:
        inputs["input_ids"] = inputs["input_ids"].to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(device)

    return inputs

def infer_topics_for_documents(documents, model_path="multimodal_contrast_model.pth"):
    """Infer topics for a list of JSON objects."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultimodalContrastModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    all_topics = set()

    for document in documents:
        inputs = preprocess_document(document, processor, device)

        with torch.no_grad():
            text_topics, image_topics = None, None

            if "input_ids" in inputs:
                text_topics = model.text_encoder(inputs["input_ids"])
            if "pixel_values" in inputs:
                image_topics = model.image_encoder(inputs["pixel_values"])

            if text_topics is not None and image_topics is not None:
                combined_topics = (text_topics + image_topics) / 2
            elif text_topics is not None:
                combined_topics = text_topics
            elif image_topics is not None:
                combined_topics = image_topics
            else:
                combined_topics = torch.Tensor([])

            all_topics.update(combined_topics.cpu().numpy().tolist())

    return list(all_topics)