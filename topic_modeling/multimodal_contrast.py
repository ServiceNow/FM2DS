import torch
import torch.nn as nn
from transformers import CLIPModel

class MultimodalContrastModel(nn.Module):
    def __init__(self, text_embedding_dim=512, image_embedding_dim=512, topic_dim=100):
        super(MultimodalContrastModel, self).__init__()
        self.text_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").get_text_features
        self.image_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").get_image_features
        self.topic_projection = nn.Linear(text_embedding_dim, topic_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, text_inputs, image_inputs):
        text_features = self.text_encoder(text_inputs)
        image_features = self.image_encoder(image_inputs)
        text_topics = self.topic_projection(text_features)
        image_topics = self.topic_projection(image_features)
        return text_topics, image_topics

    def contrastive_loss(self, text_topics, image_topics):
        logits = torch.matmul(text_topics, image_topics.T) / self.temperature
        labels = torch.arange(len(text_topics)).to(text_topics.device)
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss