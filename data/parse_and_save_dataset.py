import os
from bs4 import BeautifulSoup
import tensorflow.compat.v1 as tf
from datasets import Dataset
from FM2DS.topic_modeling.infer_topics import infer_topics_for_documents

class WikiWeb2MParser:
    def __init__(self, input_dir, output_dir, model_path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_path = model_path
        self.context_feature_description = {
            'split': tf.io.FixedLenFeature([], tf.string),
            'page_title': tf.io.FixedLenFeature([], tf.string),
            'page_url': tf.io.FixedLenFeature([], tf.string),
            'clean_page_description': tf.io.FixedLenFeature([], tf.string),
            'raw_page_description': tf.io.FixedLenFeature([], tf.string),
            'is_page_description_sample': tf.io.FixedLenFeature([], tf.int64),
            'page_contains_images': tf.io.FixedLenFeature([], tf.int64),
            'page_content_sections_without_table_list': tf.io.FixedLenFeature([], tf.int64),
        }
        self.sequence_feature_description = {
            'section_title': tf.io.VarLenFeature(tf.string),
            'section_text': tf.io.VarLenFeature(tf.string),
            'section_image_url': tf.io.VarLenFeature(tf.string),
            'section_image_captions': tf.io.VarLenFeature(tf.string),
        }

    def parse_tfrecord(self, serialized_example):
        context, sequence = tf.io.parse_single_sequence_example(
            serialized_example,
            context_features=self.context_feature_description,
            sequence_features=self.sequence_feature_description,
        )
        return context, sequence

    def extract_hyperlinks(self, text):
        """Extract text inside <a> tags as hyperlinks."""
        soup = BeautifulSoup(text, "html.parser")
        return [a.text for a in soup.find_all("a")]

    def save_as_hf_dataset(self):
        data = {
            "id": [],
            "document_title": [],
            "document": [],
            "hyperlinks": [],
            "topics": [],
        }

        tfrecord_files = [
            os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) if f.endswith(".gz")
        ]
        raw_dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type="GZIP")
        parsed_dataset = raw_dataset.map(self.parse_tfrecord)

        documents = [] # List to collect all documents for topic inference
        for record in parsed_dataset:
            context, sequence = record
            doc_id = context['page_title'].numpy().decode()
            doc_title = context['page_title'].numpy().decode()
            document = []
            hyperlinks = []

            text_sections = tf.sparse.to_dense(sequence['section_text']).numpy()
            for text in text_sections:
                decoded_text = text.decode()
                document.append({"type": "text", "value": decoded_text})
                hyperlinks.extend(self.extract_hyperlinks(decoded_text))

            image_urls = tf.sparse.to_dense(sequence['section_image_url']).numpy()
            image_captions = tf.sparse.to_dense(sequence['section_image_captions']).numpy()

            for img_url, caption in zip(image_urls, image_captions):
                document.append({
                    "type": "image",
                    "value": img_url.decode()
                })

                document.append({
                    "type": "text",
                    "value": caption.decode(),
                })

            data['id'].append(doc_id)
            data['document_title'].append(doc_title)
            data['document'].append(document)
            data['hyperlinks'].append(hyperlinks)
            documents.append(document)

        data['topics'] = infer_topics_for_documents(documents, model_path=self.model_path)

        dataset = Dataset.from_dict(data)
        dataset.save_to_disk(self.output_dir)
        print(f"Dataset saved to {self.output_dir}")

if __name__ == "__main__":
    parser = WikiWeb2MParser(
        input_dir="FM2DS/data/wikiweb2m",
        output_dir="FM2DS/data/hf_dataset",
        model_path="FM2DS/topic_modeling/multimodal_contrast_model.pth"
    )
    parser.save_as_hf_dataset()
