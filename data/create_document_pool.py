from datasets import load_from_disk, Dataset
from itertools import combinations
from collections import defaultdict

def create_document_pool(dataset_path, output_path):
    """
    Creates a pool of related documents based on topics, hyperlinks, and document titles.

    Args:
        dataset_path (str): Path to the Hugging Face dataset in disk format.
        output_path (str): Path to save the newly created documents_pool dataset.
    """
    # Load the dataset
    dataset = load_from_disk(dataset_path)
    document_ids = dataset['id']
    document_topics = dataset['topics']
    document_titles = dataset['document_title']
    document_hyperlinks = dataset['hyperlinks']

    topic_to_docs = defaultdict(set)
    for doc_id, topics in zip(document_ids, document_topics):
        for topic in topics:
            topic_to_docs[topic].add(doc_id)

    hyperlink_to_docs = defaultdict(set)
    for doc_id, hyperlinks in zip(document_ids, document_hyperlinks):
        for hyperlink in hyperlinks:
            hyperlink_to_docs[hyperlink].add(doc_id)

    title_to_doc = {title: doc_id for doc_id, title in zip(document_ids, document_titles)}

    document_pool = []
    for doc_id, topics, hyperlinks, title in zip(
        document_ids, document_topics, document_hyperlinks, document_titles
    ):
        related_docs = set()

        for topic in topics:
            related_docs.update(topic_to_docs[topic])

        for hyperlink in hyperlinks:
            related_docs.update(hyperlink_to_docs[hyperlink])

        if title in title_to_doc:
            related_docs.add(title_to_doc[title])

        related_docs.discard(doc_id)

        for combination in combinations(related_docs, 2):
            document_pool.append([doc_id] + list(combination))
        for combination in combinations(related_docs, 3):
            document_pool.append([doc_id] + list(combination))

    pool_dataset = Dataset.from_dict({"document_relations": document_pool})

    pool_dataset.save_to_disk(output_path)
    print(f"Document pool dataset saved to {output_path}")

if __name__ == "__main__":
    dataset_path = "FM2DS/data/hf_dataset"
    output_path = "FM2DS/data/documents_pool"
    create_document_pool(dataset_path, output_path)
