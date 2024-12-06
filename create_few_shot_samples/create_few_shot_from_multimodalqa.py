import os
import json
from datasets import Dataset
from FM2DS.create_few_shot_samples.download_wiki_page import get_wikipedia_content_as_json

def create_few_shot_mmqa(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    few_shot_samples = []
    with open(input_file, 'r') as f:
        for line in f:
            sample = json.loads(line.strip())
            modalities = sample.get("modalities", [])
            wiki_question = sample["metadata"].get("wiki_entities_in_question", [])
            wiki_answer = sample["metadata"].get("wiki_entities_in_answers", [])
            
            if len(modalities) <= 1:
                continue

            wiki_pages = {(entity["wiki_title"], entity["url"]) for entity in wiki_question + wiki_answer}
            
            if not (2 <= len(wiki_pages) <= 3):
                continue

            data = {
                "id": sample["qid"],
                "question": sample["question"],
                "answer": sample["answers"][0]["answer"], 
                "documents": []
            }
            
            for page in wiki_pages:
                try:
                    document = get_wikipedia_content_as_json(page[1])
                    data["documents"].append({
                        "document_title": page[0],
                        "document": document
                    })
                except Exception as e:
                    print(f"Error fetching content for {page[1]}: {e}")
            
            few_shot_samples.append(data)

    hf_dataset = Dataset.from_list(few_shot_samples)

    hf_dataset.save_to_disk(output_dir)
    print(f"Dataset saved to {output_dir}")

if __name__ == "__main__":
    input_file = "MMQA_train.jsonl"
    output_dir = "FM2DS/data/few_shot_mmqa"
    create_few_shot_mmqa(input_file, output_dir)
