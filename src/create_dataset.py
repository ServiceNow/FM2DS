import os
import argparse
from datasets import Dataset
from FM2DS.src.question.generate import generate_question
from FM2DS.src.question.validate import validate_question
from FM2DS.src.answer.generate import generate_answer
from FM2DS.src.answer.validate import validate_answer
from FM2DS.src.query.generate import generate_query
from FM2DS.src.query.validate import validate_query
from datasets import load_from_disk

def main(args):
    # Load required datasets
    documents_pool = load_from_disk("FM2DS/data/documents_pool")
    hf_dataset = load_from_disk("FM2DS/data/hf_dataset")
    few_shot_examples = load_from_disk("FM2DS/data/few_shot_mmqa")

    # Prepare output directory
    os.makedirs(os.path.dirname(args.output_dataset), exist_ok=True)

    # Initialize the dataset to store generated examples
    generated_data = {
        "question": [],
        "answer": [],
        "documents": [],
        "query": [],
    }

    example_count = 0

    while example_count < args.num_examples:
        # Step 1: Select a pool of related documents
        pool = documents_pool[example_count % len(documents_pool)]  # Cycle through the pool
        document_ids = pool["document_ids"]
        documents = [doc for doc in hf_dataset if doc["id"] in document_ids]

        # Step 2: Generate a question
        question = generate_question(
            model_name=args.model,
            documents=documents,
            few_shot_examples=few_shot_examples[:args.num_few_shot],
        )

        # Step 3: Validate the question
        if not validate_question(question, documents, model_name=args.model):
            continue  # Skip if validation fails

        # Step 4: Generate an answer
        answer = generate_answer(
            model_name=args.model,
            question=question,
            documents=documents,
            few_shot_examples=few_shot_examples[:args.num_few_shot],
        )

        # Step 5: Validate the answer
        if not validate_answer(answer, documents, question):
            continue  # Skip if validation fails

        # Step 6: Generate a query
        query = generate_query(
            model_name=args.model,
            question=question,
            answer=answer,
            documents=documents,
        )

        # Step 7: Validate the query
        if not validate_query(query):
            continue  # Skip if validation fails

        # Add the validated example to the dataset
        generated_data["question"].append(question)
        generated_data["answer"].append(answer)
        generated_data["documents"].append(documents)
        generated_data["query"].append(query)

        example_count += 1
        print(f"Generated {example_count}/{args.num_examples} examples.")

    # Save the dataset
    output_dataset = Dataset.from_dict(generated_data)
    output_dataset.save_to_disk(args.output_dataset)
    print(f"Generated dataset saved at {args.output_dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset with multimodal QA examples.")
    parser.add_argument("--model", type=str, default="gpt", help="Model to use for generation (default: gpt)")
    parser.add_argument("--num-few-shot", type=int, default=1, help="Number of few-shot examples to use (default: 1)")
    parser.add_argument("--num-examples", type=int, default=5000, help="Number of examples to generate (default: 5000)")
    parser.add_argument(
        "--output-dataset",
        type=str,
        default="FM2DS/data/generated_data/synth",
        help="Output dataset path (default: FM2DS/data/generated_data/synth)",
    )

    args = parser.parse_args()
    main(args)
