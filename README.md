# <p align="center">FM<sup>2</sup>DS: Few-Shot Multimodal Multihop Data Synthesis with Knowledge Distillation for Question Answering</p>

<p align="center">
  <br>
  <a href="https://www.arxiv.org/abs/2412.07030"><img alt="Paper" src="https://img.shields.io/badge/📃-Paper-808080"></a>
  <a href="https://fm2ds.github.io/"><img alt="Website" src="https://img.shields.io/badge/%F0%9F%8C%90-Website-008080"></a>
  <a href="https://huggingface.co/datasets/AmirhosseinAbaskohi/M2QA_Bench"><img alt="Huggingface" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Benchmark-yellow"></a>
</p>

## Abstract
Multimodal multihop question answering is a complex task that requires reasoning over multiple sources of information, such as images and text, to answer questions. While there has been significant progress in visual question answering,
the multihop setting remains unexplored due to the lack of high-quality datasets. Current methods focus on single-hop question answering or a single modality,
which makes them unsuitable for real-world scenarios such as analyzing multimodal educational materials, summarizing lengthy academic articles, or interpreting scientific studies that combine charts, images,
and text. To address this gap, we propose a novel methodology, introducing the first framework for creating a high-quality dataset that enables training models for multimodal multihop question answering.
Our approach consists of a 5-stage pipeline that involves acquiring relevant multimodal documents from Wikipedia, synthetically generating high-level questions and answers, and validating them through rigorous criteria to ensure quality data.
We evaluate our methodology by training models on our synthesized dataset and testing on two benchmarks, our results demonstrate that, with an equal sample size,
models trained on our synthesized data outperform those trained on human-collected data by 1.9 in exact match (EM) on average.
We believe our data synthesis method will serve as a strong foundation for training and evaluating multimodal multihop question answering models.

<div align="center">
  <img src="https://github.com/user-attachments/assets/d1c8fde6-d02f-4e6b-b224-e327acab7c93" alt="image">
</div>


In contrast to traditional datasets that depend on human annotators, templates, and information snippets as sources, FM<sup>2</sup>DS is a fully automated approach that utilizes complete documents as its sources.
FM<sup>2</sup>DS incorporates validation steps to ensure that the generated questions are answerable, multimodal, and multihop.


## FM<sup>2</sup>DS
![image](https://github.com/user-attachments/assets/4c3c9afb-f4f6-43c3-8cf3-8a12fb2a1776)

The Five-Stage Pipeline for FM<sup>2</sup>DS. First we retrieve relevant documents from the Wikipedia dataset to create a pool of related documents based on hyperlinks and topics (Stage 1).
In Stage2, we select the few-shot samples from MultiModalQA (MMQA in the figure). Stage 3 focuses on generating and validating questions to make sure they are answerable, multihop, and multimodal.
In Stage 4, answers are generated and validated. Finally, in Stage 5 we generate queries related to the documents, which are also validated to ensure relevance and accuracy.

## M<sup>2</sup>QA-Bench
We also propose a benchmark, M<sup>2</sup>QA, to assess the LVLMs performance on a more complicated MMQA task with full documents. M<sup>2</sup>QA consists of 500 Q&A pairs,
each designed to challenge the model's ability to perform a complex reasoning task. The questions are not templated into a specific structure (as in some existing works like MultimodalQA),
instead, they are diverse and challenging. Additionally, answering the questions require access to full documents, where both information extraction and reasoning across different modalities (e.g., images and tables) are essential.

![image](https://github.com/user-attachments/assets/7a666536-3c5e-4f8d-b391-cee77a771476)

Multimodal multihop reasoning example from M<sup>2</sup>QA-Bench where the model compares the release dates of two albums, "Music from Big Pink" and "Imagine,"
using textual and visual cues. The documents are connected through their shared topic, "music," and the answer is determined as the title of the earlier-released album.

You can use this [link](https://github.com/ServiceNow/FM2DS/blob/main/M2QA_Bench.json) to access this benchmark.

## How to Run

This guide provides step-by-step instructions for running the FM²DS pipeline to synthesize multimodal multihop question answering data.

### Overview

**Important Note**: This project is designed specifically for **data synthesis**. The generated dataset can be used to train various multimodal models, but the actual model training is not included in this repository. For model training, please refer to each model's specific training approaches and documentation.

### Prerequisites

#### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended for LVLM inference)
- Sufficient storage space for datasets (~50GB+)

#### Dependencies
Install the required Python packages:

```bash
pip install datasets transformers torch tensorflow beautifulsoup4 requests scikit-learn numpy
```

For specific model APIs:
- **OpenAI GPT**: `pip install openai`
- **Anthropic Claude**: `pip install anthropic`
- **Local Llama**: `pip install vllm`

### Setup and Data Preparation

#### Step 1: Download Required Datasets

##### 1.1 Download WikiWeb2M Dataset
```bash
cd data/
bash download_wikiweb2m.sh
```

##### 1.2 Download MultiModalQA Training Data
```bash
cd create_few_shot_samples/
bash download_mmqa_train.sh
```

#### Step 2: Parse and Prepare Base Dataset

```bash
# Parse WikiWeb2M dataset and save as HuggingFace format
python data/parse_and_save_dataset.py
```

#### Step 3: Create Few-Shot Examples

```bash
# Create few-shot examples from MultiModalQA
python create_few_shot_samples/create_few_shot_from_multimodalqa.py
```

#### Step 4: Create Document Pool

```bash
# Create pools of related documents for multihop reasoning
python data/create_document_pool.py
```

### Running Data Synthesis

#### Configure Model Settings

Choose one of the following language models:

##### Option 1: OpenAI GPT (Recommended)
Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

##### Option 2: Anthropic Claude
Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

##### Option 3: Local Llama Model
Start the Llama server:
```bash
# For Llama 3.1
bash lvlm/llama/host_llama_3_1.sh

# For Llama 3.2
bash lvlm/llama/host_llama_3_2.sh
```

#### Generate Synthetic Dataset

Run the main data synthesis pipeline:

```bash
python src/create_dataset.py \
    --model gpt \
    --num-few-shot 1 \
    --num-examples 5000 \
    --output-dataset FM2DS/data/generated_data/synth
```

**Parameters:**
- `--model`: Choose from `gpt`, `claude`, or `llama`
- `--num-few-shot`: Number of few-shot examples (default: 1)
- `--num-examples`: Total number of examples to generate (default: 5000)
- `--output-dataset`: Output directory for generated dataset

### Generated Data Format

The synthesized dataset contains the following structure:

```json
{
    "question": "Which country is ranked lower in EuroCup Basketball Performance...",
    "answer": "France",
    "documents": [
        {
            "title": "Document Title",
            "content": [
                {"type": "text", "value": "Text content here..."},
                {"type": "image", "value": "http://example.com/image.jpg"}
            ]
        }
    ],
    "query": ["step-by-step reasoning process", "explanation of answer derivation"]
}
```

### Using the Data for Model Training

#### Important Training Considerations

**⚠️ Critical for Model Training**: When training multimodal models with this data, include **both the question-answer pairs AND the generated queries**. The queries contain step-by-step reasoning that is essential for teaching models multihop reasoning capabilities.

#### Data Conversion Scripts

Below are example Python scripts to convert the FM²DS data format for specific model training:

##### Example: Converting for InternVL2 Training

```python
# convert_for_internvl2.py
import json
from datasets import load_from_disk

def convert_fm2ds_to_internvl2(input_dataset_path, output_file):
    """
    Convert FM2DS dataset to InternVL2 training format
    """
    dataset = load_from_disk(input_dataset_path)
    converted_data = []
    
    for example in dataset:
        # Extract images from documents
        images = []
        text_content = ""
        
        for doc in example['documents']:
            for content in doc['content']:
                if content['type'] == 'image':
                    images.append(content['value'])
                elif content['type'] == 'text':
                    text_content += content['value'] + " "
        
        # Create InternVL2 format with question, answer, and reasoning
        reasoning_steps = " ".join(example['query']) if isinstance(example['query'], list) else example['query']
        
        internvl_example = {
            "id": f"fm2ds_{len(converted_data)}",
            "image": images[0] if images else None,  # InternVL2 typically uses single image
            "conversations": [
                {
                    "from": "human",
                    "value": f"Context: {text_content.strip()}\n\nQuestion: {example['question']}\n\nPlease provide step-by-step reasoning and then the final answer."
                },
                {
                    "from": "gpt", 
                    "value": f"Reasoning: {reasoning_steps}\n\nAnswer: {example['answer']}"
                }
            ]
        }
        converted_data.append(internvl_example)
    
    # Save in JSONL format
    with open(output_file, 'w') as f:
        for item in converted_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Converted {len(converted_data)} examples to {output_file}")

# Usage
convert_fm2ds_to_internvl2("FM2DS/data/generated_data/synth", "internvl2_training_data.jsonl")
```

##### Example: Converting for Generic VLM Training

```python
# convert_for_generic_vlm.py
import json
from datasets import load_from_disk

def convert_fm2ds_to_generic_vlm(input_dataset_path, output_file):
    """
    Convert FM2DS dataset to generic VLM training format
    """
    dataset = load_from_disk(input_dataset_path)
    converted_data = []
    
    for example in dataset:
        # Prepare multimodal input
        multimodal_input = {
            "text_documents": [],
            "images": [],
            "question": example['question'],
            "reasoning_steps": example['query'],
            "answer": example['answer']
        }
        
        for doc in example['documents']:
            text_parts = []
            for content in doc['content']:
                if content['type'] == 'text':
                    text_parts.append(content['value'])
                elif content['type'] == 'image':
                    multimodal_input['images'].append({
                        "url": content['value'],
                        "caption": ""  # Add caption if available
                    })
            
            if text_parts:
                multimodal_input['text_documents'].append({
                    "title": doc['title'],
                    "content": " ".join(text_parts)
                })
        
        converted_data.append(multimodal_input)
    
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"Converted {len(converted_data)} examples to {output_file}")

# Usage
convert_fm2ds_to_generic_vlm("FM2DS/data/generated_data/synth", "generic_vlm_training_data.json")
```

### Training Recommendations

1. **Include Reasoning Steps**: Always incorporate the generated queries/reasoning steps in your training data
2. **Multimodal Alignment**: Ensure your model can process both text and images from the documents
3. **Multihop Training**: Structure training to encourage step-by-step reasoning across multiple documents
4. **Validation**: Use the provided M²QA-Bench (`M2QA_Bench.json`) for evaluation

### Evaluation

Use the M²QA-Bench for evaluating trained models:

```python
import json

# Load benchmark
with open('M2QA_Bench.json', 'r') as f:
    benchmark = json.load(f)

# Each item contains:
# - question: The question to answer
# - answer: Ground truth answer  
# - modalities: Required modalities (text, image, table)
# - pages: Source Wikipedia pages
```

#### Performance Tips

- Use `--num-few-shot 3` for better generation quality
- Start with smaller `--num-examples` for testing
- Monitor validation success rates in the generation logs

## Citation

```
@misc{abaskohi2024fm2dsfewshotmultimodalmultihop,
      title={FM2DS: Few-Shot Multimodal Multihop Data Synthesis with Knowledge Distillation for Question Answering}, 
      author={Amirhossein Abaskohi and Spandana Gella and Giuseppe Carenini and Issam H. Laradji},
      year={2024},
      eprint={2412.07030},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.07030}, 
}
```
