# <p align="center">FM<sup>2</sup>DS: Few-Shot Multimodal Multihop Data Synthesis with Knowledge Distillation for Question Answering</p>

<p align="center">
  <br>
  <a href="#"><img alt="Paper" src="https://img.shields.io/badge/📃-Paper-808080"></a>
  <a href="https://fm2ds.github.io/"><img alt="Website" src="https://img.shields.io/badge/%F0%9F%8C%90-Website-008080"></a>
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
  <img src="https://github.com/user-attachments/assets/273fb949-214a-4c42-b3cd-c1eb058f1a82" alt="image">
</div>

In contrast to traditional datasets that depend on human annotators, templates, and information snippets as sources, FM<sup>2</sup>DS is a fully automated approach that utilizes complete documents as its sources.
FM<sup>2</sup>DS incorporates validation steps to ensure that the generated questions are answerable, multimodal, and multihop.


## FM<sup>2</sup>DS
![image](https://github.com/user-attachments/assets/f7fe381a-241c-4716-98cb-f65e9264bf05)

The Five-Stage Pipeline for FM<sup>2</sup>DS. First we retrieve relevant documents from the Wikipedia dataset to create a pool of related documents based on hyperlinks and topics (Stage 1).
In Stage2, we select the few-shot samples from MultiModalQA (MMQA in the figure). Stage 3 focuses on generating and validating questions to make sure they are answerable, multihop, and multimodal.
In Stage 4, answers are generated and validated. Finally, in Stage 5 we generate queries related to the documents, which are also validated to ensure relevance and accuracy.

## M<sup>2</sup>QA-Bench
We also propose a benchmark, M<sup>2</sup>QA, to assess the LVLMs performance on a more complicated MMQA task with full documents. M<sup>2</sup>QA consists of 500 Q&A pairs,
each designed to challenge the model's ability to perform a complex reasoning task. The questions are not templated into a specific structure (as in some existing works like MultimodalQA),
instead, they are diverse and challenging. Additionally, answering the questions require access to full documents, where both information extraction and reasoning across different modalities (e.g., images and tables) are essential.

![image](https://github.com/user-attachments/assets/b75a24d9-5372-45f5-8cb7-66dab532bfab)
Multimodal multihop reasoning example from M<sup>2</sup>QA-Bench where the model compares the release dates of two albums, "Music from Big Pink" and "Imagine,"
using textual and visual cues. The documents are connected through their shared topic, "music," and the answer is determined as the title of the earlier-released album.

You can use this [link](https://github.com/ServiceNow/FM2DS/blob/main/M2QA_Bench.json) to access this benchmark.

## Citation

```
Coming Soon
```