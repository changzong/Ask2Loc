<div align="center">
  <h2>Ask2Loc: Learning to Locate Instructional Visual Answers<br>by Asking Questions</h2>
  <p>🧠 Instructional Visual Answer Localization | 🤖 Large and Pre-trained Language Models | 📊 Human Computer Interactions  </p>
</div>

## ✨ Overall Framework
<img src="pics/framework.png" width="60%" />

We propose Ask2Loc, an interactive visual answer localization framework that identifies precise video segments to answer a user question by acquiring auxiliary knowledge through simulating multiple interactions via asking formats. The top-level framework consists of three primary phases as shown in the above figure.

## 💡 Interactive and Learning Modules
<img src="pics/method.png" width="80%" />

- Chatting for Intention Awareness: Given that instructional videos often contain extensive domain knowledge that users are unfamiliar with, which leads to vague initial queries, this work leverages large language models (LLMs) to simulate interactive dialogue, progressively refining user intent through follow-up questioning and thus provide user-expected system responses.

- Rewriting for Description Completeness: The In-VAL process faces two forms of semantic incompleteness: incomplete subtitle expressions within video segments and a semantic gap between prior QA dialogue and actual intent. These issues can be effectively addressed through LLM-based rewriting that improves linguistic completeness and alignment between user input and video content.

- Searching for Context Expansion: To simulate human-like localization behavior, we propose a context expansion strategy that leverages a fine-tuned pre-trained language model (PLM) to identify semantically similar video segments to enhance the understanding and assessment of a given segment. This method is inspired from embedding-based retrieval in retrieval-augmented generation (RAG) systems.

- Learning for Answer Location Detection: We formulate the task of identifying whether each video segment falls within the answer span as a classification problem, where visual features are projected into the same space as textual features, fused with contextual encodings via a PLM, and jointly optimized through PLM-based fine-tuning using ground-truth and predicted location labels.

## 📕 Dataset
- For the video subtitles and visual features downloading, please download from our [GoogleDrive](https://anonymous.4open.science/r/Ask2Loc-480F) (update later)

- For the In-VAL datasets including questions, descriptions, context, and visual locations, please redirect to our [DatasetFolder](https://anonymous.4open.science/r/Ask2Loc-480F/dataset/)

## 🚀 Usage

### 🛠️ Train
1. Install Requirements

2. Setup Training Configuration
```vim config.py```


3. Run Training
```python train.py```

### 📜 Evaluation
```python evaluate.py```


## 📂 Checkpoints
Please download from our [GoogleDrive](https://anonymous.4open.science/r/Ask2Loc-480F) (update later)

## 🕹️ Demo
