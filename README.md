# AI-Powered Document Intelligence using LoRA Fine-Tuning of Gemma 3

**Fine-tuned Gemma 3 (1B) with LoRA on the CORD receipt dataset to extract structured information from invoices/receipts.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/Gemma3-LoRA-Document-Intelligence/blob/main/notebooks/FineTune.ipynb)

## 🚀 Project Overview

This project demonstrates how to adapt a large language model (Gemma 3) for a **document intelligence** task using **parameter-efficient fine-tuning (LoRA)**. The model is trained to extract key fields (Total, Subtotal) from receipt text, converting raw OCR output into structured JSON.

**Key skills demonstrated:**
- Fine-tuning LLMs with LoRA (PEFT)
- Using Hugging Face Transformers & Datasets
- 4-bit quantization with BitsAndBytes
- Training on Google Colab T4 GPU
- Model evaluation and inference

## 🏗️ Architecture

Dataset (CORD) → Preprocessing → Prompt Formatting → Tokenization → Gemma 3 + LoRA → Fine-Tuning → Evaluation → Adapter Saving → Inference


## 📊 Dataset

We use the **CORD (Consolidated Receipts Dataset)** – a collection of 11,000 Indonesian receipts with annotations for fields like `total`, `sub_total`, `date`, etc.  
For this project, we focus on extracting `Total` and `Subtotal`.

- Dataset: [`naver-clova-ix/cord-v2`](https://huggingface.co/datasets/naver-clova-ix/cord-v2)

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/Gemma3-LoRA-Document-Intelligence.git
cd Gemma3-LoRA-Document-Intelligence
pip install -r requirements.txt
```

## Requirements (see requirements.txt):

transformers, datasets, peft, accelerate, bitsandbytes

trl, sentencepiece, evaluate, rouge-score

torch, huggingface_hub, matplotlib

## 🧪 Training
The training notebook notebooks/FineTune.ipynb contains the full pipeline:

Load and preprocess the CORD dataset.

Load Gemma 3 in 4‑bit quantization.

Apply LoRA (rank=16, alpha=32).

Train for 3 epochs on 808 samples.

Save the LoRA adapter.

Training Loss Curve:

https://screenshots/training_loss.png

Training Logs
Step	Training Loss	Validation Loss
100	0.690854	0.699268
200	0.639373	0.589094
300	0.550803	0.577801
303	0.550803	0.577776
The loss drops consistently, indicating successful fine‑tuning.

## 🔍 Inference
The inference notebook notebooks/Inference.ipynb loads the trained adapter and extracts fields from new receipt text.

Example:

Input:

text
Total: 45,500
Subtotal: 13,000
Output:

json
{
  "Total": "45,500",
  "Subtotal": "13,000"
}
https://screenshots/inference_output.png

## 📈 Results
The fine‑tuned model successfully extracts the Total and Subtotal fields from unseen receipt samples with high accuracy. The LoRA adapter achieves this with only ~0.15% of the parameters being trainable, making it efficient for deployment.

## 🧠 What I Learned
How to fine‑tune an LLM on a custom dataset using parameter‑efficient methods.

Practical use of the Hugging Face ecosystem (Transformers, PEFT, Datasets, Accelerate).

Handling large models in limited GPU memory via 4‑bit quantization.

Building a reproducible machine learning pipeline in Colab.

## 🚀 Future Work
Extend to more fields (Merchant, Date, Tax).

Train on a larger, multi‑language dataset.

Deploy as a web app using Gradio or Streamlit.

Use the model for real‑time document processing.

## 📁 Repository Structure

```
Gemma3-LoRA-Document-Intelligence/
├── notebooks/
│   ├── Gemma3_LoRA_FineTune.ipynb      # Training notebook
│   └── Inference.ipynb                 # Inference notebook
├── screenshots/
|   ├── json.png
|   ├── training.png
│   ├── training_loss.png               # Loss curve
│   └── inference_result/                # Extraction example
|       ├── inference_result_1.png
|       ├── inference_result_2.png
|       ├── inference_result_3.png
|       ├── inference_result_4.png
|       └── inference_result_5.png
├── requirements.txt
├── LICENSE
└── README.md
```

## 📝 License
This project is licensed under the MIT License – see the LICENSE file for details.

## 🙏 Acknowledgements

Hugging Face for Transformers, PEFT, and Datasets.

Google for the Gemma model.

CORD dataset by Clova AI.

## If you find this project useful, please give it a ⭐ on GitHub!
