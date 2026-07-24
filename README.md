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
