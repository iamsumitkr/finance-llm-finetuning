# Finance LLM Fine-Tuning with LoRA (Llama-3 8B)

This project demonstrates domain-specific fine-tuning of a large
open-source language model (**Llama-3 8B Instruct**) on finance
instruction data using **LoRA (Low-Rank Adaptation)** with limited GPU
resources (Google Colab Free).

The goal of this project was to understand practical LLM fine-tuning
workflows including quantization, parameter-efficient training,
evaluation, and model comparison.

------------------------------------------------------------------------

## 🚀 Project Overview

A finance-focused instruction dataset (100 samples) was created and used
to fine-tune Llama-3 8B using LoRA. The training was performed with
8-bit quantization to fit within Colab's VRAM constraints.

Only \~6.8M parameters (\~0.08% of total model size) were trained while
keeping the base model weights frozen.

This project demonstrates a complete LLM engineering pipeline:

-   Dataset creation\
-   Parameter-efficient fine-tuning\
-   Quantized training\
-   Baseline vs fine-tuned evaluation

------------------------------------------------------------------------

## 🔧 Technical Stack

-   **Model:** Llama-3 8B Instruct\
-   **Fine-Tuning Method:** LoRA (PEFT)\
-   **Quantization:** 8-bit (BitsAndBytes)\
-   **Frameworks:** Hugging Face Transformers, PEFT, Datasets\
-   **Training Environment:** Google Colab (Free Tier GPU)\
-   **Language:** Python

------------------------------------------------------------------------

## 📊 Training Configuration

-   Sequence Length: 256\
-   Gradient Accumulation: 8\
-   Max Steps: 300\
-   Learning Rate: 2e-4\
-   Trainable Parameters: \~6.8M

------------------------------------------------------------------------

## 📈 Evaluation

The fine-tuned LoRA model was evaluated against the base Llama-3 8B
model on 10 held-out finance prompts (not included in training data).

### Observed Differences

  Metric                 Base Model             LoRA Fine-Tuned
  ---------------------- ---------------------- -------------------------------------
  Financial vocabulary   Good                   More structured and domain-specific
  Conciseness            Verbose explanations   Clear and concise responses
  Formula usage          Present but lengthy    More direct and structured
  Tone                   Generic educational    Professional finance-style

### Example Comparison

**Prompt:** Explain Return on Equity (ROE)

-   **Base Model:** Provided a long textbook-style explanation with
    example calculation and additional commentary.
-   **LoRA Model:** Delivered a concise, structured definition including
    the formula.

**Observation:**\
The LoRA model demonstrated improved clarity and professional tone,
while the base model tended to over-explain.

Overall, fine-tuning successfully shifted the model's response style
toward more concise and finance-oriented answers.

------------------------------------------------------------------------

## 📂 Repository Structure

finance-llm-finetuning/\
│\
├── train_lora.py\
├── requirements.txt\
│\
├── data/\
│ └── finance_instructions.jsonl\
│\
├── evaluation/\
│ ├── base_results.json\
│ └── lora_results.json\
│\
└── README.md

------------------------------------------------------------------------

## ⚠️ Notes

-   LoRA adapter weights (`lora_finance/`) are not included due to size.
-   Base model access requires accepting Meta's license on Hugging Face.
-   Evaluation results are stored in the `evaluation/` directory.

------------------------------------------------------------------------

## 🎯 Key Learnings

-   Practical hardware constraints (VRAM limits) strongly influence LLM
    training design.
-   LoRA effectively modifies response style with minimal parameter
    updates.
-   Quantization enables experimentation with large models on limited
    hardware.
-   Proper evaluation is essential to validate behavioral shifts after
    fine-tuning.

------------------------------------------------------------------------

## 📌 Purpose

This project was built to gain hands-on experience with:

-   Parameter-efficient fine-tuning (PEFT)\
-   Quantized LLM training\
-   Instruction tuning\
-   Model evaluation and comparison workflows

It demonstrates a complete mini LLM engineering workflow from dataset
creation to training and evaluation.
