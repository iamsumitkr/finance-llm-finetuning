# Finance LLM Fine-Tuning with LoRA (Llama-3 8B)

This project demonstrates fine-tuning an open-source large language model
(**Llama-3 8B Instruct**) on finance-specific instruction data using
**LoRA (Low-Rank Adaptation)** on limited GPU resources (Google Colab Free).

## Highlights
- Fine-tuned Llama-3 8B using LoRA (~6.8M trainable parameters)
- Used 8-bit quantization with CPU/GPU offloading
- Ran entirely on Google Colab Free
- Handled real-world issues like VRAM limits and library version mismatches

## Files
- `train_lora.py` – LoRA fine-tuning script
- `inference.py` – Run inference with LoRA adapter
- `data/finance_instructions.jsonl` – Sample training data

## Notes
- LoRA adapter weights (`lora_finance/`) are not included due to size
- Base model access requires accepting Meta’s license on Hugging Face

## Purpose
This project was built to learn practical LLM fine-tuning workflows.
