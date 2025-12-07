# VibeVoice Bangla TTS Fine-tuning Guide

This repository contains scripts to fine-tune VibeVoice for Bangla Text-to-Speech.

## 🎯 Multi-Stage Training Approach

Since the original VibeVoice VAE encoder is not open-sourced, we use **EnCodec** as an alternative audio encoder with a projection layer. Training is done in 4 stages:

| Stage | Purpose | What's Trained |
|-------|---------|----------------|
| **1** | Latent Space Alignment | Projection layer, Acoustic connector, Diffusion head |
| **2** | Bangla Acoustic Adaptation | TTS LM (LoRA) + Diffusion head |
| **3** | Text-Speech Alignment | Both LMs (LoRA) + Acoustic connector |
| **4** | End-to-End Polish | All components (low LR) |

## 📦 Components

- `train_vibevoice.py` - Main training script with multi-stage support
- `vibevoice/modular/modeling_vibevoice_training.py` - Training model wrapper
- `vibevoice/data/bangla_dataset.py` - Dataset loader with EnCodec encoding

## 🚀 Quick Start

### Stage 1: Latent Space Alignment
```bash
python train_vibevoice.py \
    --model_path "PTE-VibeVoice/VibeVoice-TTS" \
    --dataset_name "samikhan121/bangla_tts_iitm" \
    --training_stage 1 \
    --num_epochs 5 \
    --learning_rate 1e-4 \
    --batch_size 2
```

### Stage 2: Bangla Acoustic Adaptation
```bash
python train_vibevoice.py \
    --model_path "./vibevoice-bangla-checkpoints" \
    --dataset_name "samikhan121/bangla_tts_iitm" \
    --training_stage 2 \
    --use_lora --lora_rank 8 \
    --num_epochs 10 \
    --learning_rate 5e-5
```

### Stage 3: Text-Speech Alignment
```bash
python train_vibevoice.py \
    --model_path "./vibevoice-bangla-checkpoints" \
    --dataset_name "samikhan121/bangla_tts_iitm" \
    --training_stage 3 \
    --use_lora --lora_rank 16 \
    --num_epochs 5 \
    --learning_rate 2e-5
```

## 💻 Hardware Requirements

- **VRAM**: 15GB+ (T4 GPU works with LoRA + batch_size=2)
- **Disk**: ~5GB for model + dataset cache

## 📊 Dataset Format

Your Hugging Face dataset should have:
- `audio`: Audio column (automatically resampled to 24kHz)
- `text` (or `sentence`/`transcription`): Text column

## 🔧 Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | HF model ID or local path | Required |
| `--dataset_name` | HF dataset name | Required |
| `--dataset_config` | Dataset config (e.g., 'bn') | None |
| `--training_stage` | Training stage (1-4) | 1 |
| `--use_lora` | Enable LoRA fine-tuning | False |
| `--lora_rank` | LoRA rank | 8 |
| `--num_epochs` | Training epochs | 3 |
| `--learning_rate` | Learning rate | 1e-4 |
| `--batch_size` | Batch size per device | 2 |

## 📝 Notes

- Stage 1 uses EnCodec encoder (128→64 projection) since original VAE encoder isn't available
- Qwen tokenizer already supports Bangla text
- Use `--report_to tensorboard` to enable logging
