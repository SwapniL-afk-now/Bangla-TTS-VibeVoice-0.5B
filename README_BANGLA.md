# VibeVoice Bangla Fine-tuning Guide

This directory contains the necessary scripts to fine-tune VibeVoice on a Bangla TTS dataset.

## Components
- `vibevoice/modular/modeling_vibevoice_training.py`: Model wrapper that adds diffusion loss computation.
- `vibevoice/data/bangla_dataset.py`: Dataset loader that reads audio and text metadata.
- `train_vibevoice.py`: Main training script using HuggingFace Trainer.

## Prerequisites
1. **Model Weights**: A pretrained VibeVoice (Qwen2.5 based) model checkpoint.
2. **Dataset**: A Hugging Face dataset (public or private) with:
   - `audio`: An audio column.
   - `text` (or `sentence`/`transcription`): A text column.

## How to Run

 You do **not** need to manually download model weights. You can simply pass the Hugging Face model ID (e.g., `PTE-VibeVoice/VibeVoice-TTS`).

```bash
python train_vibevoice.py \
    --model_path "PTE-VibeVoice/VibeVoice-TTS" \
    --dataset_name "mozilla-foundation/common_voice_17_0" \
    --dataset_config "bn" \
    --dataset_split "train" \
    --output_dir ./vibevoice_bangla_v1 \
    --batch_size 2 \
    --num_epochs 5 \
    --learning_rate 1e-4 \
    --use_lora
```

## Hardware Requirements
- VRAM: >24GB recommended for 1.5B model + Variational Diffusion Head.
- Reduce `--batch_size` if OOM occurs.

## Notes
- The training script uses the existing `speech_scaling_factor` and `speech_bias_factor` from the pretrained model. Ensure your base model has these buffers initialized (valid checkpoints will).
- The dataset loader currently handles raw audio normalization via the processor.
