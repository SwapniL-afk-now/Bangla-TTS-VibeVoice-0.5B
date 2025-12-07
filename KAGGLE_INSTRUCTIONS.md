# Running VibeVoice Bangla TTS on Kaggle

Complete guide for fine-tuning VibeVoice for Bangla Text-to-Speech on Kaggle T4 GPUs.

## 🚀 Quick Setup

### Cell 1: Clone & Install
```python
# Clone repository
!git clone https://github.com/SwapniL-afk-now/Bangla-TTS-VibeVoice-0.5B.git
%cd Bangla-TTS-VibeVoice-0.5B

# Install dependencies
!pip install -e .
!pip install --upgrade accelerate transformers diffusers

# IMPORTANT: Install EnCodec + compatible datasets version
!pip install encodec datasets==2.21.0 librosa soundfile
```

### Cell 2: Login to Hugging Face (Optional)
```python
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(token=hf_token)
```

---

## 📚 Multi-Stage Training

### Stage 1: Latent Space Alignment (Start Here)
```python
!python train_vibevoice.py \
    --model_path "PTE-VibeVoice/VibeVoice-TTS" \
    --dataset_name "samikhan121/bangla_tts_iitm" \
    --training_stage 1 \
    --output_dir "/kaggle/working/stage1-checkpoint" \
    --batch_size 2 \
    --num_epochs 5 \
    --learning_rate 1e-4
```

### Stage 2: Bangla Acoustic Adaptation
```python
!python train_vibevoice.py \
    --model_path "/kaggle/working/stage1-checkpoint" \
    --dataset_name "samikhan121/bangla_tts_iitm" \
    --training_stage 2 \
    --use_lora --lora_rank 8 \
    --output_dir "/kaggle/working/stage2-checkpoint" \
    --batch_size 2 \
    --num_epochs 10 \
    --learning_rate 5e-5
```

### Stage 3: Text-Speech Alignment
```python
!python train_vibevoice.py \
    --model_path "/kaggle/working/stage2-checkpoint" \
    --dataset_name "samikhan121/bangla_tts_iitm" \
    --training_stage 3 \
    --use_lora --lora_rank 16 \
    --output_dir "/kaggle/working/stage3-checkpoint" \
    --batch_size 2 \
    --num_epochs 5 \
    --learning_rate 2e-5
```

---

## 💾 Save Model
```python
!zip -r bangla_tts_model.zip /kaggle/working/stage3-checkpoint
```

---

## ⚡ Training Stages Explained

| Stage | Purpose | Duration |
|-------|---------|----------|
| 1 | Bridge EnCodec→VibeVoice latent space | ~2-4 hrs |
| 2 | Learn Bangla acoustic patterns | ~4-8 hrs |
| 3 | Improve text→speech alignment | ~3-6 hrs |
| 4 | Final polish (optional) | ~1-2 hrs |

## 🔧 Troubleshooting

**OOM Error?** → Reduce `--batch_size` to 1

**torchcodec Error?** → Ensure `datasets==2.21.0` is installed

**Model won't load?** → Try adding `--ignore_mismatched_sizes`
