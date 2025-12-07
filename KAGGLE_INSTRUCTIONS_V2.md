# Running VibeVoice Bangla TTS (Reference Implementation) on Kaggle

This guide uses the **reference implementation** from VibeVoice-finetuning with dual loss (CE + diffusion), proper token interleaving, and voice prompt support.

## 🚀 Quick Setup

### Cell 1: Clone & Install
```python
!git clone https://github.com/SwapniL-afk-now/Bangla-TTS-VibeVoice-0.5B.git
%cd Bangla-TTS-VibeVoice-0.5B

# Install dependencies
!pip install -e .
!pip install --upgrade transformers==4.51.3 accelerate diffusers peft datasets resampy librosa soundfile
```

### Cell 2: Run Training
```python
!python finetune_vibevoice_lora.py \
    --model_name_or_path "aoi-ot/VibeVoice-Large" \
    --processor_name_or_path vibevoice_v2/processor \
    --dataset_name "samikhan121/bangla_tts_iitm" \
    --text_column_name text \
    --audio_column_name audio \
    --output_dir /kaggle/working/bangla_tts_output \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2.5e-5 \
    --num_train_epochs 5 \
    --logging_steps 10 \
    --save_steps 500 \
    --report_to none \
    --remove_unused_columns False \
    --do_train \
    --gradient_clipping \
    --ddpm_batch_mul 4 \
    --diffusion_loss_weight 1.4 \
    --train_diffusion_head True \
    --ce_loss_weight 0.04 \
    --voice_prompt_drop_rate 0.2 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --max_grad_norm 0.8
```

---

## 🔧 Key Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--train_diffusion_head` | Full fine-tune diffusion head | False |
| `--train_connectors` | Train acoustic/semantic connectors | False |
| `--lora_r` | LoRA rank | 8 |
| `--ce_loss_weight` | Cross-entropy loss weight | 0.04 |
| `--diffusion_loss_weight` | Diffusion loss weight | 1.4 |
| `--voice_prompt_drop_rate` | Drop rate for voice prompts (regularization) | 0.2 |
| `--ddpm_batch_mul` | DDPM batch multiplier | 4 |

---

## 📝 Differences from Previous Approach

| Feature | Old (our code) | New (reference) |
|---------|----------------|-----------------|
| Loss | Diffusion only | CE + Diffusion (weighted) |
| Audio encoding | EnCodec (external) | acoustic_tokenizer.encode() |
| Token handling | No placeholders | speech_diffusion_id placeholders |
| Voice prompts | Not supported | Supported with drop rate |
| EMA | Not implemented | EMA on diffusion head |

---

## 💾 Save Model
```python
!zip -r bangla_tts_v2.zip /kaggle/working/bangla_tts_output
```
