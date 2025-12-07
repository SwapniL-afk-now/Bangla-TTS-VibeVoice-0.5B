# Running VibeVoice Fine-tuning on Kaggle

Since you are modifying the code locally, the best workflow is to push your changes to GitHub and then clone them in Kaggle.

## Step 1: Push Code to GitHub
1. Create a new repository on GitHub (e.g., `vibevoice-bangla`).
2. Push this entire `VibeVoice-main` folder to that repository.
   ```bash
   git init
   git add .
   git commit -m "Initial commit with training scripts"
   git remote add origin https://github.com/YOUR_USERNAME/vibevoice-bangla.git
   git push -u origin main
   ```

## Step 2: Set up Kaggle Notebook
1. Create a new Notebook.
2. In the "Session Options" (right sidebar), set **Accelerator** to **GPU T4 x2**.

## Step 3: Notebook Cells
Copy the following blocks into your Kaggle Notebook cells.

### Cell 1: Installation
```python
# Clone your modified repository
!git clone https://github.com/SwapniL-afk-now/Bangla-TTS-VibeVoice-0.5B.git
%cd Bangla-TTS-VibeVoice-0.5B

# Install dependencies
!pip install -e .
!pip install --upgrade accelerate transformers diffusers

# IMPORTANT: Install EnCodec for audio encoding + older datasets
!pip install encodec datasets==2.21.0 librosa soundfile
```

### Cell 2: Login to Hugging Face (Optional)
If using a gated model or private dataset, login here.
```python
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

# Recommendation: Store your HF token in Kaggle Secrets (name it 'HF_TOKEN')
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(token=hf_token)
```

### Cell 3: Run Training with LoRA
This command uses the T4 GPUs. LoRA is enabled to save memory.
```python
# Run the training script
# Adjust batch_size if needed (e.g., 1 or 2 per GPU)
!python train_vibevoice.py \
    --model_path "PTE-VibeVoice/VibeVoice-TTS" \
    --dataset_name "mozilla-foundation/common_voice_17_0" \
    --dataset_config "bn" \
    --dataset_split "train" \
    --output_dir "/kaggle/working/vibevoice-bangla-model" \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --use_lora
```

### Cell 4: Save Outputs
If training finishes, zipping the model makes it easier to download.
```python
!zip -r model_output.zip /kaggle/working/vibevoice-bangla-model
```
