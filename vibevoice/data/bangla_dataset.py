import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Audio
import numpy as np

class BanglaDataset(Dataset):
    def __init__(self, dataset_name, split="train", config_name=None, processor=None, target_sr=24000):
        """
        Args:
            dataset_name: Name of the HF dataset.
            split: Split to load (train/validation/test).
            config_name: Config name if any (e.g. 'bn').
            processor: VibeVoiceProcessor.
            target_sr: Target sampling rate (24kHz for VibeVoice).
        """
        self.processor = processor
        self.target_sr = target_sr
        
        print(f"Loading HF dataset: {dataset_name} ({config_name if config_name else 'default'}) split={split}")
        
        # Load dataset
        self.dataset = load_dataset(dataset_name, config_name, split=split)
        
        # Check for required columns
        if "audio" not in self.dataset.column_names:
            raise ValueError(f"Dataset {dataset_name} does not have an 'audio' column.")
        
        # Cast audio to target sample rate (uses librosa in datasets 2.21.0)
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=target_sr))
        
        # Find text column
        self.text_col = "text"
        if "text" not in self.dataset.column_names:
            candidates = ["sentence", "transcription", "normalized_text"]
            for c in candidates:
                if c in self.dataset.column_names:
                    self.text_col = c
                    break
            else:
                raise ValueError(f"Dataset {dataset_name} does not have a 'text' column and no alternatives found.")
        
        print(f"Dataset loaded with {len(self.dataset)} items")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item[self.text_col]
        audio_data = item['audio']
        
        # audio_data is a dict with 'array' and 'sampling_rate' (decoded by datasets)
        audio = audio_data['array']
        
        return {
            "text": text,
            "audio": audio
        }

class VibeVoiceDataCollator:
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model # Need model for VAE if doing online encoding

    def __call__(self, features):
        text_list = [f["text"] for f in features]
        audio_list = [f["audio"] for f in features]
        
        # Process Text
        # We need to manually construct the prompt structure:
        # System + Text + "Speech Output:"
        # The processor `__call__` does this:
        # text input -> "Speaker X: text" -> ...
        # But here we just have raw text.
        # Let's treat it as a single speaker "0".
        
        # Format for processor:
        # script = "Speaker 0: " + text
        scripts = [f"Speaker 0: {t}" for t in text_list]
        
        batch = self.processor(
            text=scripts,
            voice_samples=None, # No voice prompt for training (or maybe use snippet?)
            return_tensors="pt",
            padding=True
        )
        
        # Clean up batch keys
        # `batch` has `input_ids`, `attention_mask`. 
        # `processor` adds "Speech output:" and `speech_start_id` at the end. 
        # This matches our training expectation! 
        
        # Process Audio (Target)
        # Encode to Latents
        # We need to pad audio first? Or encode then pad latents?
        # Encode -> Pad Latents is better.
        
        latents_list = []
        with torch.no_grad():
            for audio in audio_list:
                # Normalize via processor utils?
                if self.processor.db_normalize:
                     audio = self.processor.audio_normalizer(audio)
                
                # Convert to tensor (1, 1, T)
                audio_t = torch.tensor(audio).unsqueeze(0).unsqueeze(0).to(self.model.device).float()
                
                # Encode
                dist = self.model.acoustic_tokenizer.encode(audio_t)
                latent = dist.sample() # (1, D, T_lat) or (1, T_lat, D)? 
                # Check VAE dim ordering. 
                # Usually (B, C, T).
                
                # Permute to (T, D) for easier padding
                latent = latent.squeeze(0).transpose(0, 1) 
                latents_list.append(latent.cpu())
                
        # Pad Latents
        from torch.nn.utils.rnn import pad_sequence
        input_latents = pad_sequence(latents_list, batch_first=True, padding_value=0.0) # (B, T, D)
        
        # Add to batch
        batch["speech_values"] = input_latents
        
        # Attention mask for speech?
        # We can deduce from length? Or create it explicitly.
        speech_lens = [l.shape[0] for l in latents_list]
        max_speech_len = max(speech_lens)
        speech_mask = torch.zeros((len(features), max_speech_len), dtype=torch.long)
        for i, l in enumerate(speech_lens):
            speech_mask[i, :l] = 1
            
        batch["speech_attention_mask"] = speech_mask
        
        # Cleanup extra keys from processor
        if "speech_input_mask" in batch: 
             del batch["speech_input_mask"] # We don't use this mixed mask in our custom forward
        if "speech_tensors" in batch:
             del batch["speech_tensors"]
        if "speech_masks" in batch:
             del batch["speech_masks"]
             
        return batch
