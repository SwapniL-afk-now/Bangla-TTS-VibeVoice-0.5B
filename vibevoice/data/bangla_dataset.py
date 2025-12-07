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
    def __init__(self, processor, model, target_sr=24000, n_mels=64, hop_length=320):
        """
        Args:
            processor: VibeVoiceProcessor
            model: VibeVoiceForTraining (possibly wrapped in PEFT)
            target_sr: Sample rate (24kHz for VibeVoice)
            n_mels: Number of mel bins (64 to match vae_dim)
            hop_length: Hop length for STFT (320 = 24000/75 frames per second matches ~7.5Hz rate)
        """
        self.processor = processor
        self.model = model
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        
        # Get the actual underlying model (unwrap PEFT if needed)
        self._base_model = self._get_base_model(model)
        
        # Import librosa for mel computation
        import librosa
        self.librosa = librosa
    
    def _get_base_model(self, model):
        """Unwrap PEFT/LoRA model to get the underlying VibeVoiceStreamingModel"""
        # Check if it's a PEFT model
        if hasattr(model, 'base_model'):
            # PeftModel -> LoraModel -> VibeVoiceForTraining -> VibeVoiceStreamingModel
            inner = model.base_model
            if hasattr(inner, 'model'):
                inner = inner.model
            if hasattr(inner, 'model'):
                inner = inner.model
            return inner
        # Check if it's VibeVoiceForTraining directly
        elif hasattr(model, 'model'):
            return model.model
        return model
    
    def _compute_mel_spectrogram(self, audio):
        """Compute mel-spectrogram as pseudo-latent representation.
        
        Note: VibeVoice uses a custom VAE encoder that is not shipped with the model.
        We use mel-spectrograms as a substitute for fine-tuning, which provides
        a similar acoustic representation at the target frame rate.
        """
        # Compute mel-spectrogram
        mel = self.librosa.feature.melspectrogram(
            y=audio,
            sr=self.target_sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=1024,
            power=1.0  # Magnitude spectrogram
        )
        # Convert to log scale (add small epsilon to avoid log(0))
        mel_db = self.librosa.amplitude_to_db(mel, ref=1.0, top_db=80.0)
        
        # Normalize to roughly [-1, 1] range (mel_db is typically in [-80, 0] range)
        mel_normalized = (mel_db + 40) / 40  # Shift to [-1, 1] approximately
        
        return mel_normalized  # Shape: (n_mels, T)

    def __call__(self, features):
        text_list = [f["text"] for f in features]
        audio_list = [f["audio"] for f in features]
        
        # Process Text
        scripts = [f"Speaker 0: {t}" for t in text_list]
        
        batch = self.processor(
            text=scripts,
            voice_samples=None,
            return_tensors="pt",
            padding=True
        )
        
        # Process Audio (Target)
        # Compute mel-spectrograms as pseudo-latents
        latents_list = []
        for audio in audio_list:
            # Normalize audio if needed
            if self.processor.db_normalize:
                audio = self.processor.audio_normalizer(audio)
            
            # Convert to numpy if needed
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()
            audio = np.asarray(audio, dtype=np.float32)
            
            # Compute mel-spectrogram
            mel = self._compute_mel_spectrogram(audio)  # (n_mels, T)
            
            # Convert to tensor and transpose to (T, D) for padding
            latent = torch.tensor(mel.T, dtype=torch.float32)  # (T, n_mels)
            latents_list.append(latent)
                
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
