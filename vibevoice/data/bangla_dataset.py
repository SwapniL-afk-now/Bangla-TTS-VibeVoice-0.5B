import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Audio
import numpy as np

class BanglaDataset(Dataset):
    def __init__(self, dataset_name, split="train", config_name=None, processor=None, target_sr=24000, speaker_ids=None):
        """
        Args:
            dataset_name: Name of the HF dataset.
            split: Split to load (train/validation/test).
            config_name: Config name if any (e.g. 'bn').
            processor: VibeVoiceProcessor.
            target_sr: Target sampling rate (24kHz for VibeVoice).
            speaker_ids: List of speaker IDs to filter (e.g., [0, 1]). None means all speakers.
        """
        self.processor = processor
        self.target_sr = target_sr
        
        print(f"Loading HF dataset: {dataset_name} ({config_name if config_name else 'default'}) split={split}")
        
        # Load dataset
        self.dataset = load_dataset(dataset_name, config_name, split=split)
        
        # Filter by speaker_id if specified
        if speaker_ids is not None and "speaker_id" in self.dataset.column_names:
            print(f"Filtering dataset for speaker_ids: {speaker_ids}")
            original_len = len(self.dataset)
            self.dataset = self.dataset.filter(lambda x: x["speaker_id"] in speaker_ids)
            print(f"Filtered: {original_len} -> {len(self.dataset)} samples")
        elif speaker_ids is not None:
            print(f"Warning: speaker_ids specified but 'speaker_id' column not found in dataset")
        
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
    def __init__(self, processor, model, target_sr=24000, use_encodec=True):
        """
        Args:
            processor: VibeVoiceProcessor
            model: VibeVoiceForTraining (possibly wrapped in PEFT)
            target_sr: Sample rate (24kHz for VibeVoice)
            use_encodec: Whether to use EnCodec for encoding (recommended)
        """
        self.processor = processor
        self.model = model
        self.target_sr = target_sr
        self.use_encodec = use_encodec
        
        # Get the actual underlying model (unwrap PEFT if needed)
        self._base_model = self._get_base_model(model)
        self.device = next(self._base_model.parameters()).device
        
        # Initialize EnCodec encoder if requested
        if use_encodec:
            self._init_encodec_encoder()
        else:
            import librosa
            self.librosa = librosa
    
    def _init_encodec_encoder(self):
        """Initialize EnCodec encoder with projection layer for dimension matching."""
        try:
            from encodec import EncodecModel
            from encodec.utils import convert_audio
            
            # Load 24kHz bandwidth 24 model
            self.encodec = EncodecModel.encodec_model_24khz()
            self.encodec.set_target_bandwidth(6.0)  # Use 6kbps for good quality
            self.encodec.eval()
            self.encodec.to(self.device)
            
            # EnCodec outputs 128-dim latents, VibeVoice needs 64-dim
            # Create a simple projection layer
            self.encodec_projection = torch.nn.Linear(128, 64).to(self.device)
            # Initialize with PCA-like dimensionality reduction (just use first 64 dims initially)
            with torch.no_grad():
                self.encodec_projection.weight.data = torch.eye(64, 128).to(self.device)
                self.encodec_projection.bias.data.zero_()
            
            self.convert_audio = convert_audio
            print("EnCodec encoder initialized successfully (128->64 projection)")
            
        except ImportError:
            print("Warning: EnCodec not available, falling back to mel-spectrograms")
            self.use_encodec = False
            import librosa
            self.librosa = librosa
    
    def _get_base_model(self, model):
        """Unwrap PEFT/LoRA model to get the underlying VibeVoiceStreamingModel"""
        if hasattr(model, 'base_model'):
            inner = model.base_model
            if hasattr(inner, 'model'):
                inner = inner.model
            if hasattr(inner, 'model'):
                inner = inner.model
            return inner
        elif hasattr(model, 'model'):
            return model.model
        return model
    
    def _encode_audio_encodec(self, audio):
        """Encode audio using EnCodec encoder + projection."""
        # Convert to tensor
        if isinstance(audio, np.ndarray):
            audio_t = torch.tensor(audio, dtype=torch.float32)
        else:
            audio_t = audio.float()
        
        # Ensure shape is (1, 1, T) for EnCodec
        if audio_t.dim() == 1:
            audio_t = audio_t.unsqueeze(0).unsqueeze(0)
        elif audio_t.dim() == 2:
            audio_t = audio_t.unsqueeze(0)
        
        audio_t = audio_t.to(self.device)
        
        # Get encoder output (continuous latents before quantization)
        with torch.no_grad():
            # EnCodec encoder returns continuous latent
            emb = self.encodec.encoder(audio_t)  # (1, 128, T_latent)
            # Project 128 -> 64
            emb = emb.permute(0, 2, 1)  # (1, T, 128)
            emb = self.encodec_projection(emb)  # (1, T, 64)
        
        return emb.squeeze(0)  # (T, 64)
    
    def _encode_audio_mel(self, audio):
        """Fallback: encode audio using mel-spectrogram."""
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        audio = np.asarray(audio, dtype=np.float32)
        
        mel = self.librosa.feature.melspectrogram(
            y=audio, sr=self.target_sr, n_mels=64,
            hop_length=320, n_fft=1024, power=1.0
        )
        mel_db = self.librosa.amplitude_to_db(mel, ref=1.0, top_db=80.0)
        mel_normalized = (mel_db + 40) / 40
        return torch.tensor(mel_normalized.T, dtype=torch.float32)

    def __call__(self, features):
        text_list = [f["text"] for f in features]
        audio_list = [f["audio"] for f in features]
        
        scripts = [f"Speaker 0: {t}" for t in text_list]
        
        batch = self.processor(
            text=scripts,
            voice_samples=None,
            return_tensors="pt",
            padding=True
        )
        
        # Encode audio to latents
        latents_list = []
        for audio in audio_list:
            if self.processor.db_normalize:
                audio = self.processor.audio_normalizer(audio)
            
            if self.use_encodec:
                latent = self._encode_audio_encodec(audio)
            else:
                latent = self._encode_audio_mel(audio)
            
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
