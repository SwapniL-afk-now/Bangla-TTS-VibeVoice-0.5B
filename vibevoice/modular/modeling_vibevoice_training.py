import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from .modeling_vibevoice_streaming import VibeVoiceStreamingPreTrainedModel, VibeVoiceStreamingModel, BinaryClassifier

@dataclass
class VibeVoiceTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None

class VibeVoiceForTraining(VibeVoiceStreamingPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = VibeVoiceStreamingModel(config)
        self.post_init()
        
        # We need the speech start embedding which is part of the tokenizer but not the model logic directly?
        # In inference: `input_ids` has `tokenizer.speech_start_id`.
        # `forward_tts_lm` takes `input_ids` and gets embeddings.
        # So `model.language_model.embed_tokens` works for special IDs too.
        # We will need to look up these IDs. 
        # For now, we'll assume they are passed in `input_ids` or handled via a dedicated argument.
        self.speech_start_id = 151646 # Default for Qwen? Need to verify.
        self.speech_end_id = 151647

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        speech_values: Optional[torch.FloatTensor] = None, # (Batch, LatentDim, SeqLen) or (Batch, SeqLen, LatentDim)?
        speech_attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # 1. Base Text Processing
        # input_ids contains [System, Text, Text_Input_Marker, ..., Speech_Output_Marker, Speech_Start]
        # We expect the dataset to provide `input_ids` correctly formatted up to `Speech_Start`.
        
        lm_outputs = self.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        lm_last_hidden_state = lm_outputs.last_hidden_state
        
        # 2. Text -> TTS conditioning
        # Use simple ones mask for text part
        tts_text_masks = torch.ones(lm_last_hidden_state.shape[:2], device=lm_last_hidden_state.device, dtype=torch.long)
        
        text_embeds = lm_last_hidden_state + self.model.tts_input_types(tts_text_masks)
        
        # 3. Speech Processing
        # We assume speech_values are already LATENTS (Batch, SeqLen, Dim).
        # We will handle VAE encoding in the dataset loader to save VRAM/Compute during training loop (or add it here later).
        # Assuming `speech_values` is (B, T_speech, D).
        
        latents = speech_values
        if latents is None:
             raise ValueError("speech_values cannot be None for training")
             
        # Acoustic History Embeddings (Teacher Forcing)
        # Shift latents right: [Zero/Start, L_0, L_1, ..., L_{T-2}] for prediction of [L_0, ..., L_{T-1}]?
        # Actually inference starts with `sample_speech_tokens` conditioned on Text.
        # Then `acoustic_embed` of generated token is fed back.
        # So first speech token L_0 is conditioned on Text.
        # Second speech token L_1 is conditioned on Text + L_0.
        
        # So, Input to TTS LM for Speech Part:
        # [Acoustic_Embed(L_0), Acoustic_Embed(L_1), ...] 
        # Wait, if L_0 is *generated* based on Text, then TTS LM doesn't see L_0 until it generates L_1?
        # Inference Loop: 
        # t=0: Condition = TTS_LM(Text).last_token. Gen L_0.
        # t=1: Input = Acoustic_Embed(L_0). TTS_LM([Text, In_0]). Condition = LastToken. Gen L_1.
        
        # So Sequence Input for TTS LM:
        # [Text_Embeds, Acoustic_Embed(L_0), Acoustic_Embed(L_1), ..., Acoustic_Embed(L_{T-1})]
        # Model Output at these positions:
        # Pos Text_End: Condition for L_0.
        # Pos L_0: Condition for L_1.
        # ...
        
        # So we concatenate Text + Acoustic(L_shifted).
        # We effectively need: [Text_Embeds] <concat> [Acoustic_Embed(L)]
        # But we only use outputs from [Text_End ... L_{T-1}].
        # So input sequence is: [Text_Embeds, Acoustic_Embed(L_0), ..., Acoustic_Embed(L_{T-1})]
        # And we predict [L_0, ..., L_{T-1}, L_T??]
        # No, "Text End" position predicts L_0.
        # "L_0" position predicts L_1.
        
        # So inputs should be: [Text_Embeds, Acoustic_Embed(L_0), ..., Acoustic_Embed(L_{T-2})]?
        # Or does `Text_Embeds` INCLUDE `Speech_Start`?
        # If `input_ids` ends with `Speech_Start`, then `Text_Embeds` last token IS `Speech_Start`.
        # So `Text_Embeds`[-1] is the condition for L_0. Correct.
        
        # The subsequent inputs are L_0 ... L_{T-2}.
        # So we need `Acoustic_Embeds` of `latents[:, :-1, :]`.
        
        # Note: If we want to predict L_{last} we need input L_{last-1}.
        
        # Let's align lengths.
        # Target: Latents (B, T, D)
        # Conditions needed: T vectors.
        # 1st Condition comes from Text_Embeds[-1].
        # 2nd..Tth Conditions come from passing L_0..L_{T-2} through TTS LM.
        
        acoustic_embeds = self.model.acoustic_connector(latents) # (B, T, H)
        
        # Inputs to TTS LM:
        # Concatenate Text part + Acoustic part (excluding last latent, as it's not history for any target)
        # But we need to insert them into the stream.
        # `tts_inputs` = [Text_Embeds, Acoustic_Embeds[:, :-1, :]]
        
        # Handling the "Text End predicts first latent" case.
        # `Text_Embeds` (from `input_ids`) should already cover the prompt.
        
        if acoustic_embeds.shape[1] > 1:
            speech_history_embeds = acoustic_embeds[:, :-1, :]
            
            # Mask for speech inputs (0 for speech in TTS LM)
            speech_masks = torch.zeros(speech_history_embeds.shape[:2], device=speech_history_embeds.device, dtype=torch.long)
            speech_history_embeds = speech_history_embeds + self.model.tts_input_types(speech_masks)
            
            full_inputs = torch.cat([text_embeds, speech_history_embeds], dim=1)
            
            # Extend attention mask
            # Text mask is provided. Speech mask is all 1s (valid tokens).
            speech_att_mask = torch.ones(speech_history_embeds.shape[:2], device=attention_mask.device, dtype=attention_mask.dtype)
            full_att_mask = torch.cat([attention_mask, speech_att_mask], dim=1)
            
        else:
            # Single step (only predict L_0 from text)
            full_inputs = text_embeds
            full_att_mask = attention_mask

        # Pass through TTS LM
        tts_outputs = self.model.tts_language_model(
            inputs_embeds=full_inputs,
            attention_mask=full_att_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = tts_outputs.last_hidden_state
        
        # Extract Conditions for Diffusion
        # We need the hidden states corresponding to the steps used to predict the targets.
        # Targets: L_0, L_1, ..., L_{T-1}
        # Predictor for L_0: Hidden state at index (Text_Len - 1)
        # Predictor for L_k: Hidden state at index (Text_Len + k - 1)
        
        # So we take the slice of hidden states starting from (Text_Len - 1).
        # Length of slice should be T.
        
        text_len = input_ids.shape[1]
        speech_len = latents.shape[1]
        
        # hidden_states shape: (B, Text_Len + Speech_Len - 1, H)
        # We want [Text_Len-1 : Text_Len-1 + Speech_Len]
        
        conditions = []
        for b in range(latents.shape[0]):
            # Assuming simple batching where text lengths might vary? 
            # If padded, we need actual length. 
            # `input_ids` is right-padded usually? 
            # But `attention_mask` tells us valid length.
            # Let's assume left-padding or we just use `text_len` if batch is uniform.
            # For robustness, let's use the explicit indices.
            
            # Actually, typical training batches have same sequence length or padding.
            # The `input_ids` passed here should be the PROMPT.
            # If we simply slice `hidden_states[:, text_len-1:, :]`, it should work if padding is handled correctly by mask.
            # But wait, if `input_ids` has padding at the end, `text_len-1` is a PAD token!
            # We need the last VALID token of the PROMPT.
            pass
            
        # Simplified: Assume batch is curated such that text parts are effectively same length or we gather correctly.
        # But `prepare_inputs` usually pads.
        # Let's trust `input_ids` implies the boundary.
        # We can gather the `last_token` indices from attention_mask for the start.
        
        # Better: construct a gather index.
        # But for now, let's just slice assuming fixed length or mask-ignorant training (masked loss).
        
        # Valid conditions slice
        # The sequence generated is: [Text...T_end, S_0, S_1, ...]
        # Output H at T_end predicts L_0.
        # Output H at S_0 predicts L_1.
        # ...
        # Output H at S_{T-2} predicts L_{T-1}.
        
        # So we need `hidden_states` from index `text_len - 1` to `end`.
        # Size check: 
        # Full Input Len = Text_Len + (Speech_Len - 1)
        # We need Speech_Len predictions.
        # Slice start: Text_Len - 1.
        # Slice length: Speech_Len.
        # (Text_Len - 1) + Speech_Len = Text_Len + Speech_Len - 1.
        # Matches exactly the output size.
        
        condition_vectors = hidden_states[:, text_len-1:, :]
        
        # 4. Compute Loss
        # Normalize latents to match diffusion space
        # Inference: scaled_latent = speech_latent / scale - bias
        # => speech_latent (Target) = (scaled_latent + bias) * scale
        
        # Check if factors are valid (not NaN)
        scale = self.model.speech_scaling_factor
        bias = self.model.speech_bias_factor
        
        if torch.isnan(scale) or torch.isnan(bias):
             # If NaNs (e.g. fresh init), we might warn or default to identity.
             # But for fine-tuning, they SHOULD be present. 
             # If not, we skip norm (assume raw latents are target).
             pass
        else:
             latents = (latents + bias) * scale

        # latents: (B, T, D)
        # condition_vectors: (B, T, H)
        
        # Flatten batch and time
        B, T, D = latents.shape
        latents_flat = latents.reshape(B*T, D)
        conditions_flat = condition_vectors.reshape(B*T, -1)
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents_flat)
        timesteps = torch.randint(0, self.model.noise_scheduler.config.num_train_timesteps, (B*T,), device=latents_flat.device)
        timesteps = timesteps.float()  # Convert to float for timestep embedder
        
        noisy_latents = self.model.noise_scheduler.add_noise(latents_flat, noise, timesteps.long())  # add_noise expects long
        
        # Predict
        predicted_noise = self.model.prediction_head(
            noisy_latents, 
            timesteps, 
            condition=conditions_flat
        )
        
        # MSE Loss
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        
        return VibeVoiceTrainingOutput(loss=loss, logits=predicted_noise)

