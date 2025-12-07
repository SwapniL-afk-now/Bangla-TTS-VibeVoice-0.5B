import argparse
import copy
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, TrainerCallback
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_training import VibeVoiceForTraining
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.data.bangla_dataset import BanglaDataset, VibeVoiceDataCollator


# ============ EMA Callback for Diffusion Head ============
class EmaCallback(TrainerCallback):
    """Exponential Moving Average callback for smoothing diffusion head training."""
    
    def __init__(self, attr_path="model.prediction_head", decay=0.999, device="cpu"):
        self.attr_path = attr_path
        self.decay = float(decay)
        self.device = torch.device(device)
        self.shadow = None
        self._orig = None

    def _get_module(self, model):
        mod = model
        for name in self.attr_path.split('.'):
            mod = getattr(mod, name)
        return mod

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        try:
            head = self._get_module(model)
            self.shadow = {k: p.detach().to(self.device).clone()
                           for k, p in head.state_dict().items()}
            print(f"EMA initialized for {self.attr_path} with decay={self.decay}")
        except Exception as e:
            print(f"EMA init failed: {e}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.shadow is None:
            return
        try:
            head = self._get_module(model)
            with torch.no_grad():
                for k, v in head.state_dict().items():
                    self.shadow[k].mul_(self.decay).add_(v.detach().to(self.device), alpha=(1.0 - self.decay))
        except Exception:
            pass

    def _swap_in_ema(self, model):
        try:
            head = self._get_module(model)
            self._orig = copy.deepcopy(head.state_dict())
            head.load_state_dict(self.shadow, strict=False)
        except Exception:
            pass

    def _swap_back(self, model):
        if self._orig is None:
            return
        try:
            head = self._get_module(model)
            head.load_state_dict(self._orig, strict=False)
            self._orig = None
        except Exception:
            pass

    def on_save(self, args, state, control, model=None, **kwargs):
        self._swap_in_ema(model)

    def on_save_end(self, args, state, control, model=None, **kwargs):
        self._swap_back(model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._swap_in_ema(model)


# ============ Custom Trainer with Dual Loss ============
class VibeVoiceTrainer(Trainer):
    """Custom trainer that adds CE loss alongside diffusion loss."""
    
    def __init__(self, ce_loss_weight=0.04, diffusion_loss_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ce_loss_weight = ce_loss_weight
        self.diffusion_loss_weight = diffusion_loss_weight
        print(f"Dual Loss: CE weight={ce_loss_weight}, Diffusion weight={diffusion_loss_weight}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get diffusion loss from model forward
        outputs = model(**inputs)
        diffusion_loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        
        # If diffusion_loss is a tuple, extract the loss
        if isinstance(diffusion_loss, tuple):
            diffusion_loss = diffusion_loss[0]
        
        # Compute CE loss on text tokens (if input_ids available)
        ce_loss = torch.tensor(0.0, device=diffusion_loss.device)
        
        if 'input_ids' in inputs and hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            try:
                input_ids = inputs['input_ids']
                attention_mask = inputs.get('attention_mask', None)
                
                # Get embeddings and pass through LM
                with torch.no_grad():
                    embeddings = model.model.language_model.get_input_embeddings()(input_ids)
                
                lm_outputs = model.model.language_model(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                
                # Get logits from lm_head if available
                if hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'lm_head'):
                    logits = model.model.language_model.lm_head(lm_outputs.last_hidden_state)
                else:
                    logits = lm_outputs.logits if hasattr(lm_outputs, 'logits') else None
                
                if logits is not None:
                    # Shift for causal LM loss
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()
                    
                    ce_loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction='mean'
                    )
            except Exception as e:
                # CE loss computation failed, continue with diffusion only
                pass
        
        # Combine losses
        total_loss = self.diffusion_loss_weight * diffusion_loss + self.ce_loss_weight * ce_loss
        
        # Log individual losses
        if self.state.global_step % 10 == 0:
            self.log({
                "diffusion_loss": diffusion_loss.detach().item(),
                "ce_loss": ce_loss.detach().item() if isinstance(ce_loss, torch.Tensor) else ce_loss,
                "total_loss": total_loss.detach().item(),
            })
        
        if return_outputs:
            return total_loss, outputs
        return total_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--dataset_name", type=str, required=True, help="Hugging Face dataset name")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset config (e.g. 'bn')")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--output_dir", type=str, default="vibevoice-bangla-checkpoints")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--report_to", type=str, nargs="*", default=[], help="Reporting integration")
    parser.add_argument("--training_stage", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Training stage: 1=Latent Alignment, 2=Acoustic Adaptation, 3=Text-Speech, 4=End-to-End")
    parser.add_argument("--speaker_ids", type=int, nargs="*", default=None,
                        help="Filter dataset by speaker IDs")
    # New arguments from reference implementation
    parser.add_argument("--ce_loss_weight", type=float, default=0.04, help="CE loss weight (default: 0.04)")
    parser.add_argument("--diffusion_loss_weight", type=float, default=1.0, help="Diffusion loss weight (default: 1.0)")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio for scheduler")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=1, help="Keep only N most recent checkpoints")
    parser.add_argument("--gradient_clipping", action="store_true", help="Enable gradient clipping")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--use_ema", action="store_true", default=True, help="Use EMA for diffusion head")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Training Stage: {args.training_stage}")
    print(f"Loss Weights: CE={args.ce_loss_weight}, Diffusion={args.diffusion_loss_weight}")
    print(f"Save: steps={args.save_steps}, total_limit={args.save_total_limit}")
    print("=" * 60)

    # Load Config and Model
    print(f"Loading model from {args.model_path}...")
    try:
        config = VibeVoiceConfig.from_pretrained(args.model_path)
    except:
        print("Could not load config automatically.")
        raise
        
    model = VibeVoiceForTraining.from_pretrained(args.model_path, config=config, ignore_mismatched_sizes=True)
    
    # Stage-specific LoRA configuration
    if args.training_stage == 1:
        print("Stage 1: Latent Space Alignment - training projection + acoustic connector + diffusion head")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.model.prediction_head.parameters():
            param.requires_grad = True
        for param in model.model.acoustic_connector.parameters():
            param.requires_grad = True
            
    elif args.training_stage >= 2 and args.use_lora:
        print(f"Stage {args.training_stage}: Applying LoRA...")
        from peft import LoraConfig, get_peft_model
        
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_r = args.lora_rank if args.training_stage == 2 else 16
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=None,
        )
        
        model = get_peft_model(model, lora_config)
        
        for name, param in model.base_model.model.model.prediction_head.named_parameters():
            param.requires_grad = True
        for name, param in model.base_model.model.model.acoustic_connector.named_parameters():
            param.requires_grad = True
             
        print(f"LoRA applied (r={lora_r}). Trainable parameters:")
        model.print_trainable_parameters()
    else:
        print("Training all parameters (no LoRA)")
        
    # Load Processor
    processor = VibeVoiceProcessor.from_pretrained(args.model_path)
    
    # Load Dataset
    print(f"Loading dataset {args.dataset_name}...")
    dataset = BanglaDataset(
        dataset_name=args.dataset_name,
        config_name=args.dataset_config,
        split=args.dataset_split,
        processor=processor,
        speaker_ids=args.speaker_ids
    )
    
    # Data Collator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    collator = VibeVoiceDataCollator(processor, model)
    
    # Stage 1: Also make EnCodec projection layer trainable
    if args.training_stage == 1 and hasattr(collator, 'encodec_projection'):
        print("Making EnCodec projection layer trainable...")
        for param in collator.encodec_projection.parameters():
            param.requires_grad = True

    # Training Args with improvements from reference
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,  # Keep only most recent checkpoints
        report_to=args.report_to,
        remove_unused_columns=False,
        label_names=["speech_values"],
        # Scheduler improvements
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        # Gradient clipping
        max_grad_norm=args.max_grad_norm if args.gradient_clipping else 0.0,
    )
    
    # Setup callbacks
    callbacks = []
    if args.use_ema:
        ema_path = "model.prediction_head" if not args.use_lora else "base_model.model.model.prediction_head"
        callbacks.append(EmaCallback(attr_path=ema_path, decay=args.ema_decay, device="cpu"))
        print(f"EMA callback enabled with decay={args.ema_decay}")

    # Use custom trainer with dual loss
    trainer = VibeVoiceTrainer(
        ce_loss_weight=args.ce_loss_weight,
        diffusion_loss_weight=args.diffusion_loss_weight,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
