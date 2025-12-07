import argparse
import logging
import os
import torch
from transformers import Trainer, TrainingArguments
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_training import VibeVoiceForTraining
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.data.bangla_dataset import BanglaDataset, VibeVoiceDataCollator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--dataset_name", type=str, required=True, help="Hugging Face dataset name (e.g. 'mozilla-foundation/common_voice_17_0')")
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
    parser.add_argument("--report_to", type=str, nargs="*", default=[], help="Reporting integration (wandb/tensorboard). Leave empty to disable.")
    parser.add_argument("--training_stage", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Training stage: 1=Latent Alignment, 2=Acoustic Adaptation, 3=Text-Speech Alignment, 4=End-to-End")
    parser.add_argument("--speaker_ids", type=int, nargs="*", default=None,
                        help="Filter dataset by speaker IDs (e.g., --speaker_ids 0 1)")
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"Training Stage: {args.training_stage}")
    print(f"=" * 60)

    # Load Config and Model
    print(f"Loading model from {args.model_path}...")
    try:
        config = VibeVoiceConfig.from_pretrained(args.model_path)
    except:
        print("Could not load config automatically. Using default config structure.")
        raise
        
    model = VibeVoiceForTraining.from_pretrained(args.model_path, config=config, ignore_mismatched_sizes=True)
    
    # Stage-specific LoRA configuration
    if args.training_stage == 1:
        # Stage 1: No LoRA, train projection + diffusion head only
        print("Stage 1: Latent Space Alignment - No LoRA, training projection + acoustic connector + diffusion head")
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze: prediction_head, acoustic_connector
        for param in model.model.prediction_head.parameters():
            param.requires_grad = True
        for param in model.model.acoustic_connector.parameters():
            param.requires_grad = True
            
    elif args.training_stage >= 2 and args.use_lora:
        print(f"Stage {args.training_stage}: Applying LoRA...")
        from peft import LoraConfig, get_peft_model
        
        # Stage 2: LoRA on TTS LM only
        # Stage 3+: LoRA on both LMs
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
        
        # Unfreeze diffusion head
        for name, param in model.base_model.model.model.prediction_head.named_parameters():
            param.requires_grad = True
        # Unfreeze acoustic connector
        for name, param in model.base_model.model.model.acoustic_connector.named_parameters():
            param.requires_grad = True
             
        print(f"LoRA applied (r={lora_r}). Trainable parameters:")
        model.print_trainable_parameters()
    else:
        # Stage 2+ without LoRA - train everything
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
    
    # Data Collator (needs model for on-the-fly encoding)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    collator = VibeVoiceDataCollator(processor, model)
    
    # Stage 1: Also make EnCodec projection layer trainable
    if args.training_stage == 1 and hasattr(collator, 'encodec_projection'):
        print("Making EnCodec projection layer trainable...")
        for param in collator.encodec_projection.parameters():
            param.requires_grad = True

    # Training Args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=500,
        report_to=args.report_to,
        remove_unused_columns=False, # Critical for custom collator keys
        label_names=["speech_values"] # Prevent trainer from stripping 'labels' if we had them?
        # We don't have 'labels' key, we have 'speech_values'.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
