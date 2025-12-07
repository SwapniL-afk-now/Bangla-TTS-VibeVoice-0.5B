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
    parser.add_argument("--report_to", type=str, default="none", help="Reporting integration (wandb/tensorboard/none)")
    args = parser.parse_args()

    # Load Config and Model
    print(f"Loading model from {args.model_path}...")
    try:
        config = VibeVoiceConfig.from_pretrained(args.model_path)
    except:
        print("Could not load config automatically. Using default config structure.")
        raise
        
    model = VibeVoiceForTraining.from_pretrained(args.model_path, config=config, ignore_mismatched_sizes=True)
    
    if args.use_lora:
        print("Applying LoRA...")
        from peft import LoraConfig, get_peft_model, TaskType
        
        # Define LoRA Config
        # Target modules: Qwen typically uses "q_proj", "v_proj" etc.
        # We target the language models.
        # `model.language_model` (base) and `model.tts_language_model` (tts).
        # We can target all linear layers in them.
        
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=None, # Custom model
        )
        
        # Get PEFT model
        # Note: VibeVoiceForTraining is a wrapper. We can apply PEFT to the submodules or the whole wrapper.
        # Applying to whole wrapper is easiest, targeting specific module names.
        model = get_peft_model(model, lora_config)
        
        # Ensure Diffusion Head is TRAINABLE (unless we want to freeze it too?)
        # For TTS fine-tuning, usually the diffusion head needs adaptation.
        # LoRA freezes all non-adapter parameters.
        # We should unfreeze the diffusion head if we want to train it fully.
        
        # model.model.prediction_head.requires_grad_(True) 
        # But `get_peft_model` freezes everything.
        
        for name, param in model.base_model.model.model.prediction_head.named_parameters():
             param.requires_grad = True
             
        print("LoRA applied. Trainable parameters:")
        model.print_trainable_parameters()
        
    # Load Processor
    processor = VibeVoiceProcessor.from_pretrained(args.model_path)
    
    # Load Dataset
    print(f"Loading dataset {args.dataset_name}...")
    dataset = BanglaDataset(
        dataset_name=args.dataset_name,
        config_name=args.dataset_config,
        split=args.dataset_split,
        processor=processor
    )
    
    # Data Collator (needs model for on-the-fly VAE encoding)
    # Note: On-the-fly VAE inside main process might be slow. 
    # Move model to GPU?
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    collator = VibeVoiceDataCollator(processor, model)

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
