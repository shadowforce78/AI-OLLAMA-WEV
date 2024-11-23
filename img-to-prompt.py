import sys
import subprocess
import pkg_resources

def install_requirements():
    """Install required packages if they're missing."""
    required = {
        'transformers[torch]': None,
        'accelerate': '>=0.26.0',
        'datasets': None,
        'Pillow': None,
        'torch': None
    }
    
    def pip_install(package, version=None):
        if version:
            package_spec = f"{package}{version}"
        else:
            package_spec = package
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
    
    installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    for package, version in required.items():
        try:
            pkg_resources.require(f"{package}{version if version else ''}")
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            print(f"Installing {package}{version if version else ''}...")
            pip_install(package, version)

install_requirements()

from datasets import load_dataset
from transformers import (
    VisionEncoderDecoderModel,
    AutoProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers import default_data_collator
from PIL import Image
import torch
import os
from typing import List, Dict, Any
from multiprocessing import freeze_support

def main():
    print("Loading dataset...")
    dataset = load_dataset("OleehyO/latex-formulas", "cleaned_formulas")
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    
    dataset = {
        "train": dataset["train"],
        "validation": dataset["test"]
    }

    print("Loading model and processor...")
    model = VisionEncoderDecoderModel.from_pretrained("DGurgurov/im2latex")
    image_processor = AutoProcessor.from_pretrained("microsoft/trocr-base-handwritten")  # Using TrOCR processor for images
    text_processor = AutoProcessor.from_pretrained("DGurgurov/im2latex")  # Using original processor for text

    # Configure model parameters
    model.config.decoder_start_token_id = text_processor.bos_token_id
    model.config.pad_token_id = text_processor.pad_token_id
    model.config.eos_token_id = text_processor.eos_token_id
    model.config.max_length = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def preprocess_function(examples: Dict[str, List[Any]]) -> Dict[str, torch.Tensor]:
        try:
            # Process images using image processor
            images = examples["image"]
            if not images:
                raise ValueError("No valid images found in batch")

            # Process images
            processed_images = image_processor(
                images=images,
                return_tensors="pt",
                padding=True
            )

            # Process formulas using regular tokenizer
            tokenized = text_processor.tokenizer(
                examples["formula"],  # Direct text input
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            labels = tokenized.input_ids.clone()
            labels[labels == text_processor.pad_token_id] = -100

            return {
                "pixel_values": processed_images.pixel_values,
                "labels": labels
            }
        except Exception as e:
            print(f"Error in preprocessing batch: {str(e)}")
            return None

    def process_dataset(dataset_split):
        print(f"Processing {len(dataset_split)} examples...")
        processed = dataset_split.map(
            preprocess_function,
            batched=True,
            batch_size=4,  # Reduced batch size
            remove_columns=dataset_split.column_names,
            num_proc=1  # Reduced to single process for debugging
        )
        return processed.filter(lambda x: x is not None)

    print("Processing training dataset...")
    processed_train = process_dataset(dataset["train"])
    print("Processing validation dataset...")
    processed_validation = process_dataset(dataset["validation"])

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=500,
        learning_rate=5e-5,
        report_to="tensorboard",
        save_total_limit=2,
        gradient_accumulation_steps=2,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_validation,
        data_collator=default_data_collator,
        tokenizer=text_processor,
    )

    try:
        trainer.train()
        trainer.save_model("./trained_model")
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    freeze_support()
    main()