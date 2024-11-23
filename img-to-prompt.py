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

# Load dataset
dataset = load_dataset("OleehyO/latex-formulas", "cleaned_formulas")

# Load model and processor
model = VisionEncoderDecoderModel.from_pretrained("DGurgurov/im2latex")
processor = AutoProcessor.from_pretrained("DGurgurov/im2latex")

# Access tokenizer directly
tokenizer = processor.tokenizer

# Configure model parameters
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 512

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_function(examples: Dict[str, List[Any]]) -> Dict[str, torch.Tensor]:
    """
    Preprocess a batch of examples.
    
    Args:
        examples: Dictionary containing batched data
        
    Returns:
        Dictionary with processed pixel values and labels
    """
    try:
        # Load and process images
        images = []
        for img_path in examples["image"]:
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
                continue
        
        if not images:
            raise ValueError("No valid images found in batch")

        # Process images
        pixel_values = processor(
            images=images,
            return_tensors="pt",
            padding=True
        ).pixel_values

        # Process formulas
        tokenized = tokenizer(
            examples["formula"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        labels = tokenized.input_ids.clone()
        
        # Replace padding tokens with -100 to ignore in loss calculation
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }
    
    except Exception as e:
        print(f"Error in preprocessing batch: {str(e)}")
        return None

# Process datasets with error handling
def process_dataset(dataset_split):
    """Process a dataset split with error handling"""
    processed = dataset_split.map(
        preprocess_function,
        batched=True,
        batch_size=8,
        remove_columns=dataset_split.column_names,
        num_proc=4
    )
    return processed.filter(lambda x: x is not None)

# Process train and validation sets
processed_train = process_dataset(dataset["train"])
processed_validation = process_dataset(dataset["validation"])

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    num_train_epochs=3,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=500,
    learning_rate=5e-5,
    report_to="tensorboard",
    save_total_limit=2,
    # Added arguments for better training
    gradient_accumulation_steps=2,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
)

# Create trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_train,
    eval_dataset=processed_validation,
    data_collator=default_data_collator,
    tokenizer=tokenizer,
)

# Train model with error handling
try:
    trainer.train()
    trainer.save_model("./trained_model")
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {str(e)}")