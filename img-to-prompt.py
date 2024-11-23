from datasets import load_dataset
from transformers import VisionEncoderDecoderModel, AutoProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import default_data_collator
from PIL import Image
import torch
import os

# Charger le dataset
dataset = load_dataset("OleehyO/latex-formulas", "cleaned_formulas")

# Charger le modèle et le processeur
model = VisionEncoderDecoderModel.from_pretrained("DGurgurov/im2latex")
processor = AutoProcessor.from_pretrained("DGurgurov/im2latex")

# Configurer le modèle
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id  # Utilise le token de début de séquence
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.config.max_length = 512

# Prétraitement des données
def preprocess_function(batch):
    # Charger et traiter l'image
    images = [Image.open(img_path).convert("RGB") for img_path in batch["image"]]
    pixel_values = processor(images=images, return_tensors="pt").pixel_values

    # Tokenizer les formules LaTeX
    labels = processor.tokenizer(batch["formula"], padding="max_length", truncation=True, max_length=512).input_ids
    labels = torch.tensor(labels)
    
    # Remplacer les tokens de padding par -100 pour éviter qu'ils contribuent à la loss
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    return {"pixel_values": pixel_values, "labels": labels}

# Appliquer le prétraitement
processed_dataset = dataset.map(preprocess_function, batched=True)

# Arguments pour l’entraînement
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
)

# Création du Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    data_collator=default_data_collator,
    tokenizer=processor.tokenizer,
)

# Entraîner le modèle
trainer.train()

# Sauvegarder le modèle
trainer.save_model("./trained_model")
