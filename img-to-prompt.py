import torch
from transformers import VisionEncoderDecoderModel, AutoProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Charger le dataset
dataset = load_dataset("OleehyO/latex-formulas", "cleaned_formulas")

# Charger un modèle pré-entraîné Vision-Encoder-Decoder
model_name = "google/vit-base-patch16-224-in21k"  # Exemple, peut être remplacé par un autre
processor = AutoProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    "google/vit-base-patch16-224-in21k",  # Modèle de l'encodeur visuel
    "gpt2"                               # Modèle du décodeur textuel
)

# Modifier la taille maximale des séquences pour le décodeur
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.max_length = 512
model.config.eos_token_id = processor.tokenizer.eos_token_id

# Prétraitement des images
image_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Fonction de transformation pour le dataset
def preprocess_data(batch):
    # Charger et transformer les images
    image = Image.open(batch["image_path"]).convert("RGB")
    batch["pixel_values"] = image_transform(image)
    # Tokenizer les formules LaTeX
    batch["labels"] = processor.tokenizer(
        batch["formula"],
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids.squeeze()  # Retirer la dimension batch
    return batch

# Appliquer la transformation
processed_dataset = dataset.map(preprocess_data, remove_columns=["image_path", "formula"])

# Configuration pour l'entraînement
training_args = Seq2SeqTrainingArguments(
    output_dir="./img2latex",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    logging_dir="./logs",
    learning_rate=5e-5,
    num_train_epochs=10,
    save_total_limit=2
)

# Création du trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    tokenizer=processor.feature_extractor,
)

# Lancer l'entraînement
trainer.train()

# Sauvegarder le modèle final
model.save_pretrained("./img2latex_model")
processor.save_pretrained("./img2latex_processor")
