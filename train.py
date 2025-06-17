import os
import torch
import numpy as np
import torchvision
import wandb
import tqdm
import evaluate
import json

from dotenv import load_dotenv
from datasets import load_from_disk
from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        TrainingArguments,
        Trainer,
        DefaultDataCollator
)

# === global variables === #
SEED=42
EPOCHS=5
LR=5e-5
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR="./data"
OUT_DIR="./out"
LOG_DIR="./logs"

torch.manual_seed(SEED)
np.random.seed(SEED)

load_dotenv()

# === wandb === #
wandb.login(key=os.getenv("WANDB_API_KEY"))

# === loading dataset === #
if not os.path.exists(DATA_DIR): 
    raise FileNotFoundError(f"Preprocessed data directory not found: {DATA_DIR}")

ds = load_from_disk(DATA_DIR)

# === model load === #
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True)
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")

# === evaluation === #
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# === training/fine-tuning === #
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    eval_strategy="epochs",
    learning_rate=LR,
    logging_dir=LOG_DIR,
    seed=SEED,
    fp16=torch.cuda.is_available(),
    report_to="wandb",
    num_train_epochs=EPOCHS,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    processing_class=processor,
    compute_metrics=compute_metrics,
    data_collator=DefaultDataCollator(),
)

# === training loop === #
print("Start training...")
try:
    trainer.train()

    # evaluate
    print("Evaluating the final model...")
    metrics = trainer.evaluate()
    for key, value in metrics.items():
        print(f" {key}: {value}")
    
    # save the model
    print(f"Saving final model to {OUT_DIR}...")
    trainer.save_model(OUT_DIR)
    processor.save_pretrained(OUT_DIR)

    print("Training completed successfully!")
    print(f"Final model saved to: {OUT_DIR}")

except Exception as e:
    print(f"Training failed with error: {e}")
    raise

finally:
    wandb.finish()
