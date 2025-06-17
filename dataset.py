import os

from datasets import load_dataset
from transformers import AutoImageProcessor

DATA_DIR = "data"
TEST_SIZE=0.2
SEED=42

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True)

# === preprocess data === #
def preprocess(examples):
    """
    Preprocess images for training
    """

    images = examples['image']
    inputs = processor(images, return_tensors="pt")
    
    # Rename 'label' to 'labels' as expected by the model
    inputs['labels'] = examples['label']
    
    return inputs

if not os.path.exists(DATA_DIR):
    print("Download and preprocess the dataset for ResNet...")
    ds = load_dataset("nateraw/pizza_not_pizza")
    ds = ds["train"].train_test_split(test_size=TEST_SIZE, seed=SEED)
    ds = ds.map(preprocess, batched=True, remove_columns=["image", "label"])
    ds.save_to_disk(DATA_DIR)
    print("Preprocessing finished!")
    print(f"The dataset is saved in {DATA_DIR} directory")
else:
    print(f"Data already downloaded in {DATA_DIR} dir!")

