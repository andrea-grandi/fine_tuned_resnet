import os
from datasets import load_dataset

DATA_DIR = "data"
TEST_SIZE=0.2
SEED=42

ds = load_dataset("nateraw/pizza_not_pizza")

if not os.path.exists(DATA_DIR):
    print(f"Save data to disk: {DATA_DIR}")
    ds = ds["train"].train_test_split(test_size=TEST_SIZE, seed=SEED)
    ds.save_to_disk(DATA_DIR)
    print("Download finished!")
else:
    print(f"Data already downloaded in {DATA_DIR} dir!")

