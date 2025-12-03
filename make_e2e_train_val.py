# make_e2e_train_val.py  ← run this instead of the previous one
import csv
import os
from sklearn.model_selection import train_test_split

# Load the original trainset.csv you downloaded
csv_path = os.path.expanduser("~/Downloads/e2e-dataset/trainset.csv")

sentences = []
with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sentences.append(row['ref'].strip())

# 90% train, 10% val (stratified-ish, but E2E is balanced enough)
train_sents, val_sents = train_test_split(
    sentences, test_size=0.1, random_state=42, shuffle=True
)

os.makedirs("data", exist_ok=True)

with open("data/e2e_train.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(train_sents))

with open("data/e2e_val.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(val_sents))

print(f"Done!")
print(f"  → data/e2e_train.txt : {len(train_sents):,} sentences (train)")
print(f"  → data/e2e_val.txt   : {len(val_sents):,} sentences (val)")
print("First validation example:")
print("  →", val_sents[0])