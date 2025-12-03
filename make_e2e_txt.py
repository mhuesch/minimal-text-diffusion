# make_e2e_txt.py — works on the original E2E CSVs you downloaded
import csv
import os

# Path to where you downloaded the files
# from https://github.com/tuetschek/e2e-dataset
csv_path = os.path.expanduser("~/Downloads/e2e-dataset/trainset.csv")

# Read the CSV and extract the 'ref' column (the human-written text)
sentences = []
with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sentences.append(row['ref'].strip())

# Save as plain txt
os.makedirs("data", exist_ok=True)
txt_path = "data/e2e_train.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(sentences))

print(f"Done! {txt_path} created with {len(sentences):,} sentences")
print("First 5 examples:")
for s in sentences[:5]:
    print(" →", s)