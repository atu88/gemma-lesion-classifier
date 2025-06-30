import pandas as pd
import json

# Load metadata
df = pd.read_csv("train_metadata.csv") # put your actual filename here

# Output file
with open("train_metadata.jsonl", "w") as out_file:
    for _, row in df.iterrows():
        json_obj = {
            "image": f"{row['image']}.jpg",
            "label": row["label"]
        }
        out_file.write(json.dumps(json_obj) + "\n")
        
print("Successfully converted")