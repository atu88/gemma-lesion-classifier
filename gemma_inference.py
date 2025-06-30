# Imports
import os
import csv
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
import re
from google.colab import drive
from huggingface_hub import login
import shutil

# Mount Google Drive
drive.mount('/content/drive')

# Login
hf_token = 'HF access token goes here'
login(hf_token)

# output folder to save prediction results
output_csv = "/content/drive/My Drive/Colab Notebooks/gemma/prediction_results.csv"  # sample path

# Google Drive zip file path
gdrive_zip_file = "/content/drive/My Drive/Colab Notebooks/gemma/test.zip"

# Local path in Colab to store the copied zip
local_zip_file = "/content/test.zip"

# Final image folder path after unzipping
image_folder_path = "/content/gemma/test"

# Copy the zip file from Gdrive
if not os.path.exists(local_zip_file):
    print("Copying zip file from Google Drive to local disk...")
    shutil.copy(gdrive_zip_file, local_zip_file)
else:
    print("Zip file already exists locally, skipping copy.")

# Unzip it
if not os.path.exists(image_folder_path):
    print("Unzipping...")
    os.makedirs(image_folder_path, exist_ok=True)
    shutil.unpack_archive(local_zip_file, "/content/gemma")
else:
    print("Image folder already extracted, skipping unzip.")

# Model Setup

# Messages setup
system_message = "You are an expert in skin lesion diagnosis."

user_prompt = "Classify the lesions based on the provided <LABEL> and image."

# Load Model with PEFT adapter
model = AutoModelForImageTextToText.from_pretrained(
    "hf_id/gemma-lesion-classifier",  # the same repo as the one you saved the model to
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)

processor = AutoProcessor.from_pretrained("hf_id/gemma-lesion-classifier")  # the same repo as the one you saved the model to

# Load the images for processing
def load_images_from_folder(folder_path):
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_paths.append(os.path.join(folder_path, filename))
    return image_paths

# Vision Info setup
def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]
        for element in content:
            if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                image = element["image"] if "image" in element else element
                image_inputs.append(image.convert("RGB"))
    return image_inputs

# Define generate predictions
def generate_prediction(image: Image.Image, model, processor, system_message: str):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image", "image": image},
        ]},
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    stop_token_ids = [
        processor.tokenizer.eos_token_id,
        processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ]

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        top_p=1.0,
        do_sample=False,
        eos_token_id=stop_token_ids,
        disable_compile=True,
    )
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    output = output_text[0]
    return output.strip()

# Load image paths
image_paths = load_images_from_folder(image_folder_path)

# Prepare list of predictions
results = []

for path in image_paths:
    filename = os.path.basename(path)
    print(f"Processing image: {filename}")
    image = Image.open(path).convert("RGB")
    prediction = generate_prediction(image, model, processor, system_message)
    print(f"Prediction: {prediction}\n")
    results.append({"filename": filename, "prediction": prediction})

# Save results to CSV
with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "prediction"])
    writer.writeheader()
    writer.writerows(results)

print(f"Predictions saved to: {output_csv}")