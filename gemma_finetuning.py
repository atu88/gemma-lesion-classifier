# Imports
from huggingface_hub import login
from datasets import load_dataset
from PIL import Image
import os
import shutil
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from google.colab import drive
import gc

# Login
hf_token = 'HF access token goes here'
login(hf_token)

# Files Preparation

# Mount Google Drive on Colab
drive.mount('/content/drive')

# Google Drive metadata file path
gdrive_metadata_file = "/content/drive/My Drive/Colab Notebooks/gemma/train_metadata.jsonl"

# Local directory in Colab to copy metadata file to
local_metadata_file = "/content/gemma/train_metadata.jsonl"

# Ensure local folder exists
os.makedirs(os.path.dirname(local_metadata_file), exist_ok=True)

# Copy metadata file from Google Drive to local folder (if not already copied)
if not os.path.exists(local_metadata_file):
    print("Copying metadata file from Google Drive to local disk...")
    shutil.copy(gdrive_metadata_file, local_metadata_file)
else:
    print("Local metadata file already exists, skipping copy.")

# Google Drive zip file path
gdrive_zip_file = "/content/drive/My Drive/Colab Notebooks/gemma/train.zip"

# Local path in Colab to store the copied zip
local_zip_file = "/content/train.zip"

# Final image folder path after unzipping
image_folder_path = "/content/gemma/train"

# Copy the zip file
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

# Model setup

# Messages setup
system_message = "You are an expert in skin lesion diagnosis."

user_prompt = "Classify the lesions based on the provided <LABEL> and image."

# Format data for training
def format_data(sample):
    # Open the image file as PIL.Image and convert to RGB
    img_id = sample["image"]
    img_path = os.path.join(image_folder_path, img_id)
    img = Image.open(img_path).convert("RGB")
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                    {
                        "type": "image",
                        "image": img,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["label"]}],
            },
        ],
    }

# Vision Info setup
def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                image = element["image"] if "image" in element else element
                image_inputs.append(image.convert("RGB"))
    return image_inputs

# Load dataset and apply formatting that loads images
dataset = load_dataset("json", data_files=local_metadata_file, split="train")
dataset = [format_data(sample) for sample in dataset]

print(dataset[10]["messages"])

# Model config
model_id = "google/gemma-3-4b-pt"

model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.float16,
    device_map="auto",
)

model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_storage=torch.float16,
    bnb_4bit_use_nested_quant=True,
)

# Load model and processor
model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

# PEFT config
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"],
)

# Training config
args = SFTConfig(
    output_dir="/content/drive/My Drive/Colab Notebooks/gemma/gemma-lesion-classifier", # Local repository to save the model goes here
    hub_model_id="hf_id/gemma-lesion-classifier", # HF repository to save the model goes here
    num_train_epochs=1,
    per_device_train_batch_size=2, # Increase or decrease this according to GPU capacity
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=5,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    push_to_hub=True,
    report_to="tensorboard",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
)

args.remove_unused_columns = False

# Data collator
def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image_inputs = process_vision_info(example["messages"])
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        images.append(image_inputs)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()

    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch

# Clean memory
gc.collect()
torch.cuda.empty_cache()

# Trainer
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)

# Train and save
trainer.train()
trainer.save_model()
