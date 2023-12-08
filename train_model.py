# %%
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
from PIL import Image
import datasets
import os

# %%
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium", use_fast=False)
tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading

# %%
def convert_to_features(example_batch):
    inputs = tokenizer(
        example_batch["caption"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    images = []
    for image_path in example_batch["image_path"]:
        assert os.path.exists(image_path), f"image_path={image_path} is not found."
        images.append(Image.open(image_path).convert("RGB"))
    pixel_values = image_processor(images=images, return_tensors="pt").pixel_values
    return {"pixel_values": pixel_values, "labels": inputs.input_ids}

# %%
dataset_dict = datasets.DatasetDict.load_from_disk("./stair_captions_dataset")


# %%
len(dataset_dict["val"])

# %%
dataset_dict['val'] = datasets.Dataset.train_test_split(dataset_dict['val'], test_size=0.01, seed=42)["test"]

# %%
dataset_dict.set_transform(convert_to_features)

# %%
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    "google/vit-base-patch16-224-in21k", 
    "rinna/japanese-gpt2-medium",
)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id


# %%
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./models/vit-gpt2-japanese-image-captioning_stair-captions",
    num_train_epochs=5,
    per_device_train_batch_size=10, 
    per_device_eval_batch_size=1,   
    warmup_steps=500,               
    weight_decay=0.01,              
    logging_dir='./logs',
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,                       
    args=training_args,                  
    train_dataset=dataset_dict['train'],        
    eval_dataset=dataset_dict['val']   
)


# %%
trainer.train()


