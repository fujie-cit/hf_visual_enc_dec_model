# %% [markdown]
# STAIR Captions を利用するためのデータセットを，```./stair_captions_dataset/```に作成する．

# %%
from transformers import ViTImageProcessor, AutoTokenizer
from PIL import Image
import datasets
import os
import json

# %%
# STAIR Captions のパスなど
STAIR_CAPTIONS_DIR = "/autofs/diamond2/share/corpus/STAIR-captions"
STAIR_CAPTIONS_TRAIN_JSON_PATH = os.path.join(STAIR_CAPTIONS_DIR, "stair_captions_v1.2_train.json")
STAIR_CAPTIONS_VAL_JSON_PATH = os.path.join(STAIR_CAPTIONS_DIR, "stair_captions_v1.2_val.json")

# STAIR Captions が使う画像データがあるディレクトリのパスなど
COCO_DIR = "/autofs/diamond2/share/corpus/MS-COCO"
COCO_TRAIN2014_DIR = os.path.join(COCO_DIR, "train2014")
COCO_VAL2014_DIR = os.path.join(COCO_DIR, "val2014")

# %%
# START Captions の読み込み
train_json = json.load(open(STAIR_CAPTIONS_TRAIN_JSON_PATH))
val_json = json.load(open(STAIR_CAPTIONS_VAL_JSON_PATH))

# %%
# STAIR Captions のデータから，データセットに必要な情報に変換する関数
def convert_stair_caption_json_to_datalist(json, coco_image_dir):
    image_id2image_info = {image_info["id"]: image_info for image_info in json["images"]}

    datalist = []    
    for data in json["annotations"]:
        image_id = data["image_id"]
        image_info = image_id2image_info[image_id]
        image_path = os.path.join(coco_image_dir, image_info["file_name"])

        datalist.append({
            'id': data["id"],
            'caption': data["caption"],
            'image_path': image_path,
            'height': image_info["height"],
            'width': image_info["width"],
        })
    return datalist

# %%
# STAIR Captions のデータをデータセットに変換
datalist_train = convert_stair_caption_json_to_datalist(train_json, COCO_TRAIN2014_DIR)
datalist_val = convert_stair_caption_json_to_datalist(val_json, COCO_VAL2014_DIR)

# %%
# データセット形式に変換して保存する
dataset_dict = datasets.DatasetDict()
dataset_dict["train"] = datasets.Dataset.from_list(datalist_train)
dataset_dict["val"] = datasets.Dataset.from_list(datalist_val)

dataset_dict.save_to_disk("./stair_captions_dataset")


