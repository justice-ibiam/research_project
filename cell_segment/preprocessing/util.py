import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
from preprocessing.ben_graham_preprocessing import ben_graham_preprocess, crop_fundus_circle, crop_fundus_circle_hough


def preprocess_idrid(cfg):
    """
    Applies crop_fundus_circle + ben_graham_preprocess to the IDRID dataset
    and stores processed images on disk.
    """

    input_train_dir = os.path.join(cfg.dataset.path, "images", "train")
    input_test_dir  = os.path.join(cfg.dataset.path, "images", "test")

    labels_train = os.path.join(cfg.dataset.path, "labels", "train.csv")
    labels_test  = os.path.join(cfg.dataset.path, "labels", "test.csv")

    output_train_dir = os.path.join(cfg.dataset.path, "processed", "train")
    output_test_dir  = os.path.join(cfg.dataset.path, "processed", "test")

    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)


    # Process TRAIN set
    print("Processing TRAIN images...")
    df_train = pd.read_csv(labels_train)

    for _, row in tqdm(df_train.iterrows(), total=len(df_train)):
        image_name = row["Image name"]
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
            image_name += ".jpg"

        input_path = os.path.join(input_train_dir, image_name)
        output_path = os.path.join(output_train_dir, image_name.replace(".tif", ".jpg"))

        image = Image.open(input_path).convert("RGB")

        # Step 1: crop circle
        image_small = image.resize((int(image.width*0.3), int(image.height*0.3)))
        image_small = np.array(image_small)
        image = crop_fundus_circle_hough(image_small)

        # Step 2: Graham pre-processing (returns numpy array)
        image = ben_graham_preprocess(image, sigma=45, scale=256)

        # Step 3: convert to PIL and save
        image = Image.fromarray(image)
        image.save(output_path, quality=95)

    # Process TEST set
    print("Processing TEST images...")
    df_test = pd.read_csv(labels_test)

    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        image_name = row["Image name"]
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
            image_name += ".jpg"

        input_path = os.path.join(input_test_dir, image_name)
        output_path = os.path.join(output_test_dir, image_name.replace(".tif", ".jpg"))

        image = Image.open(input_path).convert("RGB")

        # Step 1: crop circle
        image_small = image.resize((int(image.width*0.3), int(image.height*0.3)))
        image_small = np.array(image_small)
        image = crop_fundus_circle_hough(image_small)

        # Step 2: Graham pre-processing
        image = ben_graham_preprocess(image, sigma=35, scale=256)

        # Step 3: convert to PIL and save
        image = Image.fromarray(image)
        image.save(output_path, quality=95)

    print("Done! Preprocessed images saved to:")
    print(output_train_dir)
    print(output_test_dir)
