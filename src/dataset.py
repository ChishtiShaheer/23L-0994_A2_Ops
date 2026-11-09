"""
Task 2.1: tf.data Input Pipeline with Augmentation and Prefetching
Author: 23L-0994
Project: 23L-0994_A2_Ops
Loads split files, decodes, resizes, batches, prefetches, applies data augmentation to train set,
and saves a visualization of one augmented batch for the report.
"""
import os
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Augmentation layer (applied to training only)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="data_augmentation")

def parse_split_file(split_path):
    filepaths = []
    labels = []
    with open(split_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                filepath = parts[0]
                label_str = parts[1]
                label = 1 if ('ai' in label_str.lower() or 'fake' in label_str.lower()) else 0
                filepaths.append(filepath)
                labels.append(label)
    return filepaths, labels

def load_and_preprocess_image(path, label):
    img_raw = tf.io.read_file(path)
    img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    # Ensure shape is set
    img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return img, label

def create_dataset(split_path, is_training=False, batch_size=BATCH_SIZE, augment=True):
    filepaths, labels = parse_split_file(split_path)
    if not filepaths:
        raise ValueError(f"No records found in {split_path}")

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(filepaths), 5000), seed=42)

    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    if is_training and augment:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                              num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

def save_augmented_batch_figure(dataset, output_path="reports/augmented_batch.png"):
    os.makedirs(Path(output_path).parent, exist_ok=True)
    for images, labels in dataset.take(1):
        plt.figure(figsize=(10, 10))
        for i in range(min(9, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            # Clip between 0 and 255 for display
            img_disp = tf.clip_by_value(images[i], 0, 255).numpy().astype("uint8")
            plt.imshow(img_disp)
            lbl_name = "AI-Generated" if labels[i].numpy() == 1 else "Real"
            plt.title(f"Augmented: {lbl_name}")
            plt.axis("off")
        plt.suptitle("Task 2.1: Visualization of One Augmented Training Batch", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[+] Saved augmented batch figure to {output_path}")
        break

if __name__ == "__main__":
    splits_file = "data/splits/train.txt"
    if os.path.exists(splits_file):
        ds = create_dataset(splits_file, is_training=True)
        save_augmented_batch_figure(ds)
    else:
        print(f"Splits file {splits_file} not found. Run src/split_data.py first.")
