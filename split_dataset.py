import os
import shutil
import random

dataset_dir = "dataset/cell_images"

train_dir = "dataset/train"
val_dir = "dataset/validation"

classes = ["Parasitized", "Uninfected"]

for cls in classes:

    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    images = os.listdir(os.path.join(dataset_dir, cls))
    random.shuffle(images)

    split = int(0.8 * len(images))

    train_images = images[:split]
    val_images = images[split:]

    for img in train_images:
        shutil.copy(
            os.path.join(dataset_dir, cls, img),
            os.path.join(train_dir, cls, img)
        )

    for img in val_images:
        shutil.copy(
            os.path.join(dataset_dir, cls, img),
            os.path.join(val_dir, cls, img)
        )

print("Dataset split completed")
