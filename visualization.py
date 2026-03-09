import os
import matplotlib.pyplot as plt
from PIL import Image
import random

# Dataset path
dataset_path = "dataset/cell_images"

parasitized_path = os.path.join(dataset_path, "Parasitized")
uninfected_path = os.path.join(dataset_path, "Uninfected")

# Count images
parasitized_count = len(os.listdir(parasitized_path))
uninfected_count = len(os.listdir(uninfected_path))

print("Parasitized Images:", parasitized_count)
print("Uninfected Images:", uninfected_count)

# Dataset distribution graph
labels = ['Parasitized', 'Uninfected']
counts = [parasitized_count, uninfected_count]

plt.figure(figsize=(6,4))
plt.bar(labels, counts, color=['red','green'])

plt.title("Malaria Dataset Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")

plt.show()

# Display sample images
fig, axes = plt.subplots(2,5, figsize=(12,6))

for i in range(5):

    img_path = os.path.join(parasitized_path, random.choice(os.listdir(parasitized_path)))
    img = Image.open(img_path)

    axes[0,i].imshow(img)
    axes[0,i].set_title("Parasitized")
    axes[0,i].axis("off")

for i in range(5):

    img_path = os.path.join(uninfected_path, random.choice(os.listdir(uninfected_path)))
    img = Image.open(img_path)

    axes[1,i].imshow(img)
    axes[1,i].set_title("Uninfected")
    axes[1,i].axis("off")

plt.tight_layout()
plt.show()