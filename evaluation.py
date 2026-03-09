import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image settings
IMG_SIZE = (128,128)
BATCH_SIZE = 32

# Load validation data
val_datagen = ImageDataGenerator(rescale=1./255)

val_gen = val_datagen.flow_from_directory(
    "dataset/validation",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# Load best model
model = tf.keras.models.load_model("best_malaria_model.h5")

# Predictions
predictions = model.predict(val_gen)

y_pred = (predictions > 0.5).astype(int)

y_true = val_gen.classes

# -------------------------
# 1️⃣ Confusion Matrix
# -------------------------

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Parasitized","Uninfected"],
    yticklabels=["Parasitized","Uninfected"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()


# -------------------------
# 2️⃣ ROC Curve
# -------------------------

fpr, tpr, thresholds = roc_curve(y_true, predictions)

roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc)

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend(loc="lower right")

plt.show()


# -------------------------
# 3️⃣ Accuracy Comparison Chart
# -------------------------

models = ["CustomCNN","MobileNetV2","EfficientNetB0"]

accuracies = [0.948, 0.83, 0.50]  # use your training results

plt.figure(figsize=(6,4))

plt.bar(models, accuracies, color=["green","orange","red"])

plt.ylabel("Accuracy")

plt.title("Model Accuracy Comparison")

plt.ylim(0,1)

plt.show()