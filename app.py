from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model("best_malaria_model.h5")

def prepare_image(image):

    image = image.resize((128,128))
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0)

    return image


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    image = Image.open(file).convert("RGB")

    img = prepare_image(image)

    prediction = float(model.predict(img)[0][0])

    if prediction > 0.5:
        result = "UNINFECTED"
        confidence = prediction * 100
    else:
        result = "PARASITIZED"
        confidence = (1 - prediction) * 100

    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence,2)
    )


if __name__ == "__main__":
    app.run(debug=True)