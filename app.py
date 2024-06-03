import numpy as np
from keras.models import load_model
from flask import Flask, request
from io import BytesIO
from PIL import Image
from numpy import asarray

app = Flask(__name__)

@app.route("/predict_image", methods=['POST'])

def api():
    temp_image = BytesIO(request.files["image"].read())
    image = Image.open(temp_image)
    image = image.resize((32, 32))
    np_image = asarray(image)
    np_image = np.expand_dims(np_image, axis = 0)
    model = load_model("model.h5")
    result = model.predict(np_image)
    message = formating(result)
    return message

def formating(result):
    chosen = result.argmax()
    certainty = result.max()
    chosenStr = ""
    match chosen:
        case 0:
            chosenStr = "airplane"
        case 1:
            chosenStr = "automobile"
        case 2:
            chosenStr = "bird"
        case 3:
            chosenStr = "cat"
        case 4:
            chosenStr = "deer"
        case 5:
            chosenStr = "dog"
        case 6:
            chosenStr = "frog"
        case 7:
            chosenStr = "horse"
        case 8:
            chosenStr = "ship"
        case 9:
            chosenStr = "truck"
    message = {
        "prediction": "This is a " + chosenStr,
        "certainty": str(certainty)
    }, 200
    return message

if __name__ == "main":
    app.run(debug=True)