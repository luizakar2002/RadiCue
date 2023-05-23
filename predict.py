from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import model_utils

app = Flask(__name__)
CORS(app)

loaded_model=None
def load_model():
    global loaded_model
    loaded_model = tf.keras.models.load_model('model.h5', compile=False)

@app.route("/predict", methods=["POST", "GET"])
def predict():
    data = request.get_json()
    print(f"Received data: {data}")
    print(f"Received image url: {data['imageUrl']}")
    response = model_utils.predictFromModel(data['imageUrl'])
    print(f"Response: {response}")
    return response


if __name__ == "__main__":
    load_model()
    app.run()
