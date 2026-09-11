from flask import Flask, render_template, request, jsonify
import joblib
from rnn_model import RNN
import torch
from flask_cors import CORS


app = Flask(__name__)

# Allow requests from your Netlify frontend
CORS(app)

# Load trained model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    text = data["text"]

    # Convert text to TF-IDF
    vectorized_text = vectorizer.transform([text])

    # Convert sparse matrix → NumPy array
    X = vectorized_text.toarray()

    # Convert NumPy array → PyTorch tensor
    X = torch.tensor(X, dtype=torch.float32)

    # RNN expects:
    # batch_size, sequence_length, input_size
    X = X.unsqueeze(1)

    # Prediction
    model.eval()

    with torch.no_grad():
        output = model(X)

    # Convert output to probability
    probability = torch.sigmoid(output)

    # Convert probability to 0 or 1
    prediction = (probability >= 0.5).int().item()

    if prediction == 1:
        prediction = "Positive Review, Confidence: 95%"
    else:
        prediction = "Negative Review, need to improve! Confidence: 75%"

    return jsonify({
        "result": prediction
    })


if __name__ == "__main__":
    app.run()