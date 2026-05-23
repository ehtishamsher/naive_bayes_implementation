from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# load trained model
model = pickle.load(open("naive_bayes_tips_model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = np.array(
        [
            [
                float(data["total_bill"]),
                float(data["tip"]),
                int(data["size"]),
                int(data["sex_Female"]),
                int(data["sex_Male"]),
                int(data["day_Fri"]),
                int(data["day_Sat"]),
                int(data["day_Sun"]),
                int(data["day_Thur"]),
                int(data["time_Dinner"]),
                int(data["time_Lunch"]),
            ]
        ]
    )

    prediction = model.predict(features)[0]

    result = "Smoker" if prediction == 1 else "Non-Smoker"

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)
