from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('Breast_Cancer_model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('CancerUI.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])
        result = "Cancer is diagnosed" if prediction[0] == 1 else "Cancer is not diagnosed"
        return render_template('Result.html', prediction_text=f'The diagnosis is {result}.')


if __name__ == '__main__':
    app.run(debug=True)
