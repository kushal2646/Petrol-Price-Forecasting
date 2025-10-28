from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    prev1 = float(request.form['prev1'])
    prev2 = float(request.form['prev2'])
    prev3 = float(request.form['prev3'])

    # Predict using the model
    input_data = np.array([[prev1, prev2, prev3]])
    prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction_text=f'Predicted Petrol Price: ₹{prediction:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
