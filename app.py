from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model and scaler
model = pickle.load(open('model_file.pkl', 'rb'))
scaler = pickle.load(open('scaler_file.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve data from form
        bedrooms = float(request.form.get('bedrooms'))
        bathrooms = float(request.form.get('bathrooms'))
        sqft_living = float(request.form.get('sqft_living'))
        floors = float(request.form.get('floors'))
        condition = int(request.form.get('condition'))
        yr_built = int(request.form.get('yr_built'))

        # Create an input array for the model
        input_query = np.array([[bedrooms, bathrooms, sqft_living, floors, condition, yr_built]])

        # Scale the input using the pre-trained scaler
        input_query_scaled = scaler.transform(input_query)

        # Make the prediction using the model
        prediction = model.predict(input_query_scaled)

        # Render the result to the template
        return render_template('index.html', prediction=round(prediction[0], 2))

if __name__ == '__main__':
    app.run(debug=True)
