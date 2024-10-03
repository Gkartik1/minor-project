from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained machine learning model
model = pickle.load(open('saved_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction-form')
def prediction_form():
    return render_template('prediction-form.html')

@app.route('/sound-upload')
def sound_upload():
    return render_template('sound-upload.html')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/health-info')
def health_info():
    return render_template('health-info.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.form
        
        # Define the expected keys in the correct order
        expected_keys = ['age', 'sex', 'cp', 'restbp', 'chol', 'fbs', 'restecg', 
                         'thalach', 'exang', 'oldpeak', 'slope', 'target']
        
        # Prepare the input data for the model
        input_data = [float(data[key]) for key in expected_keys]
        prediction = model.predict([input_data])[0]
        
        # Return the prediction
        return jsonify({
            'status': 'success',
            'prediction': int(prediction)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True)