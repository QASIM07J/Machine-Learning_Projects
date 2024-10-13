# app.py

import joblib
from flask import Flask, request, render_template

# Initialize the Flask application
app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Load the fitted vectorizer

# Define a route for the default URL
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None  # Initialize prediction variable
    if request.method == 'POST':
        message = request.form['message']
        # Transform the input message to the same format as the training data
        message_vectorized = vectorizer.transform([message])
        prediction = model.predict(message_vectorized)

        if prediction[0] == 1:
            result = "Spam"
        else:
            result = "Not-Spam"
        
        return render_template('index.html', prediction=result)

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
