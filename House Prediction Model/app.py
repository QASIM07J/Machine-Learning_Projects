
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model
with open('house_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        bedrooms = request.form.get("BEDROOMS")
        bathrooms = request.form.get("BATHROOMS")
        garage = request.form.get("GARAGE")
        floor_area = request.form.get("FLOOR_AREA")
        land_area = request.form.get("LAND_AREA")
        longitude = request.form.get("LONGITUDE")
        latitude = request.form.get("LATITUDE")
        cbd_dist = request.form.get("CBD_DIST")

        if bedrooms and bathrooms and garage and floor_area and land_area and longitude and latitude and cbd_dist:
            # Convert values to float
            bedrooms = float(bedrooms)
            bathrooms = float(bathrooms)
            garage = float(garage)
            floor_area = float(floor_area)
            land_area = float(land_area)
            longitude = float(longitude)
            latitude = float(latitude)
            cbd_dist = float(cbd_dist)

            # Create a list to hold the input values
            input_data = [[bedrooms, bathrooms, garage, floor_area, land_area, longitude, latitude, cbd_dist]]

            # Make prediction
            prediction = model.predict(input_data)[0]

            return render_template('index.html', prediction_text=f"Predicted House Price: ${prediction:.2f}")
        else:
            return "Please fill in all fields", 400
    return render_template('index.html')

if __name__ == '__main__':
    app.run()