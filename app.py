from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
 
app = Flask(__name__, static_folder='static')

# Load the trained model and preprocessing scalers
with open('model.pkl', 'rb') as file:
    rfc, ms, sc = pickle.load(file)

# Load your dataset
data = pd.read_csv('crop_production.csv')  # Replace with the actual path to your dataset
columns_to_keep = ['State_Name', 'Crop', 'N', 'P', 'K', 'pH', 'rainfall', 'temperature']
data=data[columns_to_keep]
data.columns = data.columns.str.strip()
crop_counts = data['Crop'].value_counts()

# Get a list of crops with count >= 200
crops_to_keep = crop_counts[(crop_counts >= 2000) & (crop_counts <= 8000)].index.tolist()
# Filter rows with Crop in the list of crops to keep
data = data[data['Crop'].isin(crops_to_keep)]
data = data.dropna()
def get_limits_for_state(selected_state):
    # Filter the dataset for the selected state
    state_data = data[data['State_Name'] == selected_state]

    # Calculate maximum and minimum values for N, P, K, and rainfall
    max_n = state_data['N'].max()
    min_n = state_data['N'].min()
    max_p = state_data['P'].max()
    min_p = state_data['P'].min()
    max_k = state_data['K'].max()
    min_k = state_data['K'].min()
    max_rainfall = state_data['rainfall'].max()
    min_rainfall = state_data['rainfall'].min()

    return max_n, min_n, max_p, min_p, max_k, min_k, max_rainfall, min_rainfall

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    max_n, min_n, max_p, min_p, max_k, min_k, max_rainfall, min_rainfall = None, None, None, None, None, None, None, None

    if request.method == 'POST':
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorous'])
        K = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        rainfall = float(request.form['rainfall'])
        ph = float(request.form['ph'])
        state_num = int(request.form['region'])

        if 'region' in request.form:
            state_num = int(request.form['region'])
            selected_state = data['State_Name'].unique()[state_num - 1]
            max_n, min_n, max_p, min_p, max_k, min_k, max_rainfall, min_rainfall = get_limits_for_state(selected_state)

        features = np.array([[N, P, K, temperature, rainfall, ph, state_num]])
        transformed_features = ms.transform(features)
        transformed_features = sc.transform(transformed_features)
        prediction = rfc.predict(transformed_features)

        crop_dict = {
    1: 'cotton',
    2: 'horsegram',
    3: 'jowar',
    4: 'moong',
    5: 'ragi',
    6: 'sunflower',
    7: 'wheat',
    8: 'sesamum',
    9: 'soyabean',
    10: 'rapeseed',
    11: 'onion',
    12: 'potato',
    13: 'sweetpotato',
    14: 'turmeric',
    15: 'barley',
    16: 'banana',
    17: 'coriander',
    18: 'garlic'
}


        if prediction[0] in crop_dict:
            suitable_crop = crop_dict[prediction[0]]
            result = f"The best crop to be cultivated is: {suitable_crop}"
        else:
            result = "Sorry, we are not able to recommend a proper crop for this environment"

    return render_template('index.html', result=result, max_n=max_n, min_n=min_n, max_p=max_p, min_p=min_p, max_k=max_k, min_k=min_k, max_rainfall=max_rainfall, min_rainfall=min_rainfall)

if __name__ == "__main__":
    app.run(debug=True)