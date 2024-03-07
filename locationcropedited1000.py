import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle

# Read the crop data from CSV file
crop_data = pd.read_csv('Crop_production.csv')
columns_to_keep = ['State_Name', 'Crop', 'N', 'P', 'K', 'pH', 'rainfall', 'temperature']
crop_data=crop_data[columns_to_keep]
crop_data.columns = crop_data.columns.str.strip()
crop_counts = crop_data['Crop'].value_counts()

# Get a list of crops with count >= 200
crops_to_keep = crop_counts[(crop_counts >= 2000) & (crop_counts <= 8000)].index.tolist()
# Filter rows with Crop in the list of crops to keep
crop_data = crop_data[crop_data['Crop'].isin(crops_to_keep)]
crop_data = crop_data.dropna()

state_dict = {
    'andhra pradesh': 1,
    'arunachal pradesh': 2,
    'assam': 3,
    'bihar': 4,
    'goa': 5,
    'gujarat': 6,
    'haryana': 7,
    'jammu and kashmir': 8,
    'karnataka': 9,
    'kerala': 10,
    'madhya pradesh': 11,
    'maharashtra': 12,
    'manipur': 13,
    'meghalaya': 14,
    'mizoram': 15,
    'nagaland': 16,
    'odisha': 17,
    'punjab': 18,
    'rajasthan': 19,
    'tamil nadu': 20,
    'telangana': 21,
    'uttar pradesh': 22,
    'west bengal': 23,
    'chandigarh': 24,
    'dadra and nagar haveli': 25,
    'himachal pradesh': 26,
    'puducherry': 27,
    'sikkim': 28,
    'tripura': 29,
    'andaman and nicobar islands': 30,
    'chhattisgarh': 31,
    'uttarakhand': 32,
    'jharkhand': 33
}
crop_data['state_num'] = crop_data['State_Name'].map(state_dict)
crop_data.drop('State_Name', axis=1, inplace=True)
# Define a dictionary to map crop labels to numbers
crop_dict = {
    'cotton': 1,
    'horsegram': 2,
    'jowar': 3,
    'maize': 4,
    'moong': 5,
    'ragi': 6,
    'rice': 7,
    'sunflower': 8,
    'wheat': 9,
    'sesamum': 10,
    'soyabean': 11,
    'rapeseed': 12,
    'onion': 13,
    'potato': 14,
    'sweetpotato': 15,
    'turmeric': 16,
    'barley': 17,
    'banana': 18,
    'coriander': 19,
    'garlic': 20
}

# Map the labels to numbers and create a 'crop_num' column
crop_data['crop_num'] = crop_data['Crop'].map(crop_dict)

# Drop the 'Crop' column
crop_data.drop('Crop', axis=1, inplace=True)

# Separate features (x) and labels (y)
x = crop_data.drop('crop_num', axis=1)
y = crop_data['crop_num']

# Create MinMaxScaler and StandardScaler instances
ms = MinMaxScaler()
sc = StandardScaler()

# Fit and transform the scalers on the features
x = ms.fit_transform(x)
x = sc.fit_transform(x)

# Create a RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x, y)

# Save the trained model and preprocessing scalers to a pickle file
with open('model.pkl', 'wb') as file:
    pickle.dump((rfc, ms, sc), file)
