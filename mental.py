import pandas as pd
import joblib

# Load the model
model = joblib.load('./heart_disease_model.pkl')

# Assuming you have your RandomForestRegressor model 'rf' already loaded and trained

# Manually provide the values for each feature
BMI = 35.44
Smoking = 1
AlcoholDrinking = 0
Stroke = 0
PhysicalHealth = 0.0
MentalHealth = 0.0
DiffWalking = 0
Sex = 1
AgeCategory = 50
Race = 3
Diabetic = 0
PhysicalActivity = 0
GenHealth = 4
SleepTime = 6.0
Asthma = 0
KidneyDisease = 0
SkinCancer = 0


data = pd.DataFrame({
    'BMI': [BMI],
    'Smoking': [Smoking],
    'AlcoholDrinking': [AlcoholDrinking],
    'Stroke': [Stroke],
    'PhysicalHealth': [PhysicalHealth],
    'MentalHealth': [MentalHealth],
    'DiffWalking': [DiffWalking],
    'Sex': [Sex],
    'AgeCategory': [AgeCategory],
    'Race': [Race],
    'Diabetic': [Diabetic],
    'PhysicalActivity': [PhysicalActivity],
    'GenHealth': [GenHealth],
    'SleepTime': [SleepTime],
    'Asthma': [Asthma],
    'KidneyDisease': [KidneyDisease],
    'SkinCancer': [SkinCancer]
})

# Use the model to predict on this data
prediction = model.predict(data)

# Print the prediction
print(prediction)
