import joblib
import numpy as np
import os

# Example function to take input from the user with input validation
def get_user_input():
    try:
        # Taking input for CropType
        crop_type = float(input("Enter the Crop Type (e.g., 1 for Wheat, 2 for Rice): "))
        
        # Taking input for CropDays
        crop_days = float(input("Enter the number of Crop Days: "))
        
        # Taking input for SoilMoisture
        soil_moisture = float(input("Enter the Soil Moisture percentage: "))
        
        # Taking input for Temperature
        temperature = float(input("Enter the Temperature in Celsius: "))
        
        # Taking input for Humidity
        humidity = float(input("Enter the Humidity percentage: "))
        
        # Convert the input into a format suitable for the model (e.g., an array of numbers)
        input_data = np.array([[crop_type, crop_days, soil_moisture, temperature, humidity]])
        
        return input_data
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return get_user_input()  # Retry if input is invalid


# Assuming your model is saved as 'svm_model.pkl'
model_path = 'Z:/irigation optimization/Prediction-for-Crop-Irrigation-System-using-ML-and-DL/svm_model.pkl'

# Ensure the path is correct
if os.path.exists(model_path):
    svm_model = joblib.load(model_path)
else:
    print("Model file not found. Please check the path.")
    exit()

# Get input data from the user
input_data = get_user_input()

# Make a prediction using the trained model
prediction = svm_model.predict(input_data)

# Output the prediction result
print(f"The predicted outcome for the given input is: {prediction[0]}")
