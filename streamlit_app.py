import streamlit as st
import joblib
import numpy as np

# List of crop labels for the dropdown
CROP_LABELS = [
    "Rice", "Maize", "Chickpea", "Kidneybeans", "Pigeonpeas", "Mothbeans", 
    "Mungbeans", "Blackgram", "Lentil", "Pomegranate", "Banana", "Mango", 
    "Grapes", "Watermelon", "Muskmelon", "Apple", "Orange", "Papaya", 
    "Coconut", "Cotton", "Jute", "Coffee"
]

# Load the pre-trained model (ensure model path is correct)
model_path = 'Z:/irigation optimization/Prediction-for-Crop-Irrigation-System-using-ML-and-DL/DT_model.pkl'
DT_model = joblib.load(model_path)

# Streamlit app code
def main():
    st.title("Crop Irrigation Prediction")

    # Dropdown for selecting the crop label
    label = st.selectbox("Select Crop Label", CROP_LABELS)

    # Taking user input via Streamlit widgets
    crop_days = st.number_input("Enter the number of Crop Days (days passed after sowing the crop)", min_value=0, value=10)
    soil_moisture = st.number_input("Enter the Soil Moisture percentage (read by a soil moisture sensor)", min_value=0.0, max_value=100.0, value=30.0)
    temperature = st.number_input("Enter the Temperature in Celsius (°C)", min_value=-10.0, max_value=60.0, value=25.0)
    humidity = st.number_input("Enter the Humidity percentage (%)", min_value=0.0, max_value=100.0, value=60.0)
    n = st.number_input("Enter the Nitrogen content (N)", min_value=0, max_value=100, value=10)
    p = st.number_input("Enter the Phosphorus content (P)", min_value=0, max_value=100, value=10)
    k = st.number_input("Enter the Potassium content (K)", min_value=0, max_value=100, value=10)
    ph = st.number_input("Enter the Soil pH level", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Enter the Rainfall (in mm)", min_value=0.0, max_value=500.0, value=100.0)

    # Prepare the input data for the model
    input_data = np.array([[crop_days, soil_moisture, temperature, humidity, n, p, k, ph, rainfall]])

    # When the user clicks the "Predict" button
    if st.button("Predict Irrigation Requirement"):
        # Make prediction
        prediction = DT_model.predict(input_data)

        # Display the result
        if prediction[0] == 1:
            st.success("Prediction: Irrigation is Required.")
        else:
            st.success("Prediction: No Irrigation is Required.")

        # Display the selected crop label
        st.write(f"Selected Crop Label: {label}")

# Running the main function
if __name__ == '__main__':
    main()
