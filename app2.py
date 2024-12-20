import streamlit as st
import pickle
import numpy as np

# Load the trained model
filename = "model.sv"
model = pickle.load(open(filename, 'rb'))

# Dictionaries for categorical variables
sex_d = {0: "Kobieta", 1: "Mężczyzna"}
chest_pain_d = {0: "ATA", 1: "NAP", 2: "ASY", 3: "TA"}
resting_ecg_d = {0: "Normal", 1: "ST", 2: "LVH"}
st_slope_d = {0: "Up", 1: "Flat", 2: "Down"}

def main():
    # Configure the Streamlit app page
    st.set_page_config(page_title="Heart Disease Prediction App")

    # Layout containers
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    # Image for the app
    st.image("https://leksykon.com.pl/image?name=images/ekg22.jpg&width=750&height=350")

    # Overview section
    with overview:
        st.title("Heart Disease Prediction App")

    # Input section in the left column
    with left:
        # Radio buttons for categorical variables
        sex_radio = st.radio("Płeć", list(sex_d.keys()), format_func=lambda x: sex_d[x])
        chest_pain_radio = st.radio("Ból w Klatce Piersiowej", list(chest_pain_d.keys()), format_func=lambda x: chest_pain_d[x])
        resting_ecg_radio = st.radio("EKG", list(resting_ecg_d.keys()), format_func=lambda x: resting_ecg_d[x])
        st_slope_radio = st.radio("ST Slope", list(st_slope_d.keys()), format_func=lambda x: st_slope_d[x])

    # Input section in the right column
    with right:
        # Sliders for numerical inputs
        age_slider = st.slider("Wiek", min_value=20, max_value=80, value=50)
        resting_bp_slider = st.slider("Spoczynkowe Ciśnienie Krwi", min_value=80, max_value=200, value=120)
        cholesterol_slider = st.slider("Poziom Cholesterolu", min_value=100, max_value=400, value=200)
        max_hr_slider = st.slider("Maksymalne Tętno", min_value=60, max_value=220, value=150)
        exercise_angina_radio = st.radio("Dławica piersiowa", [0, 1], format_func=lambda x: "Tak" if x == 1 else "Nie")

    # Prediction section
    with prediction:
        st.subheader("Czy wystąpi choroba serca?")

        # Prepare input data for prediction
        input_data = np.array([[
            age_slider,
            sex_radio,
            chest_pain_radio,
            resting_bp_slider,
            cholesterol_slider,
            0,  # FastingBS (not included in inputs)
            resting_ecg_radio,
            max_hr_slider,
            exercise_angina_radio,
            0.0,  # Oldpeak (not included in inputs)
            st_slope_radio
        ]])

        # Perform prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Display prediction results
        if prediction[0] == 1:
            st.success(f"Wystąpi choroba serca z prawdopodobieństwem {prediction_proba[0][1] * 100:.2f}%.")
        else:
            st.error(f"Brak choroby serca z prawdopodobieństwem {prediction_proba[0][0] * 100:.2f}%.")

if __name__ == '__main__':
    main()
