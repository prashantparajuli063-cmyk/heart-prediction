import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# Load saved model, scaler, and expected columns
# ---------------------------------------------------
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")


# ---------------------------------------------------
# Page title
# ---------------------------------------------------
st.title("❤️ Heart Disease Risk Prediction")

st.markdown(
    "Enter the patient's health information below to estimate "
    "the risk of heart disease."
)

st.info(
    "ℹ️ Please enter accurate values. This application is for "
    "educational/prediction purposes and is not a medical diagnosis."
)


# ---------------------------------------------------
# 1. AGE
# ---------------------------------------------------
age = st.slider(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)

st.caption(
    "📌 Age of the person in years. Enter the person's current age."
)


# ---------------------------------------------------
# 2. SEX
# ---------------------------------------------------
sex_option = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

# Convert user-friendly option back to dataset value
sex = "M" if sex_option == "Male" else "F"

st.caption(
    "📌 Biological sex of the person. "
    "Male = M, Female = F."
)


# ---------------------------------------------------
# 3. CHEST PAIN TYPE
# ---------------------------------------------------
chest_pain_option = st.selectbox(
    "Chest Pain Type",
    [
        "ATA - Atypical Angina",
        "NAP - Non-Anginal Pain",
        "TA - Typical Angina",
        "ASY - Asymptomatic"
    ]
)

# Convert displayed option back to dataset code
chest_pain = chest_pain_option.split(" - ")[0]

st.caption(
    "📌 Type of chest pain experienced by the person:\n\n"
    "• **TA (Typical Angina):** Chest pain with typical heart-related characteristics.\n"
    "• **ATA (Atypical Angina):** Chest pain with some, but not all, typical heart-related characteristics.\n"
    "• **NAP (Non-Anginal Pain):** Chest pain that is not typical of heart-related pain.\n"
    "• **ASY (Asymptomatic):** No chest pain or related symptoms."
)


# ---------------------------------------------------
# 4. RESTING BLOOD PRESSURE
# ---------------------------------------------------
resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=80,
    max_value=200,
    value=120
)

st.caption(
    "📌 Blood pressure measured while the person is resting. "
    "The value is given in mm Hg."
)


# ---------------------------------------------------
# 5. CHOLESTEROL
# ---------------------------------------------------
cholesterol = st.number_input(
    "Cholesterol (mg/dL)",
    min_value=100,
    max_value=600,
    value=200
)

st.caption(
    "📌 Blood cholesterol level. "
    "The value is measured in mg/dL."
)


# ---------------------------------------------------
# 6. FASTING BLOOD SUGAR
# ---------------------------------------------------
fasting_bs_option = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    [
        "No - 120 mg/dL or below",
        "Yes - Above 120 mg/dL"
    ]
)

# Convert to dataset value
fasting_bs = 1 if fasting_bs_option.startswith("Yes") else 0

st.caption(
    "📌 Indicates whether fasting blood sugar is above "
    "120 mg/dL. This is measured after fasting."
)


# ---------------------------------------------------
# 7. RESTING ECG
# ---------------------------------------------------
resting_ecg_option = st.selectbox(
    "Resting ECG",
    [
        "Normal",
        "ST - ST-T Wave Abnormality",
        "LVH - Left Ventricular Hypertrophy"
    ]
)

# Convert displayed option back to dataset value
if resting_ecg_option.startswith("ST"):
    resting_ecg = "ST"
elif resting_ecg_option.startswith("LVH"):
    resting_ecg = "LVH"
else:
    resting_ecg = "Normal"

st.caption(
    "📌 Resting ECG (electrocardiogram) records the heart's "
    "electrical activity while the person is resting.\n\n"
    "• **Normal:** Normal ECG result.\n"
    "• **ST:** ST-T wave abnormality.\n"
    "• **LVH:** Possible left ventricular hypertrophy."
)


# ---------------------------------------------------
# 8. MAXIMUM HEART RATE
# ---------------------------------------------------
max_hr = st.slider(
    "Maximum Heart Rate",
    min_value=60,
    max_value=220,
    value=150
)

st.caption(
    "📌 The maximum heart rate recorded during the exercise/test. "
    "Measured in beats per minute (BPM)."
)


# ---------------------------------------------------
# 9. EXERCISE-INDUCED ANGINA
# ---------------------------------------------------
exercise_angina_option = st.selectbox(
    "Exercise-Induced Angina",
    [
        "No",
        "Yes"
    ]
)

# Convert to dataset value
exercise_angina = "Y" if exercise_angina_option == "Yes" else "N"

st.caption(
    "📌 Indicates whether chest discomfort (angina) occurs "
    "during exercise or physical activity."
)


# ---------------------------------------------------
# 10. OLDPEAK
# ---------------------------------------------------
oldpeak = st.slider(
    "Oldpeak (ST Depression)",
    min_value=0.0,
    max_value=6.0,
    value=1.0,
    step=0.1
)

st.caption(
    "📌 Oldpeak represents the amount of ST-segment depression "
    "observed during exercise compared with rest. "
    "It is a value obtained from the ECG/exercise test."
)


# ---------------------------------------------------
# 11. ST SLOPE
# ---------------------------------------------------
st_slope_option = st.selectbox(
    "ST Slope",
    [
        "Up - Upsloping",
        "Flat - Flat",
        "Down - Downsloping"
    ]
)

# Convert displayed option back to dataset value
if st_slope_option.startswith("Up"):
    st_slope = "Up"
elif st_slope_option.startswith("Flat"):
    st_slope = "Flat"
else:
    st_slope = "Down"

st.caption(
    "📌 Describes the slope of the ST segment during the exercise test.\n\n"
    "• **Up:** Upsloping\n"
    "• **Flat:** Flat\n"
    "• **Down:** Downsloping"
)


# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------
st.divider()

if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):

    # -----------------------------------------------
    # Create raw input dictionary
    # -----------------------------------------------
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,

        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    # -----------------------------------------------
    # Create DataFrame
    # -----------------------------------------------
    input_df = pd.DataFrame([raw_input])

    # -----------------------------------------------
    # Add missing columns
    # -----------------------------------------------
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # -----------------------------------------------
    # Make sure column order is exactly the same
    # as used during model training
    # -----------------------------------------------
    input_df = input_df[expected_columns]

    # -----------------------------------------------
    # Scale input
    # -----------------------------------------------
    scaled_input = scaler.transform(input_df)

    # -----------------------------------------------
    # Prediction
    # -----------------------------------------------
    prediction = model.predict(scaled_input)[0]

    # -----------------------------------------------
    # Display result
    # -----------------------------------------------
    st.divider()

    if prediction == 1:
        st.error(
            "⚠️ The model predicts a HIGHER RISK of heart disease."
        )
    else:
        st.success(
            "✅ The model predicts a LOWER RISK of heart disease."
        )

    st.caption(
        "⚠️ This prediction is generated by a machine-learning model "
        "and should not be considered a medical diagnosis."
    )