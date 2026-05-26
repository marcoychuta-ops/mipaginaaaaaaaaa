import streamlit as st
import pandas as pd
import joblib

# -------------------------
# CONFIGURACIÓN
# -------------------------
st.set_page_config(
    page_title="Predicción de Diabetes",
    page_icon="🩺",
    layout="centered"
)

# -------------------------
# CARGAR MODELOS
# -------------------------
@st.cache_resource
def cargar_modelos():

    modelo_lr = joblib.load(
        "model/logistic_regression_model.pkl"
    )

    modelo_rf = joblib.load(
        "model/random_forest_model.pkl"
    )

    return modelo_lr, modelo_rf


modelo_lr, modelo_rf = cargar_modelos()

# -------------------------
# INTERFAZ
# -------------------------
st.title("🩺 Predicción de Diabetes")

st.write("""
Ingrese las medidas del paciente y seleccione el modelo para obtener la predicción.

**Target → Outcome**
- 0 → No Diabetes
- 1 → Diabetes
""")

# Selección modelo
modelo_seleccionado = st.selectbox(
    "Seleccione el modelo",
    [
        "Logistic Regression",
        "Random Forest"
    ]
)

st.subheader("Datos del Paciente")

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        value=0
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0.0
    )

    blood_pressure = st.number_input(
        "BloodPressure",
        min_value=0.0
    )

    skin_thickness = st.number_input(
        "SkinThickness",
        min_value=0.0
    )

with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0
    )

    diabetes_pedigree = st.number_input(
        "DiabetesPedigreeFunction",
        min_value=0.0,
        format="%.3f"
    )

    age = st.number_input(
        "Age",
        min_value=1
    )

# -------------------------
# BOTÓN PREDICCIÓN
# -------------------------
if st.button("🔍 Obtener Predicción"):

    datos = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

    if modelo_seleccionado == "Logistic Regression":
        modelo = modelo_lr
    else:
        modelo = modelo_rf

    pred = modelo.predict(datos)[0]

    st.subheader("Resultado")

    if pred == 1:
        st.error(
            "Outcome = 1 → Tiene Diabetes"
        )
    else:
        st.success(
            "Outcome = 0 → No Tiene Diabetes"
        )

    try:
        prob = modelo.predict_proba(datos)

        st.progress(
            float(prob[0][1])
        )

        st.write(
            f"Probabilidad Diabetes: {prob[0][1]*100:.2f}%"
        )

    except:
        pass
