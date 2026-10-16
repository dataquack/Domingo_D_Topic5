import streamlit as st
import pandas as pd
import joblib

model = joblib.load(
    "student_grade_regression_model.pkl"
)

model_columns = joblib.load(
    "model_columns.pkl"
)

st.title("Student Final Grade Predictor")

st.write(
    "Predict a student's final grade using academic and demographic information."
)

G1 = st.slider("First Period Grade (G1)", 0, 20, 10)
G2 = st.slider("Second Period Grade (G2)", 0, 20, 10)

absences = st.slider("Absences", 0, 100, 5)

studytime = st.selectbox(
    "Study Time",
    [1, 2, 3, 4]
)

failures = st.selectbox(
    "Previous Failures",
    [0, 1, 2, 3]
)

if st.button("Predict Final Grade"):

    sample = pd.DataFrame(0, index=[0], columns=model_columns)
    sample.loc[0, [
        "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
        "famrel", "freetime", "goout", "Dalc", "Walc", "health",
        "absences", "G1", "G2",
    ]] = [
        17, 2, 2, 1, studytime, failures, 4, 3, 3, 1, 1, 3,
        absences, G1, G2,
    ]

    prediction = model.predict(sample)[0]

    st.subheader(
        f"Predicted Final Grade: {prediction:.2f}"
    )

    if prediction >= 15:
        st.success(
            "Expected Performance: Excellent"
        )

    elif prediction >= 10:
        st.info(
            "Expected Performance: Satisfactory"
        )

    else:
        st.error(
            "Expected Performance: At Risk"
        )