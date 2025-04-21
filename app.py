import streamlit as st
import gdown
import joblib
import os
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestClassifier


@st.cache_resource
def load_model():
    url = 'https://drive.google.com/uc?id=1XnV4cRtWfh7V9v30EoKSyXj0PYS0S4-l'
    output = 'best_model.pkl'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    return joblib.load(output)

education_map = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}
home_map = {'MORTGAGE': 0.115961, 'OTHER': 0.333333, 'OWN': 0.075229, 'RENT': 0.323977}
intent_map = {
    'DEBT CONSOLIDATION': 0.302729, 'EDUCATION': 0.169562,
    'HOME IMPROVEMENT': 0.263015, 'MEDICAL': 0.278194,
    'PERSONAL': 0.201404, 'VENTURE': 0.144264
}

def repair_gender(gender):
    gender = gender.lower().replace(" ", "")
    if "fe" in gender or "fem" in gender:
        return 0  
    return 1  

def scale_input(input_user):
    scaler = QuantileTransformer(output_distribution='normal')
    input_user = np.array(input_user).reshape(1, -1)
    return scaler.fit_transform(input_user)[0]

def main():
    st.title('Loan Approval Prediction App (UTS - 2702274835)')

    person_age = st.number_input("Person's Age", min_value=18, step=1)
    person_gender = st.selectbox("Gender", ["Male", "Female"])
    person_education = st.selectbox("Education Level", list(education_map.keys()))
    person_income = st.number_input("Annual Income", min_value=0.0, step=100.0)
    person_emp_exp = st.number_input("Employment Experience (Years)", min_value=0.0, step=1.0)
    person_home_ownership = st.selectbox("Home Ownership", list(home_map.keys()))
    loan_amnt = st.number_input("Loan Amount", min_value=0.0, step=100.0)
    loan_intent = st.selectbox("Loan Intent", list(intent_map.keys()))
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, step=0.1)
    loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, step=0.1)
    cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=0, step=1)
    credit_score = st.number_input("Credit Score", min_value=0.0, step=1.0)
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Default on File?", ["No", "Yes"])

    if st.button("Make Prediction"):
        gender_enc = repair_gender(person_gender)
        edu_enc = education_map[person_education]
        home_enc = home_map[person_home_ownership]
        intent_enc = intent_map[loan_intent]
        default_enc = 1 if previous_loan_defaults_on_file == "Yes" else 0

        features = [
            person_age, gender_enc, edu_enc, person_income,
            person_emp_exp, home_enc, loan_amnt, intent_enc,
            loan_int_rate, loan_percent_income, cb_person_cred_hist_length,
            credit_score, default_enc
        ]

        scaled_features = scale_input(features)
        prediction = model.predict([scaled_features])[0]

        st.success(f"The predicted loan status is: {prediction}")

if __name__ == '__main__':
    main()
