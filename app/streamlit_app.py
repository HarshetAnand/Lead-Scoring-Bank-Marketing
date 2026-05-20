"""
Streamlit demo for the lead scoring model.

Loads the trained model and lets the user input lead attributes to
get a real-time score and tier classification.

Usage:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import pickle

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scoring import probability_to_score, assign_tier


MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'model.pkl')


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def main():
    st.set_page_config(page_title="Lead Scoring Demo", page_icon="📊", layout="centered")

    st.title("Lead Scoring Demo")
    st.markdown(
        "Score a lead from 0 to 100 based on demographic, financial, "
        "and campaign features. Built on the UCI Bank Marketing dataset."
    )

    model = load_model()
    if model is None:
        st.error(
            "Trained model not found. Run `python src/train.py` first to "
            "generate `outputs/model.pkl`."
        )
        return

    st.divider()
    st.subheader("Lead Attributes")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 95, 35)
        job = st.selectbox("Job", [
            'admin.', 'blue-collar', 'technician', 'services', 'management',
            'retired', 'self-employed', 'entrepreneur', 'unemployed',
            'housemaid', 'student'
        ])
        marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
        education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
        balance = st.number_input("Account Balance", value=1500, step=100)
        default = st.selectbox("Has Credit in Default", ['no', 'yes'])
        housing = st.selectbox("Has Housing Loan", ['no', 'yes'])
        loan = st.selectbox("Has Personal Loan", ['no', 'yes'])

    with col2:
        contact = st.selectbox("Contact Type", ['cellular', 'telephone', 'unknown'])
        month = st.selectbox("Contact Month", [
            'jan', 'feb', 'mar', 'apr', 'may', 'jun',
            'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ])
        duration = st.slider("Last Call Duration (seconds)", 0, 2000, 250)
        campaign = st.slider("Contacts This Campaign", 1, 30, 2)
        previous = st.slider("Contacts Before This Campaign", 0, 20, 0)
        poutcome = st.selectbox("Previous Outcome", ['unknown', 'success', 'failure', 'other'])

    if st.button("Score Lead", type='primary', use_container_width=True):
        lead = pd.DataFrame([{
            'age': age, 'job': job, 'marital': marital, 'education': education,
            'default': default, 'balance': balance, 'housing': housing,
            'loan': loan, 'contact': contact, 'month': month,
            'duration': duration, 'campaign': campaign, 'previous': previous,
            'poutcome': poutcome
        }])

        from feature_engineering import engineer_features, get_categorical_features, get_numerical_features
        lead_eng = engineer_features(lead)
        feature_cols = get_categorical_features() + get_numerical_features()

        prob = model.predict_proba(lead_eng[feature_cols])[0, 1]
        score = int(probability_to_score(prob))
        tier = assign_tier(score)

        st.divider()
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Lead Score", f"{score}/100")
        col_b.metric("Tier", tier)
        col_c.metric("Conversion Probability", f"{prob:.1%}")

        st.progress(score / 100)

        if score >= 80:
            st.success("Hot lead - prioritize immediate outreach.")
        elif score >= 60:
            st.info("Warm lead - schedule follow-up within 48 hours.")
        elif score >= 40:
            st.warning("Lukewarm lead - add to nurture sequence.")
        else:
            st.error("Cold lead - low priority, consider deprioritizing.")


if __name__ == '__main__':
    main()
