import numpy as np
import pandas as pd
import streamlit as st
from pipeline import LoanRecoveryPipeline

st.set_page_config(
    page_title="ARC Loan Recovery Rate Predictor", page_icon="🏦", layout="wide"
)


@st.cache_resource
def load_model_pipeline():
    return LoanRecoveryPipeline.load("model_pipeline.joblib")


try:
    pipeline_instance = load_model_pipeline()
except Exception:
    st.error(
        "Model artifact missing! Run `python train.py` first to create `model_pipeline.joblib`."
    )
    st.stop()

st.title("🏦 ARC Loan Recovery Rate Predictor")
st.markdown(
    "Predict expected recovery rates on defaulted loan accounts using XGBoost."
)

(tab1,) = st.tabs(["Single Prediction Input"])

# --- TAB 1: SINGLE PREDICTION INPUT ---
with tab1:
    st.subheader("Loan & Borrower Information")

    # Column section for Upper Profile Inputs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.caption("**Loan & Terms**")
        loan_amount = st.number_input(
            "Loan Amount ($)", 1000.0, 100000.0, 15000.0, step=500.0
        )
        term_months = st.selectbox("Term Months", [36, 60])
        int_rate = st.number_input(
            "Interest Rate (%)", 1.0, 40.0, 12.5, step=0.5
        )
        purpose = st.selectbox(
            "Purpose",
            [
                "debt_consolidation",
                "credit_card",
                "home_improvement",
                "major_purchase",
                "small_business",
                "car",
                "other",
            ],
        )

    with col2:
        st.caption("**Risk & Credit Scores**")
        grade = st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"], index=1)
        sub_grade = st.text_input("Sub-grade", "B3", help = "A-> low Risk | G->High Risk" | Assigned by Lender)

        fico_low = st.slider(
            "FICO Range",
            min_value=300,
            max_value=846,
            value=680,
            step=5,
            help="Select lower bound. Upper bound is automatically lower bound + 4.",
        )
        fico_high = fico_low + 4
        st.caption(f"Selected FICO Range: **{fico_low} – {fico_high}**")

        dti = st.number_input(
            "Debt-To-Income Ratio (%)",
            -999.0,
            100.0,
            18.5,
            help="-999 if missing/failed pull",
        )

    with col3:
        st.caption("**Borrower Profile**")
        emp_length_years = st.number_input(
            "Employment (Years)", 0.0, 50.0, 5.0
        )
        home_ownership = st.selectbox(
            "Home Ownership", ["RENT", "MORTGAGE", "OWN", "ANY"]
        )
        annual_income = st.number_input(
            "Annual Income ($)", 0.0, 1000000.0, 65000.0, step=1000.0
        )
        verification_status = st.selectbox(
            "Income Verification Status",
            ["Verified", "Source Verified", "Not Verified"],
        )

    with col4:
        st.caption("**Credit History & Default Status**")
        delinq_2yrs = st.number_input("30+ days payment delay count", 0, 20, 0)
        pub_rec = st.number_input("Derogatory Public Records", 0, 20, 0)
        open_acc = st.number_input("Open Credit Accounts", 0, 100, 8)
        total_acc = st.number_input("Total Credit Accounts ever", 0, 100, 18)
        revol_bal = st.number_input("Revolving Credit Balance ($)", 0.0, 500000.0, 12000.0, step = 500.0, help = "unpaid monthly bill that carries over to the next billing cycle")
        revol_util = st.number_input("Revolving Util (%)", 0.0, 200.0, 55.0)
        collections_12m = st.number_input("Debts taken in 12M (Non-Medical)", 0, 10, 0)

    st.divider()
    st.caption("**Collateral & Post-Default Details**")
    
    # 3-column layout at the bottom
    c1, c2, c3 = st.columns(3)

    with c1:
        # Reactive selection outside st.form triggers instant rerun
        collateral_flag = st.selectbox(
            "Secured by Collateral?",
            options=[0, 1],
            format_func=lambda x: "No (Unsecured)" if x == 0 else "Yes (Secured)",
            index=0,
            key="collateral_flag_select"
        )
        
        # Instantly displays Collateral Value box right under the dropdown
        if collateral_flag == 1:
            collateral_value = st.number_input(
                "Collateral Value ($)", 0.0, 500000.0, 5000.0, step=500.0
            )
        else:
            collateral_value = 0.0

    with c2:
        total_pymnt_before_default = st.number_input(
            "Paid Before Default ($)", 0.0, 100000.0, 3200.0, step = 1000.0
        )
        outstanding_principal = loan_amount - total_pymnt_before_default
        st.caption(f"Outstanding Principal is {outstanding_principal}")

    with c3:
        days_past_due_at_default = st.number_input("Days Past Due at Default", 0, 365, 120)
        months_in_recovery = st.number_input("Months in Recovery", 0, 120, 6)

    st.markdown("<br>", unsafe_allow_html=True)
    submit_btn = st.button("Predict Recovery Rate", type="primary")

    if submit_btn:
        input_data = pd.DataFrame(
            [
                {
                    "loan_id": "LOAN_AUTO_GEN",
                    "loan_amount": loan_amount,
                    "term_months": term_months,
                    "grade": grade,
                    "sub_grade": sub_grade,
                    "int_rate": int_rate,
                    "emp_length_years": emp_length_years,
                    "home_ownership": home_ownership,
                    "annual_income": annual_income,
                    "verification_status": verification_status,
                    "purpose": purpose,
                    "dti": dti,
                    "fico_range_low": fico_low,
                    "fico_range_high": fico_high,
                    "delinq_2yrs": delinq_2yrs,
                    "pub_rec": pub_rec,
                    "open_acc": open_acc,
                    "total_acc": total_acc,
                    "revol_bal": revol_bal,
                    "revol_util": revol_util,
                    "collections_12_mths_ex_med": collections_12m,
                    "collateral_flag": collateral_flag,
                    "collateral_value": collateral_value,
                    "total_pymnt_before_default": total_pymnt_before_default,
                    "outstanding_principal": outstanding_principal,
                    "days_past_due_at_default": days_past_due_at_default,
                    "months_in_recovery": months_in_recovery,
                }
            ]
        )

        pred_val = pipeline_instance.predict(input_data)[0]
        est_dollars = pred_val * outstanding_principal

        st.metric(
            label="Predicted Recovery Rate", value=f"{pred_val * 100:.2f}%"
        )
        st.metric(
            label="Estimated Dollar Recovery", value=f"${est_dollars:,.2f}"
        )
        st.progress(float(pred_val))
