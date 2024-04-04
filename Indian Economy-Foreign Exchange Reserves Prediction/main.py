import datetime
from unittest import result
import pandas as pd
import streamlit as st
import joblib


model = joblib.load('foreign_reserves_predictor.joblib')
scaler = joblib.load('scaler.joblib')

# Setting up Streamlit app page
st.set_page_config(page_title='IndianFutureReserves', page_icon='🤖', menu_items={'Get Help': 'https://github.com/PranayJagtap06/ML_Projects/tree/main/Indian Economy-Foreign Exchange Reserves Prediction', 'About': "To be updated soon."})
st.title(body="IndianFutureReserves: Predicting India's Financial Fortunes 🔮💰")
st.markdown("*Leverage the power of machine learning to unveil future trends in India's foreign exchange reserves with pinpoint precision, fueled by authoritative RBI data.*")

# columns = ['Period', 'Forward Premia of US$ 1-month (%)', 'Forward Premia of US$ 3-month (%)', 'Forward Premia of US$ 6-month (%)', 'Reverse Repo Rate (%)', 'Marginal Standing Facility (MSF) Rate (%)', 'Bank Rate (%)', 'Base Rate (%)', '91-Day Treasury Bill (Primary) Yield (%)', '182-Day Treasury Bill (Primary) Yield (%)', '364-Day Treasury Bill (Primary) Yield (%)', '10-Year G-Sec Yield (FBIL) (%)', 'Cash Reserve Ratio (%)', 'Statutory Liquidity Ratio (%)', 'Policy Repo Rate (%)', 'Standing Deposit Facility (SDF) Rate (%)']

period = st.date_input('Date', min_value=datetime.date(2023, 11, 1), max_value=datetime.date(2074, 12, 31))
fp_1month = st.slider("Forward Premia of US$ 1-month (%)", min_value=1.00, max_value=100.00, value=7.25)
fp_3month = st.slider("Forward Premia of US$ 3-month (%)", min_value=1.00, max_value=100.00, value=4.15)
fp_6month = st.slider("Forward Premia of US$ 6-month (%)", min_value=1.00, max_value=100.00, value=3.25)
repo_rate = st.slider("Reverse Repo Rate (%)", min_value=1.00, max_value=100.00, value=5.50)
msf_rate = st.slider("Marginal Standing Facility (MSF) Rate (%)", min_value=1.00, max_value=100.00, value=6.25)
bank_rate = st.slider("Bank Rate (%)", min_value=1.00, max_value=100.00, value=5.15)
base_rate = st.slider("Base Rate (%)", min_value=1.00, max_value=100.00, value=8.00)
tby_91 = st.slider("91-Day Treasury Bill (Primary) Yield (%)", min_value=1.00, max_value=100.00, value=6.15)
tby_182 = st.slider("182-Day Treasury Bill (Primary) Yield (%)", min_value=1.00, max_value=100.00, value=4.63)
tby_364 = st.slider("364-Day Treasury Bill (Primary) Yield (%)", min_value=1.00, max_value=100.00, value=6.35)
fbil = st.slider("10-Year G-Sec Yield (FBIL) (%)", min_value=1.00, max_value=100.00, value=5.48)
crr = st.slider("Cash Reserve Ratio (%)", min_value=1.00, max_value=100.00, value=3.65)
slr = st.slider("Statutory Liquidity Ratio (%)", min_value=1.00, max_value=100.00, value=19.56)
prr = st.slider("Policy Repo Rate (%)", min_value=1.00, max_value=100.00, value=5.45)
sdf = st.slider("Standing Deposit Facility (SDF) Rate (%)", min_value=1.00, max_value=100.00, value=4.56)
predict = st.button('Predict')
st.markdown("# Prediction")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Creating DataFrame
df = pd.DataFrame(data={'Period': [period], 'Forward Premia of US$ 1-month (%)': [fp_1month],
                        'Forward Premia of US$ 3-month (%)': [fp_3month], 'Forward Premia of US$ 6-month (%)': [fp_6month],
                        'Reverse Repo Rate (%)': [repo_rate], 'Marginal Standing Facility (MSF) Rate (%)': [msf_rate],
                        'Bank Rate (%)': [bank_rate], 'Base Rate (%)': [base_rate],
                        '91-Day Treasury Bill (Primary) Yield (%)': [tby_91], '182-Day Treasury Bill (Primary) Yield (%)': [tby_182],
                        '364-Day Treasury Bill (Primary) Yield (%)': [tby_364], '10-Year G-Sec Yield (FBIL) (%)': [fbil],
                        'Cash Reserve Ratio (%)': [crr], 'Statutory Liquidity Ratio (%)': [slr],
                        'Policy Repo Rate (%)': [prr], 'Standing Deposit Facility (SDF) Rate (%)': [sdf]})

# Creating a function to predict foreign reserves
def PredictRserves(dataframe: object, model_: object, scaler_: object):
    df_copy = dataframe.copy()
    df_copy['Period'] = pd.to_datetime(df['Period'])
    df_copy['Period'] = df_copy['Period'].map(datetime.datetime.toordinal)

    # preprocessing dataframe for model
    df_copy_scaled = scaler_.transform(df_copy)

    # Predict the future foreign reserves
    pred_ = model_.predict(df_copy_scaled)

    return pred_[0], dataframe['Period'][0]

# If Predict button is pressed generate result and print
if predict:
    result_, period_ = PredictRserves(dataframe=df, model_=model, scaler_=scaler)

    # Write prediction in Streamlit
    st.write(f"Foreign Reserves on {period_} will be ${result_} Million.")



