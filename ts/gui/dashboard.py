import streamlit as st
from datetime import date
import pandas as pd
import xgboost as xgb
import numpy as np


def get_dataset() -> pd.DataFrame:
    return pd.read_csv("datasets/BTC-USD.csv")


def get_last_delta(dataset: pd.DataFrame, predicted_price: float) -> float:
    return predicted_price - dataset["Close"].values[-1]


def predict_price(dataset: pd.DataFrame, look_back: int = 20) -> float:
    features = []
    feature_dataset = dataset.drop(columns="Date", axis=1)
    features.append(feature_dataset.values[-look_back:])
    features = np.array(features)
    features_flattened = features.reshape(features.shape[0], -1)
    dm = xgb.DMatrix(features_flattened)
    model = xgb.core.Booster()
    model.load_model("weights/xgb_next_day_price.json")
    pred = model.predict(dm)
    return pred[0]


def predict_trend(dataset: pd.DataFrame, look_back: int = 20) -> float:
    features = []
    feature_dataset = dataset.drop(columns="Date", axis=1)
    features.append(feature_dataset.values[-look_back:])
    features = np.array(features)
    features_flattened = features.reshape(features.shape[0], -1)
    dm = xgb.DMatrix(features_flattened)
    model = xgb.core.Booster()
    model.load_model("weights/xgb_next_day_trend.json")
    pred = model.predict(dm)
    return pred[0]


df = get_dataset()


st.set_page_config(
    page_title='Baseline Dashboard',
    page_icon="extras/btc.png"
)

st.header("Asset Dashboard")

today = date.today()
st.write("Today's date:", today)

st.subheader("BTC-USD")

next_day_price = predict_price(dataset=df)
price_delta = get_last_delta(dataset=df, predicted_price=next_day_price)
trend = -1

price_col, trend_col = st.columns(2)
with price_col:
    st.metric(label="Next day price prediction", value=next_day_price, delta=price_delta)
with trend_col:
    color = "red" if trend < 0.5 else "green"
    text = "Positive" if trend > 0.5 else "Negative"
    st.write(f"<span style='color:{color};'>{text} trend</span>", unsafe_allow_html=True)



