import streamlit as st
import pandas as pd
import requests
import yaml
import json

from ts.models import (
    GradientBoostingRegressor,
    LinearRegressor,
    LSTMRegressor,
    PerceptronRegressor,
    RandomForestRegressor
)


@st.cache_data
def get_dataset() -> pd.DataFrame:
    df = pd.read_csv("datasets/BTC-USD.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


@st.cache_data
def next_day_bitcoin_usd_price(day):
    api_url = f"https://min-api.cryptocompare.com/data/pricehistorical?fsym=BTC&tsyms=USD&ts={int(prev_day.timestamp())}"
    response = requests.get(api_url)
    try:
        data = response.json()
        return int(data["BTC"]["USD"])
    except:
        raise ValueError(f"cannot fetch bitcoin data for day {day}")


def baseline_expander(name: str, rmse: float, mape: float) -> None:
    info = {
        "gbr": {
            "wandb": "crypto-gradient-boosting",
            "expander": "Gradient Boosting Regressor",
            "config_path": "ts/configs/gradient_boosting/best.yaml",
            "model_class": GradientBoostingRegressor,
            "model_dir": "weights"
        },
        "ar": {
            "wandb": "crypto-linear-regressor",
            "expander": "Linear Regressor",
            "config_path": "ts/configs/linear_model/best.yaml",
            "model_class": LinearRegressor,
            "model_dir": "weights"
        },
        "lstm": {
            "wandb": "crypto-lstm-regressor",
            "expander": "LSTM Regressor",
            "config_path": "ts/configs/lstm/best.yaml",
            "model_class": LSTMRegressor,
            "model_dir": "weights"
        },
        "mlp": {
            "wandb": "crypto-mlp-regressor",
            "expander": "Multi-layer Perceptron",
            "config_path": "ts/configs/mlp/best.yaml",
            "model_class": PerceptronRegressor,
            "model_dir": "weights"
        },
        "rf": {
            "wandb": "crypto-random-forest",
            "expander": "Random Forest Regressor",
            "config_path": "ts/configs/random_forest/best.yaml",
            "model_class": RandomForestRegressor,
            "model_dir": "weights"
        }
    }
    with st.expander(info[name]['expander']):
        # link to wandb logs
        wandb_link = f"https://wandb.ai/dmmc123/{info[name]['wandb']}"
        st.markdown(f"[Logs]({wandb_link})")

        pred_col, metric_col = st.columns(2)
        with pred_col:
            model = info[name]["model_class"].from_weights(
                weights_dir=info[name]["model_dir"]
            )
            x, _ = model.df_to_samples(
                df=df,
                target_col="Close",
                include_targets=False
            )
            y_hat = model.predict(x)
            last_pred = int(y_hat[-1])
            st.metric(
                label="Predicted BTC Price",
                value=f"{last_pred} $",
                delta=f"{last_pred - prev_day_price} $"
            )
        with metric_col:
            metric_df = pd.DataFrame({"Value": [rmse, mape]}, index=['RMSE', 'MAPE'])
            st.markdown("##### Testing Metrics")
            st.write(metric_df)
        # output parameters
        with open(info[name]["config_path"], "r") as f:
            yaml_content = yaml.safe_load(f)
        json_content = json.dumps(yaml_content, indent=4)
        st.markdown("##### Model Parameters")
        st.code(json_content, language="json")


st.set_page_config(
    page_title='Baseline Methods',
    page_icon="extras/btc.png",
)
df = get_dataset()

st.subheader("Menu")

df_converted = convert_df(df)
st.download_button(
    label="Dataset BTC-USD",
    data=df_converted,
    file_name='BTC-USD.csv',
    mime='text/csv',
)

github_repo_url = "https://github.com/Dmmc123/crypto-endurance"
st.markdown(f"[GitHub]({github_repo_url})")

prev_day = df["Date"].iloc[-1]
prev_day_price = int(df["Close"].values[-1])
next_day_price = next_day_bitcoin_usd_price(day=prev_day)
st.metric(label="BTC Price", value=f"{next_day_price} $", delta=f"{next_day_price - prev_day_price} $")

st.subheader("Method Descriptions")

baseline_expander(
    name="gbr",
    rmse=1349.9598,
    mape=0.0444
)
baseline_expander(
    name="ar",
    rmse=1868.2915,
    mape=0.0614
)
baseline_expander(
    name="lstm",
    rmse=305.0756,
    mape=0.0101
)
baseline_expander(
    name="mlp",
    rmse=779.3188,
    mape=0.0258
)
baseline_expander(
    name="rf",
    rmse=2331.7525,
    mape=0.0766
)













