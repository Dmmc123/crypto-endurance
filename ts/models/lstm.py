import tqdm
import yaml

from ts.models.base import BaseNextDayPriceRegressor
from typing import Any, Self
from pathlib import Path
from torch import nn

import pandas as pd
import numpy as np
import torch


class TorchLSTM(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        lstm_out_last = lstm_out[:, -1, :]
        output = self.fc(lstm_out_last)
        return output


class LSTMRegressor(BaseNextDayPriceRegressor):
    name: str = "lstm"
    n_indicators: int = 15
    hidden_size: int = None
    num_layers: int = None
    x_mean: Any = None
    x_std: Any = None
    y_mean: Any = None
    y_std: Any = None

    def _fit(self, x: Any, y: Any, params: dict) -> Self:
        # data standardization
        x_flat = x.view(-1, self.n_indicators)
        self.x_mean = x_flat.mean(dim=0)
        self.x_std = x_flat.std(dim=0)
        x = (x - self.x_mean) / self.x_std
        self.y_mean = y.mean()
        self.y_std = y.std()
        y = (y - self.y_mean) / self.y_std
        # training
        y = y.view(-1, 1)
        self.hidden_size = params["hidden_size"]
        self.num_layers = params["num_layers"]
        self.model = TorchLSTM(
            input_size=self.n_indicators,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        optimizer = torch.optim.Adam(self.model.lstm.parameters(), lr=params["lr"])
        criterion = nn.MSELoss()
        for _ in tqdm.tqdm(range(params["epochs"]), desc="Training LSTM"):
            optimizer.zero_grad()
            preds = self.model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, x: Any) -> Any:
        x = (x - self.x_mean) / self.x_std
        y = self.model(x).view(-1).detach()
        y = self.y_std * y + self.y_mean
        return y.numpy()

    def df_to_samples(self, df: pd.DataFrame, target_col: str, include_targets: bool) -> tuple[Any, Any]:
        df = df.drop(columns=["Date"])
        features = []
        targets = None
        for i in range(self.look_back_days, len(df)):
            features.append(df.values[i - self.look_back_days:i])
        features = np.array(features)
        features = torch.tensor(features).float()
        if not include_targets:
            return features, targets
        features = features[:-1]
        targets = df[target_col].values[self.look_back_days + 1:]
        targets = torch.tensor(targets).float()
        return features, targets

    def save(self, weights_dir: str) -> None:
        Path(weights_dir).mkdir(exist_ok=True, parents=True)
        model_params = {
            "input_size": self.n_indicators,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "scalers": {
                "x": {
                    "mean": self.x_mean.tolist(),
                    "std": self.x_std.tolist()
                },
                "y": {
                    "mean": self.y_mean.item(),
                    "std": self.y_std.item()
                }
            }
        }
        with open(f"{weights_dir}/lstm_params.yaml", "w") as f:
            yaml.dump(model_params, f)
        torch.save(self.model.state_dict(), f"{weights_dir}/{self.name}.pt")

    @classmethod
    def from_weights(cls, weights_dir: str) -> Self:
        with open(f"{weights_dir}/lstm_params.yaml", "r") as f:
            model_params = yaml.safe_load(f)
        model = TorchLSTM(
            input_size=model_params["input_size"],
            hidden_size=model_params["hidden_size"],
            num_layers=model_params["num_layers"]
        )
        model.load_state_dict(torch.load(f"{weights_dir}/lstm.pt"))
        return cls(
            model=model,
            n_indicators=model_params["input_size"],
            hidden_size=model_params["hidden_size"],
            num_layers=model_params["num_layers"],
            x_mean=torch.tensor(model_params["scalers"]["x"]["mean"]),
            x_std=torch.tensor(model_params["scalers"]["x"]["std"]),
            y_mean=torch.tensor(model_params["scalers"]["y"]["mean"]),
            y_std=torch.tensor(model_params["scalers"]["y"]["std"])
        )


if __name__ == "__main__":
    lstm = LSTMRegressor()
    df = pd.read_csv("datasets/BTC-USD.csv")
    x, y = lstm.df_to_samples(df=df, target_col="Close", include_targets=True)
    lstm.fit(
        x=x, y=y,
        params={
            "hidden_size": 20,
            "num_layers": 1,
            "epochs": 500,
            "lr": 0.05,
        },
        wandb_config={"log_run": False}
    )
    metrics = lstm.evaluate(x=x, y_true=y)
    y_pred = lstm.predict(x)
    print(y_pred[0], y_pred[-1])
    lstm.save(weights_dir="weights")
    lstm_2 = LSTMRegressor.from_weights(weights_dir="weights")
    y_pred_2 = lstm_2.predict(x)
    print(y_pred_2[0], y_pred_2[-1])

