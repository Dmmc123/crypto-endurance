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
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        lstm_out_last = lstm_out[:, -1, :]
        output = self.fc(lstm_out_last)
        return output


class LSTMRegressor(BaseNextDayPriceRegressor):
    name: str = "lstm"
    n_indicators: int = 15

    def _fit(self, x: Any, y: Any, params: dict) -> Self:
        self.model = TorchLSTM(
            input_size=self.n_indicators,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"]
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params["lr"])
        criterion = nn.MSELoss()
        for _ in range(params["epochs"]):
            preds = self.model(x)
            optimizer.zero_grad()
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, x: Any) -> Any:
        return self.model(x).view(-1).detach().numpy()

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
        targets = torch.tensor(targets).view(-1, 1).float()
        return features, targets

    def save(self, weights_dir: str) -> None:
        pass

    @classmethod
    def from_weights(cls, weights_dir: str) -> Self:
        pass


if __name__ == "__main__":
    lstm_reg = LSTMRegressor()
    df = pd.read_csv("datasets/BTC-USD.csv")
    x, y = lstm_reg.df_to_samples(df=df, target_col="Close", include_targets=True)
    wandb_config = {
        "log_run": True,
        "proj_name": "crypto-lstm-regressor"
    }
    lstm_reg.sample_grid_search(
        df=df,
        target_col="Close",
        grid_config_path="ts/configs/lstm/grid.yaml",
        wandb_config=wandb_config,
        n_samples=50
    )
