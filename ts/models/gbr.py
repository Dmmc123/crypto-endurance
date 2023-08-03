from ts.models.base import BaseNextDayPriceRegressor
from typing import Self, Any
from pathlib import Path

import xgboost as xgb
import pandas as pd
import numpy as np


class GradientBoostingRegressor(BaseNextDayPriceRegressor):
    name: str = "gbr"

    def _fit(self, x: Any, y: Any, params: dict) -> Self:
        dmx = xgb.DMatrix(data=x, label=y)
        new_params = {k: v for k, v in params.items() if k != "num_boost_round"}
        self.model = xgb.train(params=new_params, dtrain=dmx, num_boost_round=params["num_boost_round"])
        return self

    def predict(self, x: Any) -> Any:
        dmx = xgb.DMatrix(x)
        return self.model.predict(dmx)

    def df_to_samples(self, df: pd.DataFrame, target_col: str, include_targets: bool) -> tuple[Any, Any]:
        df = df.drop(columns=["Date"])
        features = []
        targets = None
        for i in range(self.look_back_days, len(df)):
            features.append(df.values[i - self.look_back_days:i])
        features = np.array(features)
        features = features.reshape(features.shape[0], -1)
        if not include_targets:
            return features, targets
        features = features[:-1]
        targets = df[target_col].values[self.look_back_days + 1:]
        return features, targets

    def save(self, weights_dir: str) -> None:
        Path(weights_dir).mkdir(exist_ok=True, parents=True)
        self.model.save_model(f"{weights_dir}/{self.name}.json")

    @classmethod
    def from_weights(cls, weights_dir: str) -> Self:
        model = xgb.core.Booster()
        model.load_model(f"{weights_dir}/gbr.json")
        return cls(model=model)


if __name__ == "__main__":
    gbr = GradientBoostingRegressor()
    df = pd.read_csv("datasets/BTC-USD.csv")
    wandb_config = {
        "log_run": True,
        "proj_name": "crypto-gradient-boosting"
    }
    gbr.sample_grid_search(
        df=df,
        target_col="Close",
        grid_config_path="ts/configs/gradient_boosting/grid.yaml",
        wandb_config=wandb_config,
        n_samples=50
    )