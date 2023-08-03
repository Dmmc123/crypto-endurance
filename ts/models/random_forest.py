from ts.models.base import BaseNextDayPriceRegressor
from typing import Any, Self
from pathlib import Path

import xgboost as xgb
import pandas as pd
import numpy as np


class RandomForestRegressor(BaseNextDayPriceRegressor):
    name: str = "random_forest"

    def _fit(self, x: Any, y: Any, params: dict) -> Self:
        dmx = xgb.DMatrix(data=x, label=y)
        self.model = xgb.train(params=params, dtrain=dmx, num_boost_round=1)
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
        model.load_model(f"{weights_dir}/random_forest.json")
        return cls(model=model)


if __name__ == "__main__":
    random_forest = RandomForestRegressor()
    df = pd.read_csv("datasets/BTC-USD.csv")
    wandb_config = {
        "log_run": False,
        "proj_name": "crypto-random-forest"
    }
    params = {
        'colsample_bynode': 0.8,
        'learning_rate': 1,
        'max_depth': 5,
        'num_parallel_tree': 100,
        'objective': 'reg:squarederror',
        'subsample': 0.8,
        'tree_method': 'gpu_hist'
    }
    x, y = random_forest.df_to_samples(df=df, target_col="Close", include_targets=True)
    random_forest.fit(x=x, y=y, params=params, wandb_config=wandb_config)
    random_forest.save("weights")
    rf2 = RandomForestRegressor.from_weights("weights")
    print(rf2)
