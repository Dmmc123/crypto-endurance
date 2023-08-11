from ts.models.base import BaseNextDayPriceRegressor
from sklearn.linear_model import Ridge
from typing import Self, Any
from pathlib import Path

import pandas as pd
import numpy as np
import joblib


class LinearRegressor(BaseNextDayPriceRegressor):
    name: str = "linear_model"

    def _fit(self, x: Any, y: Any, params: dict) -> Self:
        self.model = Ridge(**params).fit(x, y)
        return self

    def predict(self, x: Any) -> Any:
        return self.model.predict(x)

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
        joblib.dump(self.model, f"{weights_dir}/{self.name}.pkl")

    @classmethod
    def from_weights(cls, weights_dir: str) -> Self:
        model = joblib.load(f"{weights_dir}/linear_model.pkl")
        return cls(model=model)
