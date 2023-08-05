from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from typing import Self, Union, Any
from abc import ABC, abstractmethod
from pydantic import BaseModel
from itertools import product
from random import shuffle

import pandas as pd
import wandb
import yaml
import tqdm


class BaseNextDayPriceRegressor(BaseModel, ABC):
    look_back_days: int = 20
    test_days: int = 7
    model: Any = None

    @abstractmethod
    def _fit(self, x: Any, y: Any, params: dict) -> Self:
        pass

    @abstractmethod
    def save(self, weights_dir: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def from_weights(cls, weights_dir: str) -> Self:
        pass

    def fit(self,
            x: Any,
            y: Any,
            params: dict,
            wandb_config: dict[str, str] = None,
            x_dev: Any = None,
            y_dev: Any = None) -> Self:
        if wandb_config is None:
            wandb_config = {}
        run = None
        if wandb_config.get("log_run", False):
            if "proj_name" not in wandb_config:
                raise ValueError("`wandb_config` should have `proj_name` specified if run is to be logged")
            run = wandb.init(project=wandb_config["proj_name"], config=params)
            run_tag = wandb_config.get("run_tag", None)
            if run_tag is not None:
                run.tags += (wandb_config["run_tag"],)
        self._fit(x=x, y=y, params=params)
        if run is not None:
            metrics = self.evaluate(x=x, y_true=y, prefix="train")
            if x_dev is not None and y_dev is not None:
                metrics |= self.evaluate(x=x_dev, y_true=y_dev, prefix="dev")
            run.log(metrics)
            run.finish()
        return self

    @abstractmethod
    def predict(self, x: Any) -> Any:
        pass

    def sample_grid_search(self,
                           df: pd.DataFrame,
                           target_col: str,
                           grid_config_path: str,
                           wandb_config: dict[str, str] = None,
                           n_samples: int = 100,
                           reserve_test_samples: bool = True) -> None:
        if reserve_test_samples:
            df = df.iloc[:-self.test_days]
        x, y = self.df_to_samples(df=df, target_col=target_col, include_targets=True)
        x_train, x_dev, y_train, y_dev = train_test_split(x, y, shuffle=False, test_size=self.test_days)
        with open(grid_config_path, "r") as f:
            grid = yaml.safe_load(f)
        hps = list(product(*grid.values()))
        shuffle(hps)
        for param_vals in tqdm.tqdm(hps[:n_samples], desc="Grid searching"):
            params = dict(zip(grid.keys(), param_vals))
            self.fit(
                x=x_train,
                y=y_train,
                params=params,
                wandb_config=wandb_config,
                x_dev=x_dev,
                y_dev=y_dev
            )

    def evaluate(self, x: Any, y_true: Any, prefix: str = None) -> dict[str, float]:
        prefix = f"{prefix}_" if prefix is not None else ""
        y_hat = self.predict(x)
        return {
            f"{prefix}rmse": mean_squared_error(y_true=y_true, y_pred=y_hat, squared=False),
            f"{prefix}mape": mean_absolute_percentage_error(y_true=y_true, y_pred=y_hat)
        }

    def test(self,
             df: pd.DataFrame,
             target_col: str,
             config_path: str,
             wandb_config: dict[str, str] = None) -> dict[str, float]:
        with open(config_path, "r") as f:
            params = yaml.safe_load(f)
        metrics = {
            "rmse": [],
            "mape": []
        }
        for i in tqdm.tqdm(range(self.test_days), "Training models for testing"):
            if i == 0:
                cur_df = df.copy()
            else:
                cur_df = df.iloc[:-i]
            x, y = self.df_to_samples(df=cur_df, target_col=target_col, include_targets=True)
            x_train, x_test_1s, y_train, y_test_1s = train_test_split(x, y, shuffle=False, test_size=1)
            self.fit(
                x=x_train,
                y=y_train,
                params=params,
                wandb_config=wandb_config,
                x_dev=x_test_1s,
                y_dev=y_test_1s
            )
            cur_metrics = self.evaluate(x=x_test_1s, y_true=y_test_1s)
            metrics["rmse"].append(cur_metrics["rmse"])
            metrics["mape"].append(cur_metrics["mape"])
        metrics["rmse"] = sum(metrics["rmse"]) / len(metrics["rmse"])
        metrics["mape"] = sum(metrics["mape"]) / len(metrics["mape"])
        return metrics

    @abstractmethod
    def df_to_samples(self, df: pd.DataFrame, target_col: str, include_targets: bool) -> tuple[Any, Any]:
        pass
