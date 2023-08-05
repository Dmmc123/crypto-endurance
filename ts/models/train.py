from ts.models import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    LinearRegressor,
    MLPRegressor,
    LSTMRegressor
)
import pandas as pd
import argparse
import yaml


def main() -> None:
    """
    Example script:

    python -m ts.models.train \
       --csv-path datasets/BTC-USD.csv \
       --model-type gbr \
       --config-path ts/configs/gradient_boosting/best.yaml \
       --save-dir weights \
       --log-wandb \
       --wandb-proj crypto-gradient-boosting \
       --wandb-tag test
    """
    model_classes = {
        "gbr": GradientBoostingRegressor,
        "rf": RandomForestRegressor,
        "ar": LinearRegressor,
        "mlp": MLPRegressor,
        "lstm": LSTMRegressor
    }
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-path", required=True,
        help="Path to the `csv` file with training data"
    )
    parser.add_argument(
        "--target-col", required=False, default="Close",
        help="Column name to predict in training dataframe"
    )
    parser.add_argument(
        "--model-type", choices=model_classes.keys(), required=True,
        help="Type of model to train"
    )
    parser.add_argument(
        "--grid", action="store_true", required=False,
        help="Apply grid search for hyperparameter search"
    )
    parser.add_argument(
        "--config-path", required=True,
        help="Path to the `.yaml` config file of either grid (in case of `--grid`) of model itself"
    )
    parser.add_argument(
        "--save-dir", required=True,
        help="Path to save the model file"
    )
    parser.add_argument(
        "--log-wandb", action="store_true", default=False, required=False,
        help="Enable logging into W&B"
    )
    parser.add_argument(
        "--wandb-tag", required=False, default=None,
        help="Tag to add into W&B run"
    )
    parser.add_argument(
        "--wandb-proj", required=False,
        help="Project name in W&B"
    )
    args = parser.parse_args()
    wandb_config = {
        "log_run": args.log_wandb,
        "proj_name": args.wandb_proj,
        "run_tag": args.wandb_tag
    }
    df = pd.read_csv(args.csv_path)
    model = model_classes[args.model_type]()
    if args.grid:
        model.sample_grid_search(
            df=df,
            target_col=args.target_col,
            grid_config_path=args.config_path,
            wandb_config=wandb_config,
        )
    else:
        x, y = model.df_to_samples(
            df=df,
            target_col=args.target_col,
            include_targets=True
        )
        with open(args.config_path, "r") as f:
            params = yaml.safe_load(f)
        model.fit(
            x=x,
            y=y,
            params=params,
            wandb_config=wandb_config
        )
        model.save(weights_dir=args.save_dir)


if __name__ == "__main__":
    main()
