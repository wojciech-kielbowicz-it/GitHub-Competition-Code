import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb

def create_carthesian_matchup_grid(train_data_df: pl.DataFrame) -> pl.DataFrame:
    """
    Generates a Cartesian product of all possible unique team matchups for the current season.

    This method identifies all unique teams from the training data for the year 2026 
    and creates a non-redundant grid where each team is paired with every other team 
    exactly once (ordered by TeamID to avoid duplicate A-B and B-A pairs).

    Args:
        train_data_lf (pl.LazyFrame): A Polars LazyFrame containing historical match 
            data with columns 'Season', 'WTeamID', and 'LTeamID'.

    Returns:
        pl.DataFrame: A DataFrame containing columns 'ATeamID', 'BTeamID', and 'Season', 
            representing all possible future matchups.
    """
    def _extract_unique_teams() -> pl.DataFrame:
        """
        Extracts a unique list of all Team IDs that participated in the current season.

        Returns:
            pl.DataFrame: A single-column DataFrame ('TeamID') of unique teams found 
                in both winning and losing columns for the year 2026.
        """
        extracted_df: pl.DataFrame = (
            train_data_df
            .filter(pl.col("Season") == current_year)
            .select(pl.col("WTeamID").alias("TeamID"))
            .vstack(train_data_df
                    .filter(pl.col("Season") == current_year)
                    .select(pl.col("LTeamID").alias("TeamID"))
            )
            .unique()
        )
        return extracted_df
    
    current_year: int = 2026
    team_df: pl.DataFrame = _extract_unique_teams()

    grid_df: pl.DataFrame = (
        team_df.rename({"TeamID": "ATeamID"})
        .join(
            team_df.rename({"TeamID": "BTeamID"}), 
            how="cross"
        )
        .filter(
            pl.col("ATeamID") < pl.col("BTeamID")
        )
        .with_columns(
            pl.lit(current_year).alias("Season").cast(pl.Int16)
        )
    )
    return grid_df

def generate_model(train_data_df: pl.DataFrame) -> xgb.Booster:
    """
    Trains an XGBoost model using the provided Polars DataFrame.

    Args:
        train_data_df (pl.DataFrame): The input training data containing features and the 'Target' column.

    Returns:
        xgb.Booster: The trained XGBoost model object.
    """
    cols_to_drop: list[str] = ["Season", "ATeamID", "BTeamID", "DayNum", "Target", "Seed_Diff"]
    X_: np.ndarray = train_data_df.drop(cols_to_drop).to_numpy()
    y_: np.ndarray = train_data_df.select("Target").to_numpy().flatten()
    dtrain = xgb.DMatrix(X_, label=y_)

    parameters: dict[str, str] = {
        "objective": "binary:logistic", 
        "eval_metric": "rmse", 
        "max_depth": 4, 
        "learning_rate": 0.05, 
        "seed": 42
    }

    model: xgb.Booster = xgb.train(params=parameters, dtrain=dtrain, num_boost_round=100)
    return model

def predict_and_create_submission_data(
        model: xgb.Booster, feature_cols: list[str], infer_df: pl.DataFrame) -> pl.DataFrame:
    """
    Generates predictions using a trained XGBoost model and formats them into a competition submission schema.

    Args:
        model (xgb.Booster): The trained XGBoost model used for inference.
        feature_cols (list[str]): The list of column names used as features during training.
        infer_df (pl.DataFrame): The input DataFrame containing features and metadata (Season, ATeamID, BTeamID).

    Returns:
        pl.DataFrame: A DataFrame containing the 'ID' (formatted as Season_ATeamID_BTeamID) and 'Pred' columns.
    """
    dtest: np.ndarray = xgb.DMatrix(infer_df.select(feature_cols).to_numpy())
    infer_df: pl.DataFrame = infer_df.with_columns(pl.Series("Pred", model.predict(dtest)))

    sub_df: pl.DataFrame = infer_df.with_columns(
        pl.concat_str(
            [pl.col("Season").cast(pl.Utf8), 
             pl.col("ATeamID").cast(pl.Utf8), 
             pl.col("BTeamID").cast(pl.Utf8)], 
             separator="_").alias("ID")
        ).select(["ID", "Pred"])
    
    return sub_df
