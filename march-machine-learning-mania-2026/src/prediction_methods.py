import polars as pl
import numpy as np
import xgboost as xgb

def create_carthesian_matchup_grid(train_data_lf: pl.LazyFrame) -> pl.DataFrame:
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
            train_data_lf
            .filter(pl.col("Season" == current_year))
            .select(pl.col("WTeamID").alias("TeamID"))
            .vstack(train_data_lf
                    .filter(pl.col("Season") == current_year)
                    .select(pl.col("LTeamID").alias("TeamID"))
            )
            .unique().collect()
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

