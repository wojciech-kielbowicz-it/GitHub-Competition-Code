import polars as pl
import polars.selectors as cs

def create_team_season_profile(detailed_results_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
        Transforms game-level LazyFrame into a seasonal summary by doubling 
        the observations (one for each team per game), calculating 
        pace-adjusted efficiency metrics, and aggregating results 
        by Season and TeamID.
    Args:
        detailed_results_lf (pl.LazyFrame): The raw detailed results 
                                            from regular season or tourney.
    Returns:
        pl.LazyFrame: A summarized table with one row per team per season 
                    containing mean efficiency ratings and win ratios.
    """
    winners_lf: pl.LazyFrame = detailed_results_lf.select(
        pl.col(["Season", "DayNum"]), 
        cs.starts_with("W").exclude("WLoc").name.map(
            lambda x: x.replace("WTeam" , "Team", 1).replace("W", "Team", 1)
            ), 
        cs.starts_with("L").name.map(
            lambda x: x.replace("LTeam", "Opponent", 1).replace("L", "Opponent", 1)
            ),
        pl.lit(1).alias("WinRatio"), 
        pl.when(pl.col("WLoc") == "H").then(1)
            .when(pl.col("WLoc") == "A").then(-1)
            .otherwise(0).alias("Location")
        )
    
    losers_lf: pl.LazyFrame = detailed_results_lf.select(
        pl.col(["Season", "DayNum"]), 
        cs.starts_with("W").exclude("WLoc").name.map(
            lambda x: x.replace("WTeam" , "Opponent", 1).replace("W", "Opponent", 1)
            ),
        cs.starts_with("L").name.map(
            lambda x: x.replace("LTeam", "Team", 1).replace("L", "Team", 1)
            ), 
        pl.lit(0).alias("WinRatio"), 
        pl.when(pl.col("WLoc") == "H").then(-1)
            .when(pl.col("WLoc") == "A").then(1)
            .otherwise(0).alias("Location")
    )

    win_los_combined_lf: pl.LazyFrame = pl.concat([winners_lf, losers_lf])

    team_pos: pl.Expr = (
        pl.col("TeamFGA") + (0.475 * pl.col("TeamFTA")) - pl.col("TeamOR") + pl.col("TeamTO")
    )
    oppo_pos: pl.Expr = (
        pl.col("OpponentFGA") + (0.475 * pl.col("OpponentFTA")) - pl.col("OpponentOR") + pl.col("OpponentTO")
    )

    win_los_combined_lf = win_los_combined_lf.with_columns(
        team_pos.alias("TeamPOS"), 
        oppo_pos.alias("OpponentPOS")
        )
    
    off_efficiency: pl.Expr = (
        (pl.col("TeamScore") / pl.col("TeamPOS")) * 100
    )

    def_efficiency: pl.Expr = (
        ((pl.col("OpponentScore")) / pl.col("OpponentPOS")) * 100
    )

    team_efg: pl.Expr = (
        (pl.col("TeamFGM") + (0.5 * pl.col("TeamFGM3"))) / pl.col("TeamFGA")
    )

    oppo_efg: pl.Expr = (
        (pl.col("OpponentFGM") + (0.5 * pl.col("OpponentFGM3"))) / pl.col("OpponentFGA")
    )
    

    turnover_rate: pl.Expr = (
        pl.col("TeamTO") / pl.col("TeamPOS")
    )
    
    win_los_combined_lf = win_los_combined_lf.with_columns(
        off_efficiency.alias("OffEfficiency"), 
        def_efficiency.alias("DefEfficiency"), 
        team_efg.alias("TeamEFG"),  # Team Effective Field Goal Percentage
        oppo_efg.alias("OpponentEFG"),  # Opponent Effective Field Goal Percentage
        turnover_rate.alias("TurnoverRate")
    )

    col_to_exclude: list[str] = [
        "DayNum", "OpponentID", "TeamLoc", 
        "TeamFGM", "TeamFGA", "TeamFGM3", 
        "TeamFGA3", "TeamFTM", "TeamFTA",
        "TeamOR", "TeamDR", "TeamAst", 
        "TeamTO", "TeamStl", "TeamBlk", "TeamPF",
        "OpponentFGM", "OpponentFGA", "OpponentFGM3", 
        "OpponentFGA3", "OpponentFTM", "OpponentFTA",
        "OpponentOR", "OpponentDR", "OpponentAst", 
        "OpponentTO", "OpponentStl", "OpponentBlk", "OpponentPF"
    ]


    grouped_lf: pl.LazyFrame = (
        win_los_combined_lf.
        filter(
            (pl.col("Season") >= 2015) & (pl.col("Season") != 2020)
        )
        .sort(["Season", "TeamID"])
        .group_by(["Season", "TeamID"])
        .agg(cs.numeric().exclude(col_to_exclude).mean())
    )
    
    return grouped_lf

def clean_seed(seed_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Filters tournament seeds by year and extracts the numerical seed value from alphanumeric strings.

    Args:
        seed_lf (pl.LazyFrame): LazyFrame containing tournament seeds (e.g., 'W01', 'Z16a').

    Returns:
        pl.LazyFrame: Filtered data from 2015 onwards (excluding 2020) with seeds converted to 8-bit integers.
    """
    cleaned_seed: pl.LazyFrame = (
        seed_lf
            .filter(
                (pl.col("Season") >= 2015) & (pl.col("Season") != 2020)
            )
            .with_columns(
                pl.col("Seed")
                .str.replace(r"^[A-Z]*(\d{2})[a,b]*$", r"$1")
                .cast(pl.Int8)
            )
    )

    return cleaned_seed

def merge_seed_with_regular(seed_lf: pl.LazyFrame, regular_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Merges tournament seed information into regular season team statistics.

    Args:
        seed_lf (pl.LazyFrame): LazyFrame containing 'Season', 'TeamID', and their tournament seed.
        regular_lf (pl.LazyFrame): LazyFrame containing aggregated regular season metrics per team and season.

    Returns:
        pl.LazyFrame: The regular season statistics enriched with seed data where available via a left join.
    """
    merged_lf: pl. LazyFrame = regular_lf.join(
        seed_lf,
        on=["Season", "TeamID"], 
        how="left"
    )

    return merged_lf

def prepare_tourney_matchups(tourney_matchups_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Filters and reshapes tournament results into a balanced binary classification dataset.

    Args:
        tourney_matchups_lf (pl.LazyFrame): Raw tournament match data containing 'WTeamID' and 'LTeamID'.

    Returns:
        pl.LazyFrame: A dataset starting from 2015 (excluding 2020) where each game is represented twice 
        to balance the 'Target' (1 for Team A win, 0 for Team A loss) and IDs are normalized to 'ATeamID' and 'BTeamID'.
    """
    matchups_lf: pl.LazyFrame = (
        tourney_matchups_lf
            .select(
                pl.col(["Season", "WTeamID", "LTeamID", "DayNum"])
            )
            .filter(
                (pl.col("Season") >= 2015) & (pl.col("Season") != 2020)
            )
    )

    winner_matchups_lf: pl.LazyFrame = (
        matchups_lf
        .clone()
        .rename({
            "WTeamID": "ATeamID", 
            "LTeamID": "BTeamID"
        })
    )

    winner_matchups_lf = winner_matchups_lf.with_columns(
        cs.numeric().cast(pl.Int16), 
        pl.lit(1).alias("Target").cast(pl.Int8)
    )

    beaten_matchups_lf: pl.LazyFrame = (
        matchups_lf
        .clone()
        .rename({
            "WTeamID": "BTeamID", 
            "LTeamID": "ATeamID"
        })
    )
    
    beaten_matchups_lf = (
        beaten_matchups_lf
            .select([
                "Season", 
                "ATeamID", 
                "BTeamID", 
                "DayNum"
            ])
            .with_columns(
                cs.numeric().cast(pl.Int16), 
                pl.lit(0).alias("Target").cast(pl.Int8)
            )
    )

    matchups_lf: pl.LazyFrame = pl.concat([winner_matchups_lf, beaten_matchups_lf])

    return matchups_lf

def finalize_training_data(matchups_lf: pl.LazyFrame, golden_table_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Enriches match data with seasonal statistics and calculates the differences between Team A and Team B.

    Args:
        matchups_lf (pl.LazyFrame): LazyFrame containing match data with 'Season', 'ATeamID', and 'BTeamID'.
        golden_table_lf (pl.LazyFrame): LazyFrame containing seasonal metrics per team, keyed by 'Season' and 'TeamID'.

    Returns:
        pl.LazyFrame: A dataset where original team-specific stats are replaced by their differences 
        (suffixed with '_Diff'), while retaining core match identifiers.
    """
    final_train_lf: pl.LazyFrame = (
        matchups_lf
        .join(
            golden_table_lf, 
            left_on=["Season", "ATeamID"], 
            right_on=["Season", "TeamID"],
            how="left"
        )
        .rename({
            col: f"{col}_A" for col in golden_table_lf.collect_schema().names()
            if col not in ["Season", "TeamID"]
        })
    )

    final_train_lf = (
        final_train_lf
        .join(
            golden_table_lf, 
            left_on=["Season", "BTeamID"], 
            right_on=["Season", "TeamID"], 
            how="left"
        )
        .rename({
            col: f"{col}_B" for col in golden_table_lf.collect_schema().names() 
            if col not in ["Season", "TeamID"]
        })
    )

    cols_a: list[str] = [c for c in final_train_lf.collect_schema().names() if c.endswith("_A")]

    cols_expr: list[pl.Expr] = [
        (pl.col(c) - pl.col(c.replace("_A", "_B")))
        .alias(c.replace("_A", "_Diff"))
        for c in cols_a
    ]

    final_train_lf = (
        final_train_lf
        .with_columns(cols_expr)
        .drop(cs.ends_with("_A", "_B"))
    )

    return final_train_lf