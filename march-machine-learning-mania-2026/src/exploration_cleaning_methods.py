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
    
    merged_lf: pl. LazyFrame = regular_lf.join(
        seed_lf,
        on=["Season", "TeamID"], 
        how="left"
    )

    return merged_lf