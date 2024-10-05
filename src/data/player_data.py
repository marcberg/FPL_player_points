import pyspark.sql.functions as f

from src.data.read import create_temp_view
from src.util.data_calculations import calculate_rolling_avg

def player_data(spark):
    """
    Retrieves and processes detailed player data, calculating various metrics such as total points, goals, assists, 
    and other game-related statistics per minute. The function also computes rolling averages over a specified window size 
    for various performance indicators like goals scored, assists, and minutes played.

    Args:
        spark (SparkSession): The Spark session to use for executing SQL queries.

    Returns:
        DataFrame: A Spark DataFrame with the calculated player statistics and rolling averages.
    """
        
    df_get_player_details = create_temp_view("get_player_details", spark)
    df_get_player_info = create_temp_view("get_player_info", spark)
    df_get_game_list = create_temp_view("get_game_list", spark)

    player_data = spark.sql(
    """
        with players_data_setup as (
            select
                pd.season_start_year
                , case 
                    when pd.was_home = 'true' then gl.home 
                    else gl.away 
                    end as team
                , case 
                    when pd.was_home = 'true' then gl.away 
                    else gl.home 
                    end as opponent
                , pd.kickoff_time
                , pd.round

                , pi.code as player_id
                , pd.playername
                , case
                    when pi.element_type = 1 then "Goalkeeper"
                    when pi.element_type = 2 then "Defender"
                    when pi.element_type = 3 then "Midfielder"
                    else "Striker"
                    end as player_position
                , lead(coalesce(pd.total_points, 0)) over(partition by pd.season_start_year, pi.code order by pd.kickoff_time) as target
                , pd.minutes
                , pd.total_points
                , pd.total_points / pd.minutes as total_points_per_minute
                , pd.goals_scored
                , pd.goals_scored / pd.minutes as goals_scored_per_minute
                , pd.assists
                , pd.assists / pd.minutes as assists_per_minute
                , pd.clean_sheets
                , pd.goals_conceded
                , pd.goals_conceded / pd.minutes as goals_conceded_per_minute
                , pd.own_goals
                , pd.own_goals / pd.minutes as own_goals_per_minute
                , pd.penalties_saved
                , pd.penalties_saved / pd.minutes as penalties_saved_per_minute
                , pd.saves
                , pd.saves / pd.minutes as saves_per_minute
                , pd.penalties_missed
                , pd.penalties_missed / pd.minutes as penalties_missed_per_minute
                , pd.yellow_cards
                , pd.yellow_cards / pd.minutes as yellow_cards_per_minute
                , pd.red_cards
                , pd.red_cards / pd.minutes as red_cards_per_minute
                , pd.bonus
                , pd.bonus / pd.minutes as bonus_per_minute
                , pd.bps
                , pd.bps / pd.minutes as bps_per_minute
                , pd.influence
                , pd.influence / pd.minutes as influence_per_minute
                , pd.creativity
                , pd.creativity / pd.minutes as creativity_per_minute
                , pd.threat
                , pd.threat / pd.minutes as threat_per_minute
                , pd.ict_index
                , pd.ict_index / pd.minutes as ict_index_per_minute
                , pd.starts
                , pd.expected_goals
                , pd.expected_goals / pd.minutes as expected_goals_per_minute
                , pd.expected_assists
                , pd.expected_assists / pd.minutes as expected_assists_per_minute
                , pd.expected_goal_involvements
                , pd.expected_goal_involvements / pd.minutes as expected_goal_involvements_per_minute
                , pd.expected_goals_conceded
                , pd.expected_goals_conceded / pd.minutes as expected_goals_conceded_per_minute
                , pd.value
                , pd.transfers_balance
                , pd.selected
                , coalesce(
                    pd.selected / (lag(pd.selected, 3) over(partition by pd.season_start_year, pi.code order by pd.kickoff_time)),
                    pd.selected / (lag(pd.selected, 2) over(partition by pd.season_start_year, pi.code order by pd.kickoff_time)),
                    pd.selected / (lag(pd.selected, 1) over(partition by pd.season_start_year, pi.code order by pd.kickoff_time)),
                    1
                ) as selected_index_change4
                , pd.transfers_in
                , pd.transfers_out
                , pd.transfers_balance / selected as pct_transfer_balance

            from 
                get_player_details pd 
                    left join 
                        get_game_list gl 
                            on pd.season_start_year = gl.season_start_year 
                                and pd.fixture = gl.id
                                and pd.opponent_team = (case when pd.was_home = 'true' then gl.team_a else gl.team_h end)
                    left join 
                        get_player_info pi 
                            on pd.element = pi.id
                                and pd.season_start_year = pi.season_start_year
        )

        select 
            * 
            , case 
                when season_start_year = max(season_start_year) over() and target is null and round > 1 then "score"
                when target is null then "delete"
                else "train" 
                end as data
        from 
            players_data_setup 
        order by 
            player_id
            , kickoff_time desc

    """
    )

    player_calculations = calculate_rolling_avg(player_data, 
                                            partition_by_cols=["player_id", "season_start_year"], 
                                            order_by_cols=["kickoff_time"], 
                                            window_size=4,
                                            cols=["total_points",
                                                    "minutes",
                                                    "goals_scored",
                                                    "assists",
                                                    "clean_sheets",
                                                    "goals_conceded",
                                                    "own_goals",
                                                    "penalties_saved",
                                                    "saves",
                                                    "penalties_missed",
                                                    "yellow_cards",
                                                    "red_cards",
                                                    "bonus",
                                                    "bps",
                                                    "influence",
                                                    "creativity",
                                                    "threat",
                                                    "ict_index",
                                                    "starts",
                                                    "expected_goals",
                                                    "expected_assists",
                                                    "expected_goal_involvements",
                                                    "expected_goals_conceded",
                                                    "total_points_per_minute",
                                                    "goals_scored_per_minute",
                                                    "assists_per_minute",
                                                    "goals_conceded_per_minute",
                                                    "own_goals_per_minute",
                                                    "penalties_saved_per_minute",
                                                    "saves_per_minute",
                                                    "penalties_missed_per_minute",
                                                    "yellow_cards_per_minute",
                                                    "red_cards_per_minute",
                                                    "bonus_per_minute",
                                                    "bps_per_minute",
                                                    "influence_per_minute",
                                                    "creativity_per_minute",
                                                    "threat_per_minute",
                                                    "ict_index_per_minute",
                                                    "expected_goals_per_minute",
                                                    "expected_assists_per_minute",
                                                    "expected_goal_involvements_per_minute",
                                                    "expected_goals_conceded_per_minute",
                                                    "selected_index_change4",
                                                    "pct_transfer_balance",
                                                    ]
                                                )
    
    # filter out players which doesn't play and effect the models. These players are not of interest in this project.
    player_calculations = player_calculations.filter((f.col("minutes_rolling_avg") > 0) | ((f.col("value") >= 45) & (f.col("selected") >= 10000)) | (f.col("total_points_rolling_avg") != 0))

    return player_calculations
