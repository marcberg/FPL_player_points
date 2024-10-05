from src.data.read import create_temp_view
from src.util.data_calculations import calculate_rolling_avg

def team_data(spark):
    """
    Retrieves and processes team data, computing team-level metrics such as wins, draws, losses, and upcoming matches.
    Rolling averages are computed over a specified window size for team performance indicators like goals scored, conceded, 
    and match results.

    Args:
        spark (SparkSession): The Spark session to use for executing SQL queries.

    Returns:
        DataFrame: A Spark DataFrame with team-level statistics and rolling averages.
    """
    
    df_get_game_list = create_temp_view("get_game_list", spark)

    team_data = spark.sql(
    """
        with team_data_setup as (
            select 
                *
                , case 
                    when team_scored > team_conceded then 1
                    else 0 
                    end as win
                , case 
                    when team_scored = team_conceded then 1
                    else 0 
                    end as draw
                , case 
                    when team_scored < team_conceded then 1
                    else 0 
                    end as loss
                , lead(opponent, 1) over(partition by team, season_start_year order by kickoff_time) as next_opponent
                , lead(home, 1) over(partition by team, season_start_year order by kickoff_time) as next_game_home
                , lead(kickoff_time, 1) over(partition by team, season_start_year order by kickoff_time) as next_kickoff_time
            from (
                select 
                    home as team
                    , away as opponent
                    , team_h_score as team_scored
                    , team_a_score as team_conceded
                    , kickoff as kickoff_time
                    , season_start_year
                    , 1 as home
                from 
                    get_game_list

                union all

                select 
                    away as team
                    , home as opponent
                    , team_a_score as team_scored
                    , team_h_score as team_conceded
                    , kickoff as kickoff_time
                    , season_start_year
                    , 0 as home
                from 
                    get_game_list
            ) a
        )

        select 
            * 
        from 
            team_data_setup 
        order by 
            kickoff_time
    """
    )

    team_calculations = calculate_rolling_avg(team_data, 
                                            partition_by_cols=["team", "season_start_year"], 
                                            order_by_cols=["kickoff_time"], 
                                            window_size=4,
                                            cols=["team_scored","team_conceded","home","win","draw","loss"])
    
    return team_calculations