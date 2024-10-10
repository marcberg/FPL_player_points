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

        , form_home_away as (
            select 
                team
                , kickoff_time
                , avg(win) over(partition by team, next_game_home, season_start_year order by kickoff_time rows between 1 preceding and current row) as win_rate_home_away_2
                , avg(draw) over(partition by team, next_game_home, season_start_year order by kickoff_time rows between 1 preceding and current row) as draw_rate_home_away_2
                , avg(loss) over(partition by team, next_game_home, season_start_year order by kickoff_time rows between 1 preceding and current row) as loss_rate_home_away_2
                , avg(team_scored) over(partition by team, next_game_home, season_start_year order by kickoff_time rows between 1 preceding and current row) as avg_team_scored_home_away_2
                , avg(team_conceded) over(partition by team, next_game_home, season_start_year order by kickoff_time rows between 1 preceding and current row) as avg_team_conceded_home_away_2
            from 
                team_data_setup
        )

        , table (
            with sorted_team_games as (
                select
                    *
                    , row_number() over(partition by season_start_year, team order by kickoff_time) as row_num
                from 
                    team_data_setup
            )

            , points_and_games as (
                select
                    *
                    , case 
                        when win = 1 then 3 
                        when draw = 1 then 1 
                        else 0 
                    end as points_from_game
                    , 1 as game
                from 
                    sorted_team_games
            )

            , team_stats as (
                select
                    *
                    , sum(points_from_game) over(partition by season_start_year, team order by kickoff_time) as team_points
                    , sum(game) over(partition by season_start_year, team order by kickoff_time) as total_games
                from 
                    points_and_games
            )

            , games_left as (
                select
                    season_start_year
                    , kickoff_time
                    , team
                    , total_games
                    , points_from_game
                    , team_points
                    , 38 - total_games as games_remaining
                from 
                    team_stats
            )

            , next_dates as (
                select distinct 
                    season_start_year
                    , kickoff_time as next_kickoff_time
                from 
                    games_left
                order by 
                    next_kickoff_time
            )

            , merged_data as (
                select
                    a.*
                    , b.next_kickoff_time
                from 
                    games_left a
                left join 
                    next_dates b 
                        on a.season_start_year = b.season_start_year
                where 1 = 1 
                    and a.kickoff_time < b.next_kickoff_time
            )

            , ranked_data as (
                select
                    *
                    , row_number() over(partition by team, next_kickoff_time order by kickoff_time desc) as rn
                from 
                    merged_data
            )

            , latest_games as (
                select 
                    * 
                from 
                    ranked_data
                where 1 = 1 
                    and rn = 1
            )

            , final_ranking as (
                select
                    *
                    , rank() over(partition by season_start_year, next_kickoff_time order by team_points desc, team) as position
                from 
                    latest_games
            )

            , points_comparison as (
                select
                    *
                    , coalesce(team_points - lead(team_points) over(partition by next_kickoff_time order by team_points desc), 0) as points_above
                    , coalesce(team_points - lag(team_points) over(partition by next_kickoff_time order by team_points desc), 0) as points_below
                    , coalesce(games_remaining - lead(games_remaining) over(partition by next_kickoff_time order by team_points desc), 0) as games_diff_above
                    , coalesce(games_remaining - lag(games_remaining) over(partition by next_kickoff_time order by team_points desc), 0) as games_diff_below
                from 
                    final_ranking
            )

            , league_standings as (
                select
                    next_kickoff_time
                    , max(case when position = 1 then team_points else 0 end) as win_points
                    , max(case when position = 4 then team_points else 0 end) as cl_points
                    , max(case when position = 7 then team_points else 0 end) as euro_points
                    , max(case when position = 18 then team_points else 0 end) as regulation_points
                from 
                    points_comparison
                group by 
                    next_kickoff_time
            )

            select
                a.*
                , b.win_points
                , b.cl_points
                , b.euro_points
                , b.regulation_points
                , a.team_points - b.win_points as points_to_win
                , a.team_points - b.cl_points as points_to_cl
                , a.team_points - b.euro_points as points_to_euro
                , a.team_points - b.regulation_points as points_to_regulation
            from 
                points_comparison a
                    left join 
                        league_standings b 
                            on a.next_kickoff_time = b.next_kickoff_time
        )


        select 
            tds.* 

            , fha.win_rate_home_away_2
            , fha.draw_rate_home_away_2
            , fha.loss_rate_home_away_2
            , fha.avg_team_scored_home_away_2
            , fha.avg_team_conceded_home_away_2

            , t_team.position
            , t_team.team_points
            , t_team.points_above as points_to_team_above
            , t_team.points_below as points_to_team_below
            , t_team.games_remaining
            , t_team.games_diff_above as games_left_diff_above
            , t_team.games_diff_below as games_left_diff_below
            , t_team.points_to_win
            , t_team.points_to_cl
            , t_team.points_to_euro
            , t_team.points_to_regulation

            , t_opponent.position as opponent_position
            , t_opponent.team_points as opponent_team_points
            , t_opponent.points_above as opponent_points_to_team_above
            , t_opponent.points_below as opponent_points_to_team_below
            , t_opponent.games_remaining as opponent_games_remaining
            , t_opponent.games_diff_above as opponent_games_left_diff_above
            , t_opponent.games_diff_below as opponent_games_left_diff_below
            , t_opponent.points_to_win as opponent_points_to_win
            , t_opponent.points_to_cl as opponent_points_to_cl
            , t_opponent.points_to_euro as opponent_points_to_euro
            , t_opponent.points_to_regulation as opponent_points_to_regulation
        from 
            team_data_setup tds
                left join 
                    form_home_away fha
                        on tds.team = fha.team 
                            and tds.kickoff_time = fha.kickoff_time
                left join 
                    table t_team
                        on tds.season_start_year = t_team.season_start_year 
                            and tds.kickoff_time = t_team.next_kickoff_time
                            and tds.team = t_team.team
                left join 
                    table t_opponent
                        on tds.season_start_year = t_opponent.season_start_year 
                            and tds.kickoff_time = t_opponent.next_kickoff_time
                            and tds.opponent = t_opponent.team
    """
    )
    
    team_calculations = calculate_rolling_avg(team_data, 
                                            partition_by_cols=["team", "season_start_year"], 
                                            order_by_cols=["kickoff_time"], 
                                            window_size=4,
                                            cols=["team_scored",
                                                  "team_conceded",
                                                  "home",
                                                  "win",
                                                  "draw",
                                                  "loss"])

    return team_calculations