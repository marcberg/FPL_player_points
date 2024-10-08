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
        ),

        form_home_away as (
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
        ),

        table as (
            with team_points as (
                select
                    season_start_year
                    , team
                    , kickoff_time
                    , case 
                        when win = 1 then 3
                        when draw = 1 then 1
                        else 0
                        end as points_from_game
                    , 1 as game
                from 
                    team_data_setup
            ),

            cumulative_table as (
                select 
                    season_start_year
                    , team
                    , kickoff_time
                    , points_from_game
                    , sum(points_from_game) over(partition by season_start_year, team order by kickoff_time) as team_points
                    , sum(game) over(partition by season_start_year, team order by kickoff_time) as number_of_games
                    , 38 - SUM(game) over(partition by season_start_year, team order by kickoff_time) as games_left_season
                from 
                    team_points
            ),

            next_kickoff_dates as (
                select distinct
                    season_start_year
                    , kickoff_time as next_kickoff_time
                from 
                    cumulative_table
            ),

            joined_table as (
                select
                    ct.*
                    , nks.next_kickoff_time
                    , rank() over(partition by ct.team, nks.next_kickoff_time order by ct.kickoff_time desc) as rn
                from 
                    cumulative_table ct
                        left join next_kickoff_dates nks
                            on ct.season_start_year = nks.season_start_year
                where 1 = 1 
                    and ct.kickoff_time < nks.next_kickoff_time
            ),

            ranked_table as (
                select 
                    * 
                from 
                    joined_table 
                where 1 = 1 
                    and rn = 1
            ),

            positioned_table as (
                select
                    rt.*
                    , rank() over(partition by rt.season_start_year, rt.next_kickoff_time order by rt.team_points desc) as position
                from 
                    ranked_table rt
            ),

            diffs_table as (
                select
                    pt.*
                    , (team_points - lead(team_points) over(partition by next_kickoff_time order by team_points desc)) as points_to_team_above
                    , (team_points - lag(team_points) over(partition by next_kickoff_time order by team_points desc)) as points_to_team_below
                    , (games_left_season - lead(games_left_season) over(partition by next_kickoff_time order by games_left_season desc)) as games_left_diff_above
                    , (games_left_season - lag(games_left_season) over(partition by next_kickoff_time order by games_left_season desc)) as games_left_diff_below
                from 
                    positioned_table pt
            ),

            final_table as (
                select 
                    d.*
                    , win.win_points
                    , cl.cl_points
                    , euro.euro_points
                    , reg.regulation_points
                from 
                    diffs_table d
                        left join 
                            (select next_kickoff_time, team_points AS win_points from diffs_table where position = 1) win
                                on d.next_kickoff_time = win.next_kickoff_time
                        left join
                            (select next_kickoff_time, team_points AS cl_points from diffs_table where position = 4) cl
                                on d.next_kickoff_time = cl.next_kickoff_time
                        left join 
                            (select next_kickoff_time, team_points AS euro_points from diffs_table where position = 7) euro
                                on d.next_kickoff_time = euro.next_kickoff_time
                        left join 
                            (select next_kickoff_time, team_points AS regulation_points from diffs_table where position = 18) reg
                                on d.next_kickoff_time = reg.next_kickoff_time
            )

            select 
                season_start_year
                , team
                , next_kickoff_time
                , number_of_games
                , points_from_game as points_from_last_game
                , team_points
                , points_to_team_above
                , points_to_team_below
                , games_left_season
                , games_left_diff_above
                , games_left_diff_below
                , (team_points - win_points) as points_to_win
                , (team_points - cl_points) as points_to_cl
                , (team_points - euro_points) as points_to_euro
                , (team_points - regulation_points) as points_to_regulation
            from 
                final_table

        )

        select 
            tds.*
            , fha.win_rate_home_away_2
            , fha.draw_rate_home_away_2
            , fha.loss_rate_home_away_2
            , fha.avg_team_scored_home_away_2
            , fha.avg_team_conceded_home_away_2
            , t.number_of_games
            , t.points_from_last_game
            , t.team_points
            , t.points_to_team_above
            , t.points_to_team_below
            , t.games_left_season
            , t.games_left_diff_above
            , t.games_left_diff_below
            , t.points_to_win
            , t.points_to_cl
            , t.points_to_euro
            , t.points_to_regulation
        from 
            team_data_setup tds
                left join 
                    form_home_away fha
                        on tds.team = fha.team 
                            and tds.kickoff_time = fha.kickoff_time
                left join 
                    table t
                        on tds.season_start_year = t.season_start_year
                            and tds.team = t.team
                            and tds.next_kickoff_time = t.next_kickoff_time
        order by 
            kickoff_time
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