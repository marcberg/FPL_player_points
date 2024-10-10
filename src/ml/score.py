import pandas as pd 
import joblib
import os

def score_data():

    print('Score data')
    # Get the best model
    best_algo = pd.read_csv('artifacts/ml_results/best_performing_algorithm.csv').iloc[0,0]
    print(f"Scoring new data with {best_algo}.")

    model_path = 'artifacts/ml_results/{0}/'.format(best_algo)
    model = joblib.load(os.path.join(model_path, 'model.pkl'))

    # get the scoring-data
    score = pd.read_csv('artifacts/data/score.csv')
    
    # predict
    predictions = model.predict(score)

    # save and print
    score_output = score[['next_kickoff_time', 'playername', 'team', 'next_game_home', 'next_opponent', 'player_position', 'value', 'starts', 'starts_rolling_avg']]
    score_output['predicted_points_next_game'] = predictions
    score_output = score_output.sort_values('predicted_points_next_game', ascending=False).reset_index(drop=True)

    score_output.to_csv('artifacts/ml_results/predictions.csv', index=False)

    print("\n Top 10 prediction: \n\n")
    print(score_output.head(10))