import argparse
import json
import logging
import numpy
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, ARDRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pandas

from teams import all_possible_matches

REPEATS = 10

DIFFERENCE_SCORE = "difference_score"

models = [
    LinearRegression(),
    ARDRegression(),
    RandomForestRegressor(n_estimators=50, max_features='sqrt'),
    KNeighborsRegressor(n_neighbors=4),
    MLPRegressor(hidden_layer_sizes=(50,))
]


def scale(x_dataframe, y_dataframe):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    x_dataframe = pandas.DataFrame(min_max_scaler.fit_transform(x_dataframe), columns=x_dataframe.columns,
                                   index=x_dataframe.index)
    y_dataframe = pandas.DataFrame(min_max_scaler.fit_transform(y_dataframe), columns=y_dataframe.columns,
                                   index=y_dataframe.index)
    return x_dataframe, y_dataframe


def read_data_file(data_path):
    with open(data_path) as json_file:
        data_json = json.load(json_file)
    return data_json


def init_map(players):
    m = {}
    for p in players:
        m[p] = 0
    return m


def build_data(data_json, players):
    """
    We encode the following match:
    [
      {
        "teams": [
          [
            "levi",
            "andre",
            "max",
            "szimi"
          ],
          [
            "dom",
            "mayo",
            "ala",
            "oli"
          ]
        ],
        "score": [
          13,
          7
        ]
      }
    ]

    as
    levi    andre   max     szimi   dom     mayo    ala     oli     other_player_1      other_player_2      difference_score
    1       1       1       1       -1      -1      -1      -1      0                   0                   0.8 <-- this is normalised
    :param data_json:
    """
    df = pandas.DataFrame(columns=list(players) + [DIFFERENCE_SCORE])
    for match in data_json:
        score = match.get("score")
        diff_score = score[0] - score[1]
        teams = match.get("teams", [])
        m = init_map(players)
        for player in teams[0]:
            m[player] = 1
        for player in teams[1]:
            m[player] = -1
        m[DIFFERENCE_SCORE] = diff_score
        df = df.append(m, ignore_index=True)
    return df[players], df[[DIFFERENCE_SCORE]]


def build_players(data_json, new_players):
    players = set()
    for match in data_json:
        for team in match.get("teams", []):
            for player in team:
                players.add(player)
    return players.union(set(new_players))


def build_models(df_x, df_y):
    results = {}
    best_score = float("-Inf")
    best_model = ""
    for model in models:
        results_name = model.__class__.__name__
        logging.debug("Trying " + results_name)
        model_score_sum = 0
        for i in range(REPEATS):
            x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)
            # fit model on training dataset
            model.fit(x_train, y_train[DIFFERENCE_SCORE])
            score = r2_score(y_test[DIFFERENCE_SCORE], model.predict(x_test))
            model_score_sum += score
        avg_score = model_score_sum / REPEATS
        results[results_name] = [avg_score, model]

        if "linear" in results_name.lower():
            # If it's a linear model we can see how it ranks the players
            model.fit(df_x, df_y[DIFFERENCE_SCORE])
            print_coefficients(model, df_x)

        if avg_score >= best_score:
            model.fit(df_x, df_y[DIFFERENCE_SCORE])
            best_score = avg_score
            best_model = model

    logging.debug(results)
    logging.info("Picked {0} with score {1}".format(str(best_model.__class__.__name__), str(best_score)))
    return best_model


def print_team(team):
    print("\n".join(team))


def print_coefficients(best_model, x_dataframe):
    try:
        coefficients = pandas.concat([pandas.DataFrame(x_dataframe.columns, columns=["player"]),
                                      pandas.DataFrame(numpy.transpose(best_model.coef_), columns=["score"])],
                                     axis=1)
        print("Ordered Coefficients (best player on top)\n" + str(coefficients.sort_values("score", ascending=False)))
    except:
        print("The model doesn't have coefficients")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate a balanced football match')
    parser.add_argument('--data_path', type=str, help='Data json source path', default="data/matches.json")
    parser.add_argument('--verbose', help='Debug enabled', default=True, action="store_true")
    parser.add_argument('--players', nargs="+", help='Data json source path',
                        default=["max", "levi", "andre", "szimi", "dom", "tom",
                                 "ala", "luke", "rob", "john"])
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    logging.debug("Running with options:\n{}".format(args))

    data_json = read_data_file(args.data_path)
    players = build_players(data_json, args.players)

    x_dataframe, y_dataframe = build_data(data_json, players)
    x_dataframe, y_dataframe = scale(x_dataframe, y_dataframe)
    logging.debug(x_dataframe)
    logging.debug(y_dataframe)
    best_model = build_models(x_dataframe, y_dataframe)
    matches = all_possible_matches(args.players)
    best = float('Inf')
    best_match = None
    for m in matches:
        l = list(m)
        teams_as_lists = [list(l[0]), list(l[1])]
        data_point = [
            {
                "teams": teams_as_lists,
                "score": [0, 0]
            }
        ]
        df_x, df_y = build_data(data_point, players)
        score = best_model.predict(df_x)
        if abs(score) < abs(best):
            best = score
            best_match = teams_as_lists
    print(
        "Predicted normalised result (the closest to 0 the better; positive means the first team is predicted to win): " + str(
            best))
    print("Teams:")
    print("--light--")
    print_team(best_match[0])
    print("--dark--")
    print_team(best_match[1])


if __name__ == '__main__':
    main()
