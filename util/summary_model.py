from util.ModifiedData import max_day
import pandas as pd


def predict_summary(day, mlp_regression=None, linear_regression=None, support_vector_regressor=None):
    result = {}
    last_day_of_confirmed, last_day_of_deaths = max_day()

    if mlp_regression is not None:
        spain_confirmed, spain_deaths = mlp_regression

        result['Multi-Layer Perception'] = [
            spain_confirmed.best_predect([[last_day_of_confirmed + day]])[0],
            spain_deaths.best_predect([[last_day_of_deaths + day]])[0]
        ]

    if linear_regression is not None:
        spain_confirmed, spain_deaths = linear_regression[0]
        world_confirmed, world_deaths = linear_regression[1]

        result['Linear Regression'] = [
            int(((spain_confirmed.predict([[last_day_of_confirmed + day]]) -
                  world_confirmed.predict([[last_day_of_confirmed + day]])) / 2).item()),
            int(((spain_deaths.predict([[last_day_of_confirmed + day]]) -
                  world_deaths.predict([[last_day_of_deaths + day]])) / 2).item())
        ]
    if support_vector_regressor is not None:
        spain_confirmed, spain_deaths = support_vector_regressor

        result['Support Vector Regression'] = [
            int(spain_confirmed.predictValues([[last_day_of_confirmed + day]]).item()),
            int(spain_deaths.predictValues([[last_day_of_deaths + day]]).item())
        ]

    return pd.DataFrame(result, ['confirmed', 'deaths'])
