"""This project is my first attempt in solving a real world problem using
machine learning. Here I made use of Random Forests, Gradient Tree Boosting,
Neural Networks and Ensembles to participate on the "Bikesharing Demand"
(http://www.kaggle.com/c/bike-sharing-demand) competition on Kaggle.
"""

# Author: Rafael Aguiar <rfna@cin.ufpe.br>
# License: MIT


from data_handling import load, output
from ml import RegressionModel, Ensemble, NNFromR
from optimize import cv, feature_worth, tune_parameters
from sklearn import ensemble


if __name__ == '__main__':
    train, test, train_, test_, scaler = load(modify_dfs=False, load_=True)

    # Specify variables to be considered on each model.
    independent_vars = [
        'weekday', 'hour', 'year', 'season', 'holiday',
        'workingday', 'weather', 'temp', 'atemp', 'humidity',
        'windspeed'
    ]
    nn_independent_vars = filter(
        # dummy variables use underscore
        lambda x: x.split("_")[0] not in [
            'casual', 'registered', 'count', 'month'  # removes dependent vars
        ] and x.split(".")[0] not in [
            'casual', 'registered', 'count'  # removes grouped vars
        ],
        train_.columns.tolist()
    )
    # A model is composed by a regressor and its variables.
    models = {
        "RandomForestRegressor": RegressionModel(
            ensemble.RandomForestRegressor(
                n_estimators=1000,
                min_samples_split=11,
                n_jobs=-1,
                oob_score=False,
                random_state=0
            ),
            {
                'casual': independent_vars+[
                    'registered.by.month',
                    'casual.by.hour',
                    'registered.by.hour'
                ],
                'registered': independent_vars+[
                    'registered.by.month',
                    'casual.by.hour',
                    'registered.by.hour'
                ]
            }
        ),
        "GradientBoostingRegressor": RegressionModel(
            ensemble.GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=0
            ),
            {
                'casual': independent_vars+[
                    'registered.by.month',
                    'casual.by.hour',
                    'registered.by.hour'
                ],
                'registered': independent_vars+[
                    'registered.by.month',
                    'casual.by.hour',
                    'registered.by.hour'
                ]
            }
        ),
        "NeuralNetwork": RegressionModel(
            NNFromR(
                hidden="10",
                threshold=0.01,
                stepmax=1e+06,
                learningrate=0.001,
                algorithm="rprop+",
                lifesign="none"
            ),
            {
                'casual': nn_independent_vars,
                'registered': nn_independent_vars
            },
            scaler
        ),
        "NeuralNetwork_w_Momentum": RegressionModel(
            NNFromR(
                n_neurons="c(ncol(train),5,1)",
                learning_rate_global=0.001,
                momentum_global=0.001,
                error_criterium="LMS",
                hidden_layer="sigmoid",
                output_layer="sigmoid",
                method="ADAPTgdwm"
            ),
            {
                'casual': nn_independent_vars,
                'registered': nn_independent_vars
            },
            scaler
        )
    }

    # > Example: parameter optimization
    # model = models['NeuralNetwork_w_Momentum']
    # e_neurons, min_neurons = tune_parameters(
    #     model,
    #     'n_neurons',
    #     range(5, 11)
    # )

    # # > Example: test if a feature is worth including in a model
    # feature_worth(models['GradientBoostingRegressor'], train)

    # # > Example: Compare validation error between models
    # for model_key in models:
    #     print "> "+model_key
    #     model = models[model_key]
    #     if "NeuralNetwork" in model_key:
    #         error = cv(train_, 5, model)
    #     else:
    #         error = cv(train, 5, model)
    #     print model_key, error

    del models['NeuralNetwork']
    del models['NeuralNetwork_w_Momentum']
    ensemble = Ensemble(
        models=models,
        combiner=ensemble.GradientBoostingRegressor(
            n_estimators=100,
            random_state=0
        )
    )
    ensemble.fit(train)
    d = dict(
        zip(
            models['GradientBoostingRegressor'].variables['casual'],
            ensemble.feature_importances_
        )
    )
    prediction = ensemble.predict(test)
    output('12.0th.csv', prediction, test)
