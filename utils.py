import operator
import pandas as pd
import numpy as np


def feature_importances(model_key, model):
    """Creates a csv with feature_importances of a model to ease
    the analysis process.
    """
    for var in model.variables:
        df = pd.DataFrame(
            dict(
                zip(
                    model.variables[var],
                    model.fitted_models[var].feature_importances_
                )
            ),
            index=[1]
        )
        df['var'] = var
        pd.DataFrame.to_csv(
            df,
            'data/fti_%s_%s.csv' % (model_key, var)
        )


def build_R_parameters(param):
    """Builds a R parameter string from a Python dictionary.
    """
    return ",".join(
        [
            key+"="+str(value) if type(value) != str or 'c(' in str(value)
            else key+"='"+value+"'"
            for (key, value) in param.items()
        ]
    ).replace('_', '.')
