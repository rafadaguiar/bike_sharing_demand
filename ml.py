import copy

import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
import utils


class RegressionModel:
    """A regression model is composed by an estimator, its variables
    and a scaling transformation (if the dataset is scaled).
    Methods fit and predict are the same as usual, but extended to support
    multiple dependent variables, inverse scaling transformation and inverse
    log transformation.
    """
    def __init__(self, estimator, variables, scaler=None):
        self.estimator = estimator
        self.variables = variables
        self.scaler = scaler
        self.fitted_models = dict()

    def fit(self, df):
        for var in self.variables:
            estimator = copy.copy(self.estimator)
            model = estimator.fit(
                df[self.variables[var]],
                df[var]
            )
            self.fitted_models[var] = model

    def predict(self, df):
        count = 0
        for var in self.variables:
            model = self.fitted_models[var]
            prediction = model.predict(df[self.variables[var]])
            if self.scaler:
                prediction = self.scaler[var].inverse_transform(prediction)
            count += np.exp(prediction) - 1
        count = np.around(count)
        count[count < 0] = 0
        return count


class NNFromR:
    """This class acts as a wrapper handling fit and predict operations on Robjects
    (more specifically, neural networks).
    """
    class RReturn:
        """The NNFromR fit method returns an Robject, this class purpose
        is to avoid converting it to a pandas df before calling predict.
        In this way, R can get the data, fit, predit and return the result
        as a pandas df.
        """
        def __init__(self, Robject, algorithm):
            self.fitted_model = Robject
            self.algorithm = algorithm

        def predict(self, indep_vars):
            ro.globalenv['test'] = pandas2ri.py2ri(indep_vars)
            ro.globalenv['fit'] = self.fitted_model
            if self.algorithm == "rprop+":
                return pandas2ri.ri2py(
                    ro.r("compute(fit,test)$net.result")
                )
            elif self.algorithm == "ADAPTgdwm":
                return pandas2ri.ri2py(
                    ro.r("sim(fit$net, test)")
                )

    def __init__(self, **kwargs):
        self.param = kwargs
        pandas2ri.activate()
        ro.r("library(neuralnet)")
        ro.r("library(AMORE)")

    def fit(self, indep_vars, dep_var):
        ro.globalenv['train'] = pandas2ri.py2ri(indep_vars)
        ro.globalenv[dep_var.name] = pandas2ri.py2ri(dep_var)

        # Builds the parameters string
        param = utils.build_R_parameters(self.param)

        # In order to support neural networks from different packages it
        # was necessary to wrap their respective methods for the "fit" concept
        if self.param.get("algorithm") == "rprop+":
            formula = dep_var.name+"~"+"+".join(indep_vars.columns.tolist())
            ro.r("formula <- as.formula(%s)" % formula)
            return self.RReturn(
                ro.r("neuralnet(formula, data=train,%s)" % param),
                self.param.get("algorithm")
            )
        elif self.param.get("method") == "ADAPTgdwm":
            ro.r("fit <- newff(%s)" % param)
            return self.RReturn(
                ro.r(
                    "fit <- train(fit, train, %s, error.criterium='LMS',\
                    report=TRUE, show.step=1000, n.shows=100)" % dep_var.name
                ),
                self.param.get("method")
            )


class Ensemble:
    """This class provides two ways of combining the results of different
    models: averaging the response or training a new estimator (the combiner
    ) to learn the weights for each model.
    """
    def __init__(self, models, combiner):
        self.models = models
        self.combiner = combiner

    def fit(self, df, param=None):
        if len(self.models) > 1:
            predictions = pd.DataFrame()
            for key in self.models:
                self.models[key].fit(df)
                predictions[key] = self.models[key].predict(df)
            self.combiner = self.combiner.fit(predictions, df['count'])
        else:
            self.combiner = self.combiner.fit(df)

    def predict(self, df):
        if len(self.models) > 1:
            predictions = pd.DataFrame()
            for key in self.models:
                prediction = np.around(self.models[key].predict(df))
                prediction[prediction < 0] = 0
                predictions[key] = prediction
        else:
            predictions = df
        return np.around(self.combiner.predict(predictions))

    @staticmethod
    def average_response(predictions):
        """This method provides a quick way of combining models by using the
        same weight for each.
        """
        prediction = np.around(sum(predictions)/len(predictions))
        prediction[prediction < 0] = 0
        return prediction
