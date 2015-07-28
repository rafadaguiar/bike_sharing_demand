from sklearn.metrics import mean_squared_error
from sklearn import cross_validation
import matplotlib.pyplot as plt
import numpy as np
import copy


def cv(data, folds, model):
    """Cross Validation using K-fold

    Parameters
    ----------
    data    data to be sliced on training and testing datasets
    k       number of folds on K-fold algorithm

    Returns
    -------
    errors  array of model errors
    models  dict incluinding casual and registered models of lowest errors
    """
    def rmsle(predicted, actual):
        # Root Mean Squared Logarithmic Error
        return mean_squared_error(
            np.log(predicted+1),
            np.log(actual+1)
        ) ** 0.5

    errors = []
    print " Cross Validation in progress..."
    kf = cross_validation.KFold(n=len(data.index), n_folds=folds)
    for i, (train_index, validation_index) in enumerate(kf):
        print ' F%d.' % i
        train = data.iloc[train_index]
        validation = data.iloc[validation_index]

        model.fit(train)
        prediction = model.predict(validation)
        actual = data.iloc[validation_index]['count'].as_matrix()
        error = rmsle(prediction, actual)
        errors.append(error)
    return np.mean(errors)


def cv_plot(errors):
    """Plots cross-validation error.
    """
    plt.plot(range(len(errors)), errors, 'o')
    plt.ylabel("Root Mean Squared Logarithmic Error")
    plt.xlabel("Model #")
    plt.show()


def tune_parameters(model, param_name, param_range):
    errors = []
    min_error = float('Inf')
    min_param = None
    for param_value in param_range:
        print ">>> Param: %s <<<" % param_name
        if param_name == 'n_neurons':
            param_value = "c(ncol(train),%d,1)" % param_value
        m = copy.copy(model)
        m.estimator.param[param_name] = param_value
        error = cv(data=train_, folds=5, model=model)
        errors.append(error)
        if error < min_error:
            min_param = param_value
    return errors, min_param


def feature_worth(model, train):
    """Eases the process of testing if a new feature improves the model.
    """
    error = cv(data=train, folds=5, model=model)
    print error
    model.fit(train)
    for var in model.variables:
        print var
        print model.fitted_models[var].feature_importances_
