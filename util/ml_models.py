
from sklearn.ensemble import VotingClassifier
from util.regression import regressor_procedure
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from util.classification import classifier_procedure
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier


def random_forest_regression(reg_conf, datasets):
    model = RandomForestRegressor(
        n_estimators=reg_conf["n-estimators"],
        n_jobs=reg_conf["n-jobs"]
    )
    res = regressor_procedure(model, datasets)
    return res


def random_forest_classification(class_conf, datasets, submission):
    model = RandomForestClassifier(
        n_estimators=class_conf["n-estimators"],
        max_depth=class_conf["max-depth"],
        min_samples_split=class_conf["min-samples-split"],
        max_features=class_conf["max-features"],
        n_jobs=class_conf["n-jobs"]
    )
    res = classifier_procedure(model, datasets, submission)
    return res


def mlp_classification(class_conf, datasets, submission):
    model = MLPClassifier(
        hidden_layer_sizes=eval(class_conf["hidden-layers"]),
        activation=class_conf["activation-function"],
        max_iter=class_conf["max-iter"]
    )
    res = classifier_procedure(model, datasets, submission)
    return res


def gradient_boosting_classification(class_conf, datasets, submission):
    model = GradientBoostingClassifier(
        n_estimators=class_conf["n-estimators"],
        learning_rate=class_conf["learning-rate"],
        max_depth=class_conf["max-depth"],
        min_samples_split=class_conf["min-samples-split"],
        max_features=class_conf["max-features"]
    )
    res = classifier_procedure(model, datasets, submission)
    return res


def xgb_classification(class_conf, datasets, submission):
    model = XGBClassifier(
        base_score=class_conf["base-score"],
        colsample_bylevel=class_conf["colsample-bylevel"],
        colsample_bytree=class_conf["colsample-bytree"],
        gamma=class_conf["gamma"],
        learning_rate=class_conf["learning-rate"],
        max_delta_step=class_conf["max-delta-step"],
        max_depth=class_conf["max-depth"],
        objective=class_conf["objective"],
        reg_alpha=class_conf["reg-alpha"],
        reg_lambda=class_conf["reg-lambda"],
        scale_pos_weight=class_conf["scale-pos-weight"],
        seed=class_conf["seed"],
        subsample=class_conf["subsample"]
    )
    res = classifier_procedure(model, datasets, submission)
    return res


model_map = {
    "regression": {
        "rf": {
            "key": "random-forest",
            "function": random_forest_regression
        }
    },
    "classification": {
        "rf": {
            "key": "random-forest",
            "function": random_forest_classification
        },
        "mlp": {
            "key": "multilayer-perceptron",
            "function": mlp_classification
        },
        "gb": {
            "key": "gradient-boosting",
            "function": gradient_boosting_classification
        },
        "xgb": {
            "key": "xg-boost",
            "function": xgb_classification
        }
    }
}
