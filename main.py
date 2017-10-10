import pandas as pd
import datetime as dt

from conf import settings
from optparse import OptionParser
from util.logging import logg_result
from sklearn.ensemble import VotingClassifier
from util.regression import regressor_procedure
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from util.classification import classifier_procedure
from util.data_preparation import get_dataset, DataSets
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


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
        }
    }
}


def main():
    parser = OptionParser()
    parser.add_option("--type", type="string", default="classification", help="Select model.")
    parser.add_option("--model", type="string", default="rf", help="Select model.")
    parser.add_option("--roc", type="string", default="none", help="Plot.")
    parser.add_option("--submit", type="string", default="false", help="Plot.")
    kwargs, _ = parser.parse_args(args=None, values=None)

    # Data
    print("Reading data...")
    datasets, submit = get_dataset()
    submit_index = submit.index

    # Model
    print("Model procedure...")
    model_type = "regression" if "reg" in kwargs.type else "classification"
    model_conf = settings.ModelConf.get_parameters(model_type, model_map[model_type][kwargs.model].get("key"))
    res = model_map[model_type][kwargs.model].get("function")(model_conf, datasets, submit)

    # Logger
    print("Extracting results...\n\n")
    logg_result(res, model_conf, model_map[model_type][kwargs.model].get("key"))

    # Submission
    if "true" in kwargs.submit.lower():
        print("Generating submission file...")
        submit_results = res.get_submission_predictions(proba=True)
        submission_dataset = pd.DataFrame({
            "id": submit_index,
            "target": submit_results
        })
        submission_filename = settings.DataFilesConf.FileNames.submission_file.format(
            time=dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
            type=model_type,
            model=model_map[model_type][kwargs.model].get("key")
        )
        submission_dataset.to_csv(submission_filename, index=False)

    if not "none" in kwargs.roc.lower():
        print("Plotting...")
        res.plot_roc_curve(kwargs.roc.lower(), proba=True)

    print("Done.")

if __name__ == "__main__":
    main()