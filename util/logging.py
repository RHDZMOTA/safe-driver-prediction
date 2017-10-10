from conf import settings
import pandas as pd
import datetime
import os


def stringify_results(res, model_conf, model_key):
    res_string = """

    -------------------------------
    {datetime}

    SELECTED MODEL: {model}

    Link Function (y-transform): {link}
    Other Transformations (x-transform): 
{transf}

    PRAMETERS:
{params}

    TRAIN DATA
    > % correct : {correct_train} 
    > sklearn-accuracy: {accuracy_train} 
    > normalized-gini : {ngini_train} 
    > roc-auc: {auc_train}

    TEST DATA
    > % correct : {correct_test} 
    > sklearn-accuracy: {accuracy_test} 
    > normalized-gini : {ngini_test} 
    > roc-auc: {auc_test}

    VALIDATION (2017)
    > % correct : {correct_valid} 
    > sklearn-accuracy: {accuracy_valid} 
    > normalized-gini : {ngini_valid} 
    > roc-auc: {auc_valid}


    """
    # Stringify Parameters
    params = ""
    for param in model_conf:
        params += "\t> " + param + ": " + str(model_conf[param]) + "\n"
    # Stringify x-transforms
    other_transf = ""
    tranf_functions = res.datasets.transformations
    for transf in tranf_functions:
        other_transf += "\t> " + transf + ": " + str(tranf_functions[transf]) + "\n"
    # AUC
    _, _, auc_train = res.sklearn_roc_curve("train", proba=True)
    _, _, auc_test = res.sklearn_roc_curve("test", proba=True)
    _, _, auc_valid = res.sklearn_roc_curve("valid", proba=True)
    # Format Content
    now = datetime.datetime.now()
    content = res_string.format(
        datetime=now.strftime("%Y/%m/%d %H:%M:%S"),
        model=model_key,
        link=res.datasets.link,
        transf=other_transf,
        params=params,
        correct_train=res.correct_values("train", proba=False),
        accuracy_train=res.sklearn_accuracy_score("train", proba=False),
        ngini_train=res.normalized_gini_score("train", proba=True),
        auc_train=auc_train,
        correct_test=res.correct_values("test", proba=False),
        accuracy_test=res.sklearn_accuracy_score("test", proba=False),
        ngini_test=res.normalized_gini_score("test", proba=True),
        auc_test=auc_test,
        correct_valid=res.correct_values("valid", proba=False),
        accuracy_valid=res.sklearn_accuracy_score("valid", proba=False),
        ngini_valid=res.normalized_gini_score("valid", proba=True),
        auc_valid=auc_valid
    )
    filename = now.strftime("%Y-%m-%d-%H-%M-%S") + "-" + model_key + ".txt"
    return filename, content


def logg_result(res, model_conf, model_key):
    filename, content = stringify_results(res, model_conf, model_key)
    print(content)
    with open(os.path.join(settings.PROJECT_DIR, "logs", filename), "w") as file:
        file.write(content)
