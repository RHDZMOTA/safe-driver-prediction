import pandas as pd
import datetime as dt

from conf import settings
from optparse import OptionParser
from util.ml_models import model_map
from util.logging import logg_result
from util.data_preparation import get_dataset, DataSets



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
    model_key = model_map[model_type][kwargs.model].get("key")
    model_conf = settings.ModelConf.get_parameters(model_type, model_key)
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
