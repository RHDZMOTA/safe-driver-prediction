# Porto Seguro Safe Driver Prediction

[Porto Seguro](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction) kaggle competence. 

[ add content ]

## Usage

1. Create a ```.env``` file.
    * ```cp conf/.env.example conf/.env```
2. Create virtual environment and install dependencies. 
    * ```virtualenv venv```
    * ```source activate venv```
    * ```pip install -r requirements.txt```
3. Project boostrap.
    * ```python setup.py```
4. Run as following:
    * ```python main.py --model rf --type classification```

Parameters to run in console (default value in **bold**):
* ```--model``` : {**rf**, gb, mlp, lr} _machine learning model_  
* ```--type```  : {**classification**, regression} _type of model_
* ```--roc```   : {train, test, validate, **none**} _plot roc on data_
* ```--submit```: {true, **false**} _create a submit file at data/submit_


Model description (classification):
    
* **rf**: random forest.
* **gb**: gradient boosting.
* **mlp**: multilayer perceptron.
* **lr**: logistic regression [TODO]

    
[ add content ]

## Model Parameters

Model parameters can be changed by editing the file: ```model_setup.json```.

The ```main.py``` script will read this file as a python dictionary and
apply this configuration to the selected model. 

The file is structured as following: 
```
{
  
  "regression": {
    "random-forest": {
      "n-estimators": 50,
      "n-jobs": -1
    },
    "boosted-trees": {
      "max-depth": 10
    },
    "mlp": {
      "hidden-layers": "(100, 50, 20, 5)",
      "activation-function": "relu",
      "max-iter": 1000
    }
  },


  "classification": {

    "random-forest": {
      "n-estimators": 50,
      "n-jobs": -1
    },

    "gradient-boosting": {
      "loss": "deviance",
      "n-estimators": 100,
      "max-depth": 5
    },

    "multilayer-perceptron": {
      "hidden-layers": "(100, 50, 20, 5)",
      "activation-function": "relu",
      "max-iter": 1000
    },

    "logistic-regression": {
      "penalty": "l2",
      "max_iter": 1000,
      "n_jobs": -1
    }

  }
}
```

Note that the hidden layer of the multi-layer perceptron are 
indicated as a python tuple between "". The main python script 
uses the function `eval()` on this parameter to pass the 
configuration to the scikit-learn library.


Possible values for classification algorithms:

* [random-forest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.score)
    * **n-estimators**: integer value.
    * **n-jobs**: number of cores (-1 to use all available).
* [gradient-boosting](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
    * **loss**: [TODO]
    * **n-estimators**: integer value.
    * **max-depth**: integer value.
* [multilayer-perceptron](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
    * **hidden-layers**: string representation of a tuple.
    * **activation-function**:  {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
    * **max-iter**: integer value.

[ add content ]

## Variable description

[ add content ]

See file ```data-exploration.ipynb``` for a more detailed analysis.

## TODO

[ add content ]