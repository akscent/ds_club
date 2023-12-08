# Basic
import numpy as np
import pandas as pd

# Common Model Algorithms
from sklearn import (
    svm,
    tree,
    linear_model,
    neighbors,
    naive_bayes,
    ensemble,
    discriminant_analysis,
    gaussian_process,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier, DMatrix
from lightgbm import LGBMClassifier, train, Dataset
from sklearn.linear_model import LogisticRegression, Ridge

# Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from tqdm.notebook import tqdm

# Optimization Hyperparameters
import optuna
import torch
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedShuffleSplit

# Stacking
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

# Other

from pprint import pprint
from warnings import filterwarnings

# %matplotlib inline = show plots in Jupyter Notebook browser
# %matplotlib inline

filterwarnings("ignore", category=pd.errors.PerformanceWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=Warning)
filterwarnings("ignore", category=DeprecationWarning)

try:
    import cupy as cp

    gpu_available = True
except ImportError:
    gpu_available = False

n_jobs = -1 if gpu_available else 1

#########################################################################
#########################################################################


class EvalModels:
    """
    Class for evaluate binary classification models
    """

    def __init__(
        self, data, y_col, test_size=0.2, train_size=0.7, random_state=43, n_split=5
    ):
        self.data = data
        self.y_col = y_col
        self.x_cols = list(data.drop(columns=y_col).columns)
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.n_split = n_split

    def simple_eval(self, cv_splits=5):
        """
        data - pd.dataFrame
        y_col - list with target col name
        cv_split - split for cross validation
        test_size, train_size  - % of splitting
        random state - rand state for models
        """

        object_columns = (
            self.data[self.x_cols].select_dtypes(include=["object"]).columns
        )
        cat_idx = [self.data.columns.get_loc(col) for col in object_columns.to_list()]
        if not object_columns.empty:
            self.data[object_columns] = self.data[object_columns].astype("category")

        # list with models. Стоит продолжить расширять.
        MLA = []
        if not object_columns.empty:
            MLA.append(
                XGBClassifier(
                    verbose=-1,
                    enable_categorical=True,
                    n_estimators=5000,
                )
            )
        else:
            MLA.append(
                XGBClassifier(
                    verbose=-1,
                    enable_categorical=False,
                    n_estimators=5000,
                )
            )

        if not object_columns.empty:
            MLA.append(
                LGBMClassifier(
                    verbose=-1,
                    cat_feature=cat_idx,
                    n_estimators=5000,
                    learning_rate=0.1,
                    reg_alpha=0.5,
                    reg_lambda=0.3,
                    seed=42,
                )
            )
        else:
            MLA.append(
                LGBMClassifier(
                    verbose=-1,
                    n_estimators=5000,
                    learning_rate=0.1,
                    reg_alpha=0.5,
                    reg_lambda=0.3,
                    seed=42,
                )
            )

        if not object_columns.empty:
            MLA.append(
                CatBoostClassifier(
                    verbose=0,
                    cat_features=object_columns,
                    depth=4,
                    iterations=5000,
                    colsample_bylevel=0.098,
                    subsample=0.95,
                    l2_leaf_reg=9,
                    min_data_in_leaf=243,
                    max_bin=187,
                )
            )
        else:
            MLA.append(
                CatBoostClassifier(
                    verbose=0,
                    depth=4,
                    iterations=5000,
                    colsample_bylevel=0.098,
                    subsample=0.95,
                    l2_leaf_reg=9,
                    min_data_in_leaf=243,
                    max_bin=187,
                )
            )
        MLA_columns = [
            "MLA Name",
            "MLA Parameters",
            "MLA Train Accuracy Mean",
            "MLA Test Accuracy Mean",
            "MLA Test Accuracy 3*STD",
            "MLA Time",
        ]
        MLA_compare = pd.DataFrame(columns=MLA_columns)

        cv_split = model_selection.StratifiedShuffleSplit(
            n_splits=cv_splits,
            test_size=self.test_size,
            train_size=self.train_size,
            random_state=self.random_state,
        )

        row_index = 0
        for alg in tqdm(MLA, total=len(MLA), desc="ML first estimate"):
            MLA_name = alg.__class__.__name__
            MLA_compare.loc[row_index, "MLA Name"] = MLA_name
            MLA_compare.loc[row_index, "MLA Parameters"] = str(alg.get_params())

            cv_results = model_selection.cross_validate(
                alg,
                self.data[self.x_cols],
                self.data[self.y_col],
                cv=cv_split,
                return_train_score=True,
            )
            print(f"{MLA_name} estimated: {cv_results['test_score'].mean()}")
            MLA_compare.loc[row_index, "MLA Time"] = cv_results["fit_time"].mean()
            MLA_compare.loc[row_index, "MLA Train Accuracy Mean"] = cv_results[
                "train_score"
            ].mean()
            MLA_compare.loc[row_index, "MLA Test Accuracy Mean"] = cv_results[
                "test_score"
            ].mean()
            MLA_compare.loc[row_index, "MLA Test Accuracy 3*STD"] = (
                cv_results["test_score"].std() * 3
            )

            row_index += 1

        MLA_compare.sort_values(
            by=["MLA Test Accuracy Mean"], ascending=False, inplace=True
        )
        list_top_models = MLA_compare.head(3)["MLA Name"].tolist()

        return MLA_compare, list_top_models

    def fit_model(self, trial, X_train, y_train, X_valid, y_valid, model_name):
        if model_name == "CatBoostClassifier":
            params = {
                "iterations": trial.suggest_int("iterations", 500, 800),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.01),
                "auto_class_weights": trial.suggest_categorical(
                    "auto_class_weights", ["SqrtBalanced", "Balanced", "None"]
                ),
                "depth": trial.suggest_int("depth", 3, 6),
                "loss_function": "Logloss",
                "use_best_model": True,
                "nan_mode": trial.suggest_categorical("nan_mode", ["Min", "Max"]),
            }
            train_dataset = Pool(
                data=X_train,
                label=y_train,
            )
            eval_dataset = Pool(
                data=X_valid,
                label=y_valid,
            )
            clf = CatBoostClassifier(verbose=0, random_seed=41, **params)
            clf.fit(train_dataset, eval_set=eval_dataset, early_stopping_rounds=300)
        elif model_name == "LGBMClassifier":
            params = {
                "num_iterations": trial.suggest_int("num_iterations", 300, 500),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.5),
                "num_leaves": trial.suggest_int("num_leaves", 10, 20),
                "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.5),
                "boosting_type": trial.suggest_categorical(
                    "boosting_type", ["gbdt", "dart", "goss"]
                ),
                "objective": "binary",
                "metric": "binary_logloss",
                "max_bin": trial.suggest_int("max_bin", 255, 4095),
                "force_col_wise": True,
                "is_unbalance": True,
            }

            clf = LGBMClassifier(verbose=-1, random_seed=42)
            clf = clf.set_params(**params)
            clf.fit(X_train, y_train)
        elif model_name == "XGBClassifier":
            params = {
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 300, 500),
                "eta": trial.suggest_float("eta", 0.01, 0.1),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 5),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.5),
                "objective": "binary:logistic",
                "enable_categorical": True,
            }
            clf = (
                XGBClassifier(tree_method="hist", device="cuda")
                if gpu_available
                else XGBClassifier(verbose=-1, random_seed=43)
            )
            clf = clf.set_params(**params)
            clf.fit(X_train, y_train)
        else:
            raise ValueError("Invalid model_name")
        y_pred = clf.predict_proba(X_valid)[:, 1]
        return clf, y_pred

    def objective(self, trial, model_name, return_models=False):
        sss = StratifiedShuffleSplit(
            n_splits=self.n_split,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        scores, models = [], []

        for train_idx, valid_idx in sss.split(
            self.data[self.x_cols], self.data[self.y_col]
        ):
            X_train, X_valid = (
                self.data[self.x_cols].iloc[train_idx, :],
                self.data[self.x_cols].iloc[valid_idx, :],
            )
            y_train, y_valid = (
                self.data[self.y_col].iloc[train_idx],
                self.data[self.y_col].iloc[valid_idx],
            )

            X_train.columns = X_train.columns.str.replace(" ", "_")
            X_valid.columns = X_valid.columns.str.replace(" ", "_")

            clf, y_pred = self.fit_model(
                trial, X_train, y_train, X_valid, y_valid, model_name
            )
            roc_auc = roc_auc_score(y_valid, y_pred)
            scores.append(roc_auc)
            models.append(clf)
            break

        if return_models:
            return np.mean(scores), models
        else:
            return np.mean(scores)

    def optimize_models(self):
        _, list_best_models = self.simple_eval()

        optimization_results, best_score = [], []
        for model_name in tqdm(
            list_best_models, total=len(list_best_models), desc="Optimizing models"
        ):
            if model_name in list_best_models:
                study = optuna.create_study(
                    direction="maximize", sampler=TPESampler(seed=10)
                )
                study.optimize(
                    lambda trial: self.objective(trial, model_name),
                    n_trials=50,
                    n_jobs=-1,
                    show_progress_bar=True,
                )
                best_model_params = study.best_params
                valid_scores, model = self.objective(
                    optuna.trial.FixedTrial(study.best_params),
                    model_name,
                    return_models=True,
                )

                result = {
                    "model_name": model_name,
                    "num_trials": len(study.trials),
                    "best_trial_value": valid_scores,
                    "best_trial_params": best_model_params,
                }
                optimization_results.append(result)
                best_score.append(valid_scores)
                print(f"Result for {model_name}: {valid_scores}")
            else:
                continue

        print("Optimization results:")
        for result in optimization_results:
            print(
                f"Model: {result['model_name']}, Num Trials: {result['num_trials']}, Best Trial Value: {result['best_trial_value']}"
            )
            print(f"Best Trial Params: {result['best_trial_params']}")
            print("\n")

        return optimization_results, best_score


class SimpleStacking(BaseEstimator, ClassifierMixin):
    """Стэкинг моделей scikit-learn"""

    def __init__(
        self,
        dict_list: List[Dict[str, Any]],
        ens_models: List[Any],
        blend_weights: Optional[List[float]] = None,
    ) -> None:
        """
        Initialize the class.

        Args:
            dict_list: A list of dictionaries containing models and best_params.
            ens_models: A list of ensembling models.
            blend_weights: Weights for blending meta-models (optional).
        """
        self.dict_list = dict_list
        self.ens_models = ens_models
        self.n_ens = len(ens_models)
        self.n = len(dict_list)
        self.valid = None
        self.ens_predict = None
        self.ground_models = []
        self.blend_weights = blend_weights

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        p: float = 0.25,
        cv: int = 5,
        err: float = 0.001,
        random_state: int = None,
    ) -> "ClassName":
        """
        Train the stacking model.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series, optional): The target variable. Defaults to None.
            p (float): The ratio of train/test split. If p = 0, use all data for training. Defaults to 0.25.
            cv (int): The number of cross-validation folds to use. Only applicable when p = 0. Defaults to 5.
            err (float): The magnitude of random noise to add to the meta-features. Only applicable when p = 0.
                        Defaults to 0.001.
            random_state (int, optional): The seed value for the random number generator. Defaults to None.

        Returns:
            ClassName: The fitted stacking model instance.
        """
        if p > 0:
            sss = StratifiedShuffleSplit(
                n_splits=cv, test_size=p, random_state=random_state
            )
            self.valid = np.zeros((X.shape[0], self.n))
            for t, dict_ in enumerate(self.dict_list):
                model_name = dict_["model_name"]
                best_params = dict_["best_trial_params"]
                model = self.get_model_instance(model_name, best_params)
                if model is not None:
                    for train_index, valid_index in sss.split(X, y):
                        train, valid = X.iloc[train_index], X.iloc[valid_index]
                        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
                        model.fit(train, y_train)
                self.ground_models.append(model)
                self.valid[:, t] = self.ground_models[t].predict_proba(X)[:, 1].ravel()

            ss = StratifiedShuffleSplit(
                n_splits=cv, test_size=p, random_state=random_state
            )
            for t, ens_model in enumerate(self.ens_models):
                for train_index, valid_index in ss.split(self.valid, y):
                    train, valid = self.valid[train_index], self.valid[valid_index]
                    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
                    ens_model.fit(train, y_train)

        else:
            self.valid = err * np.random.randn(X.shape[0], self.n)

            for t, dict_ in enumerate(self.dict_list):
                model_name = dict_["model_name"]
                best_params = dict_["best_trial_params"]
                model = self.get_model_instance(model_name, best_params)
                if model is not None:
                    self.valid[:, t] += cross_val_predict(
                        model, X, y, cv=cv, n_jobs=-1, method="predict"
                    )
                    model.fit(X, y)
                    self.ground_models.append(model)
                    self.valid[:, t] = (
                        self.ground_models[t].predict_proba(X)[:, 1].ravel()
                    )

            ss = StratifiedShuffleSplit(
                n_splits=cv, test_size=p, random_state=random_state
            )
            for t, ens_model in enumerate(self.ens_models):
                for train_index, valid_index in sss.split(self.valid, y):
                    train, valid = self.valid[train_index], self.valid[valid_index]
                    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
                    ens_model.fit(train, y_train)

        return self

    def predict(self, X, y=None, blend=0):
        """
        Predicts the output using a stacking ensemble model.

        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).
            y (numpy.ndarray, optional): Target labels. Defaults to None.
            blend (int, optional): Indicates whether to use blending.
                                0 - No blending, 1 - Use blending.
                                Defaults to 0.

        Returns:
            numpy.ndarray: Predicted output of shape (n_samples, n_models) if blend is 0,
                        otherwise returns the blended output.
        """
        X_meta = np.zeros((X.shape[0], self.n))
        self.ens_predict = np.zeros((X.shape[0], self.n_ens))

        # Generate meta-features using the ground models
        for t, model in enumerate(self.ground_models):
            X_meta[:, t] = model.predict_proba(X)[:, 1].ravel()

        # Predict using the ensemble models
        for t, ens_model in enumerate(self.ens_models):
            self.ens_predict[:, t] = ens_model.predict(X_meta).ravel()

        if blend:
            # Perform blending if blend is 1
            blend_input = np.column_stack(self.ens_predict)
            blend_output = np.dot(blend_input.T, self.blend_weights)
            return blend_output
        else:
            # Return the ensemble predictions if blend is 0
            return np.hsplit(self.ens_predict, self.n_ens)


def get_model_instance(self, model_name: str, best_params: dict) -> Any:
    """
    Create an instance of the specified model and set the best parameters.

    Args:
        model_name: The name of the model.
        best_params: The best parameters for the model.

    Returns:
        An instance of the specified model with the best parameters set.

    Raises:
        ValueError: If the model name is not supported.
    """
    if model_name == "LGBMClassifier":
        clf = LGBMClassifier()
        clf = clf.set_params(verbose=-1, **best_params)
    elif model_name == "XGBClassifier":
        clf = XGBClassifier()
        clf = clf.set_params(verbose=-1, **best_params)
    elif model_name == "CatBoostClassifier":
        clf = CatBoostClassifier(verbose=0, **best_params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return clf


def threshold_predictions(y_pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Transform probabilities into binary labels using a threshold.

    Args:
    - y_pred: np.ndarray
        Vector of probabilities for class 1.
    - threshold: float (default 0.5)
        Threshold for transforming probabilities into binary labels.

    Returns:
    - np.ndarray
        Vector of binary labels (0 or 1).
    """
    # Plot histogram of predictions
    pd.DataFrame(y_pred).hist()

    # Add threshold line to the plot
    plt.axvline(x=threshold, color="red", linestyle="--", label="Threshold")
    plt.legend()

    # Display the plot
    plt.show()

    # Convert probabilities into binary labels
    return (y_pred > threshold).astype(int)


def evaluate_classification_metrics(y_true, y_pred, model_name):
    """
    Evaluate classification metrics for a given model.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        model_name (str): Name of the model.

    Returns:
        None
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Print classification report
    print(
        f"Classification Report for {model_name}:\n",
        classification_report(y_true, y_pred),
    )

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
