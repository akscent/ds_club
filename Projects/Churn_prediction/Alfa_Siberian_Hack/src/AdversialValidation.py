import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


class AdversarialValidation:
    def __init__(self, train_df, test_df, features, target):
        self.train_df = train_df
        self.test_df = test_df
        self.features = features
        self.target = target
        self.cat_cols = self.train_df.select_dtypes(include=["object"]).columns

    def create_adversarial_dataset(self):
        # Добавляем метку "0" для train_df и "1" для test_df
        self.train_df["is_train"] = 0
        self.test_df["is_train"] = 1

        # Объединяем оба набора данных
        adv_data = pd.concat([self.train_df, self.test_df], axis=0)

        # Заполняем пропущенные значения в категориальных признаках
        for col in self.cat_cols:
            mode_value = adv_data[col].mode()[0]
            adv_data[col].fillna(mode_value, inplace=True)

        return adv_data

    def run_adversarial_validation(self, params=None):
        # Создаем adversarial dataset
        adv_data = self.create_adversarial_dataset()

        # Разделяем на обучающий и тестовый наборы для adversarial validation
        X_adv = adv_data[self.features]
        y_adv = adv_data["is_train"]

        X_train_adv, X_valid_adv, y_train_adv, y_valid_adv = train_test_split(
            X_adv, y_adv, test_size=0.2, random_state=42
        )

        # Инициализируем и обучаем модель CatBoost
        model_params = params or {
            "objective": "Logloss",
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3,
            "verbose": 100,
        }

        model = CatBoostClassifier(**model_params)
        model.fit(
            X_train_adv,
            y_train_adv,
            cat_features=list(self.cat_cols),
            eval_set=(X_valid_adv, y_valid_adv),
            early_stopping_rounds=50,
            verbose_eval=100,
        )

        # Предсказываем вероятности для adversarial dataset
        adv_pred = model.predict_proba(X_valid_adv)[:, 1]

        # Вычисляем ROC AUC для adversarial validation
        auc_score = roc_auc_score(y_valid_adv, adv_pred)
        print(f"Adversarial Validation ROC AUC Score: {auc_score}")
        return auc_score

    def stats_for_cols(self):
        features_list = self.test_df.select_dtypes(
            include=["number"]
        ).columns.values.tolist()
        bad_features, good_features = [], []

        for feature in features_list:
            stat, p_value = stats.kstest(
                self.train_df[feature].dropna(), self.test_df[feature].dropna()
            )
            x_stat, x_p_value = stats.chisquare(
                self.train_df[feature].dropna(), self.test_df[feature].dropna()
            )
            if stat > 0.1 and p_value < 0.05:
                print(
                    "KS test value: %.3f" % statistic,
                    "with a p-value %.2f" % p_value,
                    "for the feature",
                    feature,
                )
                print(
                    f"Chi-squared Test value: {stat:.4f}, with p-value: {p_value:.4f}"
                )
                self.plot_cumulative_distribution(feature)
                bad_features.append(feature)
            else:
                good_features.append(feature)

    # next step - stats for more than 2 groups, for ex. ANOVA test

    def plot_cumulative_distribution(self, feature):
        plt.figure(figsize=(10, 6))
        sns.ecdfplot(self.train_df[feature].dropna(), label="Train Data")
        sns.ecdfplot(self.test_df[feature].dropna(), label="Test Data")

        plt.title(f"Cumulative Distribution for {feature}")
        plt.xlabel(feature)
        plt.ylabel("Cumulative Probability")
        plt.legend()
        plt.show()
