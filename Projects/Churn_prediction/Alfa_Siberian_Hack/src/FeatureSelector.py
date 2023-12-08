import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
import shap


class FeatureSelector:
    def __init__(self, data, target, features):
        self.data = data
        self.target = target
        self.features = features
        self.cat_cols = self.data.select_dtypes(include=["object"]).columns

    def fill_cat_na(self):
        for col in self.cat_cols:
            mode_value = self.data[col].mode()[0]
            self.data[col].fillna(mode_value, inplace=True)

    def fill_na(self):
        self.data.fillna(-1, inplace=True)

    def catboost_feature_importance(self):
        # CatBoost для оценки важности признаков
        self.fill_cat_na()
        model = CatBoostClassifier(
            iterations=300,
            random_state=42,
            cat_features=list(self.cat_cols),
            verbose=100,
        )
        model.fit(
            self.data[self.features],
            self.data[self.target],
            plot=False,
            early_stopping_rounds=100,
        )
        feature_importance = model.get_feature_importance()
        feature_importance_df = pd.DataFrame(
            {"Feature": self.features, "Importance": feature_importance}
        )
        feature_importance_df = feature_importance_df.sort_values(
            by="Importance", ascending=False
        ).reset_index(drop=True)

        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        fig = plt.figure(figsize=(20, 20))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align="center")
        plt.yticks(range(len(sorted_idx)), np.array(self.data.columns)[sorted_idx])
        plt.title("Feature Importance")

        return feature_importance_df

    def rfe(self, estimator, n_features_to_select=5):
        # RFE для рекурсивного исключения признаков
        self.fill_na()
        rfe = RFE(estimator, n_features_to_select=n_features_to_select)
        rfe.fit(self.data[self.features], self.data[self.target])
        selected_features = self.features[rfe.support_]
        return selected_features

    def pca(self, n_components=5):
        # PCA для отбора признаков
        self.fill_na()
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(self.data[self.features])
        principal_features = [f"PC{i}" for i in range(1, n_components + 1)]
        principal_df = pd.DataFrame(
            data=principal_components, columns=principal_features
        )

        # Вывод объясненной доли дисперсии для каждой компоненты
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_explained_variance = explained_variance_ratio.cumsum()
        print("Explained Variance Ratio for Each Principal Component:")
        for i, ratio in enumerate(explained_variance_ratio, 1):
            print(f"PC{i}: {ratio:.4f}")

        print("\nCumulative Explained Variance:")
        for i, cumulative_ratio in enumerate(cumulative_explained_variance, 1):
            print(f"PC{i}: {cumulative_ratio:.4f}")

        return principal_df

    def shap(self):
        # SHAP values
        self.fill_cat_na()
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.data[self.target], test_size=0.2, random_state=42
        )
        model = CatBoostClassifier(
            iterations=300,
            random_state=42,
            cat_features=list(self.cat_cols),
            verbose=100,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            plot=False,
            early_stopping_rounds=100,
        )
        explainer = shap.TreeExplainer(model)
        val_dataset = Pool(data=X_test, label=y_test, cat_features=list(self.cat_cols))
        shap_values = explainer.shap_values(val_dataset)
        shap.summary_plot(shap_values, X_test, max_display=25)
        features = X_test.columns[np.argsort(np.abs(shap_values).mean(axis=0))[::-1]]

        return features
