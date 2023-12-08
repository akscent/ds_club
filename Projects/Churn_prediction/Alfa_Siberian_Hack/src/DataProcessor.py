import numpy as np
import pandas as pd
import plotly.express as px

class DataProcessor:
    def __init__(self, data, test_data, target, list_drop_cols):
        self.data = data
        self.test_data = test_data
        self.cat_cols = list(data.select_dtypes(include=['object']).columns) + list(data.select_dtypes(include=['category']).columns)
        self.target = target
        
    def rm_spare_cols(self, list_drop_cols=0):
        """
        Удаление лишних столбцов.
            Params:
                list_drop_cols: содержит список наименований столбцов
        """
        if list_drop_cols: self.data.drop(self.list_drop_cols, axis=1, inplace=True)
        
    def rm_NaN_treshold(self, treshold = 200000):
        """
        Удаление всех столбцов, NAN в которых больше заданного порога.
        """
        self.data = self.data.columns[self.data.isnull().sum() > treshold]
        
    def rm_cols_small_target(self, threshold=100):
        columns_to_check = self.data.columns.difference([self.target])
        selected_columns = []
        for column in columns_to_check:
            df_subset = pd.DataFrame({column: self.data[column], target_column: self.data[self.target]})
            df_subset = df_subset.dropna()
            count_ones = df_subset[self.target].sum()
            if count_ones < threshold:
                selected_columns.append(column)

        return self.data.drop(columns = selected_columns, inplace = true)
        
    def rm_or_fill_values(self, rm = True):
        """
        Удаление строк с отрицательными значениями или замена ячеек на NaN.
        """
        if self.cat_cols is not None:
            if rm:
                self.data[~self.cat_cols] = self.data[~self.cat_cols][self.data[~self.cat_cols].le(0).all(axis=1)]
            else:
                self.data[~self.cat_cols].mask(self.data < 0, 0, inplace = True)
    
    def clean_data(self, drop_NaN = False):
        """
        Метод для очистки данных. Удаление дубликатов, обработка пропущенных значений
        """
        self.data = self.data.drop_duplicates()
        if drop_NaN: self.data = self.data.dropna()
            
    def fill_missing_cats(self):
        """
        Method for filling misssing values of categorical features with NaN rows on -1
        """
        categorical_features = self.data.select_dtypes(include=['category']).columns

        for feature in categorical_features:
            self.data[feature] = self.data[feature].astype('object')
            self.data[feature].fillna(-1, inplace=True)
            self.data[feature] = self.data[feature].astype('category')

        return self.data
        
    def fill_nan_with_group_math(self, cat_cols, math = "mean"):
        """
        Method for filling NaN of numerical features by group math variable.
        If group has 0 values, only NaN, than filling with 0
            Params:
                cat_cols: list categorical columns for missing values
                math: mean/median or other
        """
        filled_df = self.data.copy()
        for column in self.data.select_dtypes(include='number').columns:
            if self.data[column].isnull().any():
                temp_df = pd.DataFrame({column: self.data[column]})
                for cat_col in cat_cols:
                    temp_df[cat_col] = df[cat_col]

                group_math = temp_df.groupby(cat_cols)[column].transform(math)
                filled_df[column] = filled_df[column].combine_first(group_math).fillna(0)

        return filled_df
    
    def renaming_okved(self, okved_cat):
        def okved_group(value):
            for key, values in okved_cat.items():
                if pd.isna(value):
                    return 'No'
                if int(value) in values:
                    return key
            else:
                return 'No'
        self.data['okved'] = self.data['okved'].apply(okved_group)

    def preprocess_data(self, list_drop_cols = 0, rmnan_treshold = 200000, treshold = 0, rm_or_fill_values = 0
                       drop_NaN = False, fill_missing_cats = False, fill_cats_NaN = 0, fill_cats_math = 0, 
                        okved_cat = 0):
        """
        Method for preprocessing of pd.DataFrame
        Params:
            list_drop_cols: 0 - False, list_drop_cols - list with rm cols
            rmnan_treshold: treshold with counts of NaN in column for drop
            treshold: 0 - False, treshold with count of target for drop columns
            rm_or_fill_values: 0 - False, 1 - fill values with NaN, 2 - remove rows with negatives
            drop_NaN: False - do not drop rows with NAN, True - drop rows with NAN
            fill_missing_cats: False - do not missing, True - missing
            fill_cats_NaN: 0 - False, list with cat_cols for filling by group math
            fill_cats_math: 0 - False, string with name math variable. For example 'mean'
            okved_cat: 0 - False, dict with grouping okved code one more ierarchie
            
        """
        if list_drop_cols: self.rm_spare_cols(list_drop_cols)
        self.data[self.cat_cols] = self.data[self.cat_cols].astype("category")
        if okved_cat != 0: renaming_okved(self, okved_cat)
        self.rm_NaN_treshold(rmnan_treshold)
        if treshold: self.rm_cols_small_target(threshold)
        if rm_or_fill_values == 1:
            self.rm_or_fill_values(rm = True)
        elif rm_or_fill_values == 2:
            self.rm_or_fill_values(rm = False)
        self.clean_data(drop_NaN)
        if fill_missing_cats: self.data = self.fill_missing_cats(fill_missing_cats)
        if fill_cats_NaN:
            if fill_cats_math: self.fill_nan_with_group_math(cat_cols = self.cat_cols, math = fill_cats_math)
            else: raise ValueError("Invalid math variable for 'math'. Use math variable from pandas math")
        
        
    def visualize_cols_distribution(self, cols_list):
        """
        Визуализация списка столбцов по категориям
        """
        import matplotlib as plt
        labels = self.target
        for column in cols_list:
            df_subset = pd.DataFrame({column: self.data[column], target_column: self.data[labels]})
            df_subset = df_subset.dropna()
            count_ones = df_subset[target_column].sum()
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df_subset, x=column, hue=target_column, bins=30, kde=True)
            plt.title(f'Distribution of "{column}" (Total Label: {count_ones} ones)')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.legend(title=target_column)
            plt.show()
        
    def le_encode_categorical_features(self):
        """
        Кодирует категориальные признаки числовыми значениями
        """
        for col in self.cat_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])

    def visualize_feature(self, col):
        """
        Метод для визуализации признака
        """
        if col in self.cat_cols or col == 'total_target':
            fig = px.pie(self.data, names=col, title=f'Распределение признака {col}')
        else:
            fig = px.histogram(self.data, x=col, title=f'Распределение признака {col}', labels={col: col})
        fig.show()
    
    def visualize_target(self, target):
        """
        Метод для визуализации распределения таргетов
        """
        fig = px.histogram(self.data, x=target, color=target,
                           title='Распределение таргета', labels={target: target},
                           category_orders={target: [0, 1]}, barmode='overlay')
        fig.update_layout(showlegend=False)
        fig.show()
    
    def visualize_correlations(self, max_corr, min_corr):
        """
        Метод для визуализации скоррелированных фичей
        """
        labels_to_drop = {(self.data.columns[i], self.data.columns[j]) for i in range(self.data.shape[1]) for j in range(i + 1)}
        au_corr = self.data.corr().unstack()
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        filtercorr = au_corr[((au_corr >= min_corr) & (au_corr <= max_corr)) | ((au_corr <= -min_corr) & (au_corr >= -max_corr)) & (au_corr !=1.0)]
        au_corr = filtercorr.unstack(level=0)
        fig = px.imshow(au_corr, aspect="auto")
        fig.update_layout(font=dict(size=8))
        fig.show()
        
    def get_churn_category(self, group_by_column, target_column):
        """
        Метод для расчета процента оттока по категориям для категориальных фичей
        """
        grouped_data = self.data.groupby(group_by_column, as_index=False).agg({target_column: ['sum', 'count']})
        grouped_data.columns = [group_by_column, 'Churn_Sum', 'Churn_Count']
        grouped_data['Churn_Percentage'] = 100 * grouped_data['Churn_Sum'] / grouped_data['Churn_Count']
        grouped_data = grouped_data.sort_values('Churn_Percentage').reset_index(drop=True)
        return grouped_data