import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from features import *


right_feature_set = ["Vote", "Yearly_IncomeK", "Number_of_differnt_parties_voted_for", "Political_interest_Total_Score",
                     "Avg_Satisfaction_with_previous_vote", "Avg_monthly_income_all_years",
                     "Most_Important_Issue", "Overall_happiness_score", "Avg_size_per_room",
                     "Weighted_education_rank"]


def save_files(df_train, df_test, df_validation):
    df_test.to_csv('prepared_new_test.csv', index=False)


def remove_na(df_test):
    df_test = df_test.dropna()

    return df_test


def complete_missing_values(df_test: pd.DataFrame) -> pd.DataFrame:
    df_test = df_test[df_test >= 0]

    for col in df_test.columns.values:
        filler = None
        if col in nominal_features:
            filler = df_test[col].mode()[0]

        if col in integer_features:
            filler = round(df_test[col].mean())

        if col in float_features:
            filler = df_test[col].mean()

        df_test[col].fillna(filler, inplace=True)

    return df_test


def nominal_to_numerical_categories(df_test: pd.DataFrame, df_train: pd.DataFrame, df_validation: pd.DataFrame):
    # from nominal to Categorical
    df_train = df_train.apply(lambda x: pd.Categorical(x) if x.dtype != 'float64' else x, axis=0)
    # give number to each Categorical
    df_train = df_train.apply(lambda x: x.cat.codes if x.dtype != 'float64' else x, axis=0)

    # from nominal to Categorical
    df_validation = df_validation.apply(lambda x: pd.Categorical(x) if x.dtype != 'float64' else x, axis=0)
    # give number to each Categorical
    df_validation = df_validation.apply(lambda x: x.cat.codes if x.dtype != 'float64' else x, axis=0)

    # from nominal to Categorical
    df_test = df_test.apply(lambda x: pd.Categorical(x) if x.dtype != 'float64' else x, axis=0)
    # give number to each Categorical
    df_test = df_test.apply(lambda x: x.cat.codes if x.dtype != 'float64' else x, axis=0)

    return df_test, df_train, df_validation


def apply_feature_selection(df_train, df_test, df_validation, feature_set):
    df_train = df_train[feature_set]
    df_test = df_test[feature_set]
    df_validation = df_validation[feature_set]

    return df_train, df_test, df_validation


def normalize(df_test: pd.DataFrame, df_train: pd.DataFrame, df_validation: pd.DataFrame):
    # min-max for uniform features
    #uniform_scaler = MinMaxScaler(feature_range=(-1, 1))
    #df_train[uniform_features_right_features] = uniform_scaler.fit_transform(df_train[uniform_features_right_features])
    #df_validation[uniform_features_right_features] = uniform_scaler.transform(df_validation[uniform_features_right_features])
    #df_test[uniform_features_right_features] = uniform_scaler.transform(df_test[uniform_features_right_features])

    # z-score for normal features
    normal_scaler = StandardScaler()
    df_train[normal_features_right_features] = normal_scaler.fit_transform(df_train[normal_features_right_features])
    df_validation[normal_features_right_features] = normal_scaler.transform(df_validation[normal_features_right_features])
    df_test[normal_features_right_features] = normal_scaler.transform(df_test[normal_features_right_features])

    #quick fix
    df_train[["Most_Important_Issue"]] = normal_scaler.fit_transform(df_train[["Most_Important_Issue"]])
    df_validation[["Most_Important_Issue"]] = normal_scaler.transform(df_validation[["Most_Important_Issue"]])
    df_test[["Most_Important_Issue"]] = normal_scaler.transform(df_test[["Most_Important_Issue"]])

    ##now everyone will be between -1 and 1
    #df_train[normal_features_right_features] = uniform_scaler.fit_transform(df_train[normal_features_right_features])
    #df_validation[normal_features_right_features] = uniform_scaler.transform(df_validation[normal_features_right_features])
    #df_test[normal_features_right_features] = uniform_scaler.transform(df_test[normal_features_right_features])

    return df_train, df_test, df_validation


def remove_outliers(threshold: float, df_train: pd.DataFrame, df_validation: pd.DataFrame, df_test: pd.DataFrame):
    mean = df_train[normal_features_right_features].mean()
    std = df_train[normal_features_right_features].std()

    z_train = (df_train[normal_features_right_features] - mean) / std
    z_val = (df_validation[normal_features_right_features] - mean) / std
    z_test = (df_test[normal_features_right_features] - mean) / std

    df_train[z_train.mask(abs(z_train) > threshold).isna()] = np.nan
    df_validation[z_val.mask(abs(z_val) > threshold).isna()] = np.nan
    df_test[z_test.mask(abs(z_test) > threshold).isna()] = np.nan

    return df_train, df_validation, df_test


def main():
    # first part - data preparation
    df_new_test = pd.read_csv("ElectionsData_Pred_Features.csv")

    # apply feature selection
    df_new_test = apply_feature_selection(df_new_test,  right_feature_set)

    # Convert nominal types to numerical categories
    df_new_test = nominal_to_numerical_categories(df_new_test)

    # 1 - Imputation - Complete missing values
    df_new_test = complete_missing_values(df_new_test)

    # 2 - Data Cleansing
    # Outlier detection using z score

    threshold = 3  # .3
    df_new_test = remove_outliers(threshold, df_new_test)

    # Remove lines containing na values
    df_new_test = remove_na(df_new_test)

    # 3 - Normalization (scaling)
    df_new_test = normalize(df_new_test)

    # step number 3
    # CSV files of the prepared new test data set
    save_files(df_new_test)


if __name__ == '__main__':
    main()
