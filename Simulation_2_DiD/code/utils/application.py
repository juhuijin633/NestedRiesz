import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder

class application_data():
    def __init__(self):
        self.data = pd.read_csv("https://raw.githubusercontent.com/CausalAIBook/MetricsMLNotebooks/main/data/minwage_data.csv", index_col=0)

    def get_data(self, pre_treatment_year, effect_year, treatment_year = None, baseline_2001 = False):

        if treatment_year is None:
            treatment_year = effect_year

        g_filter = (self.data["G"] == 0) | (self.data["G"] > effect_year) | (self.data["G"] == treatment_year)
        g_filtered_data = self.data[g_filter]
        filtered_data = g_filtered_data[(g_filtered_data["year"] == pre_treatment_year) | (g_filtered_data["year"] == effect_year)]

        # Outcome variables
        Y1 = torch.tensor(
            filtered_data[filtered_data["year"] == pre_treatment_year]["lemp"].values, dtype=torch.float32
        ).view(-1, 1)
        Y2 = torch.tensor(
            filtered_data[filtered_data["year"] == effect_year]["lemp"].values, dtype=torch.float32
        ).view(-1, 1)
        baseline_vars = ['lemp', 'lpop', 'lavg_pay']
        # Covariates #
        remove_columns  = [
            "lemp", "treated", "state_name", "FIPS", "quarter", "censusdiv",
            "region", "ever_treated", "id", "state_mw", "fed_mw", "G", "countyreal", "emp0A01_BS",  "countyreal", 'pop', "annual_avg_pay"
        ]
        ids = filtered_data["id"]
        X = filtered_data.drop(columns=remove_columns).copy()

        X1 = X[X["year"] == pre_treatment_year].drop(columns=["year"]).reset_index(drop=True)
        ids = ids[X["year"] == pre_treatment_year]
        X2 = X[X["year"] == effect_year].drop(columns=["year"]).reset_index(drop=True)
        print("Changing covariates: ", X1.columns.to_list())    
        X1 = torch.tensor(X1.values, dtype=torch.float32)
        X2 = torch.tensor(X2.values, dtype=torch.float32)

        post_treatment_data = filtered_data[filtered_data["year"] == effect_year]
        D = torch.tensor(
            (post_treatment_data["G"] == treatment_year).astype(float).values.reshape(-1, 1),
            dtype=torch.float32
        )
            
        # Z variables
        Z = filtered_data[filtered_data["year"] == effect_year][["region"]].reset_index(drop=True)
        # One-hot encoding for categorical variables
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        Z_encoded = encoder.fit_transform(Z)
        Z = pd.DataFrame(Z_encoded, columns=encoder.get_feature_names_out())
        if baseline_2001:
            baselines = g_filtered_data[g_filtered_data["year"] == 2001][baseline_vars].reset_index(drop = True).add_prefix("2001_")
            assert Z.shape[0] == baselines.shape[0]
            Z = Z.merge(baselines, left_index=True, right_index=True)

        print("Z variables: ", Z.columns.to_list())

        Z = torch.tensor(Z.values,  dtype=torch.float32)

        return {
            "X1": X1,
            "X2": X2,
            "D": D,
            "Y1": Y1,
            "Y2": Y2,
            "Z": Z
        }