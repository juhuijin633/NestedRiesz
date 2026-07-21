import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
APP_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
os.chdir(APP_DIR)

import scipy.io
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import utils.dynamicRieszLASSO
import utils.dynamicRieszRF
import utils.dynamicRieszNet
from utils.dynamicRieszFunctions import *
import torch
import numpy as np
from sklearn.model_selection import KFold
import pdb

river_data = pd.read_csv("data/river_data.csv") # experimental
other_data = pd.read_csv("data/others_data.csv") # observational

# Define pretreatment variables
pretreat_vars = (
    [f"paid{i}" for i in range(1, 5)] +         # 4 lagged values for aid
    [f"tcpp{i}" for i in range(1, 11)] +        # 10 lagged values for employment
    [f"tcprn{i}" for i in range(1, 11)]         # 10 lagged values for earnings
)

# List of used covariates
covariates = [
    "xsexf", "xhsdip", "xchld05", "single",
    "grd1720", "grade16", "grd1315", "grade12", "grde911", "white",
    "hisp", "black", "age"
] + pretreat_vars

def create_dataset(quarters, application):
    if application not in ["earn", "employ"]:
        raise ValueError
    Y_obsorved = other_data[f"Y_{application}"]
    Y_experimental = river_data[f"Y_{application}"] # not used in fitting, only to evaluate
    if application == "employ":
        s_colums = (
            [f"{application}{i}" for i in range(1, quarters+ 1)] +
            [f"aid{i}" for i in range(1, quarters+ 1)] +
            [f"earn{i}" for i in range(1, quarters + 1)]
        )
    elif application == "earn":
        s_colums = (
            [f"{application}{i}" for i in range(1, quarters+ 1)] +
            [f"aid{i}" for i in range(1, quarters + 1)]
        )

    S_obs = other_data[s_colums]
    S_exp = river_data[s_colums]
    D_exp = river_data["e"]
    D_obs = other_data["e"]
    X_obs = other_data[covariates]
    X_exp = river_data[covariates]
    Y_all = pd.concat([Y_obsorved, Y_experimental], axis=0).reset_index(drop=True)
    X_all = pd.concat([X_obs, X_exp], axis=0).reset_index(drop=True)
    S_all = pd.concat([S_obs, S_exp], axis=0).reset_index(drop=True)
    D_all = pd.concat([D_obs, D_exp], axis=0).reset_index(drop=True)
    G_all = pd.concat([
        pd.Series(np.ones(len(D_obs))),
        pd.Series(np.zeros(len(D_exp)))
    ], axis=0).reset_index(drop=True)

    Y_all_torch = torch.tensor(Y_all.values, dtype=torch.float64).view(-1, 1)
    X_all_torch = torch.tensor(X_all.values, dtype=torch.float64)
    S_all_torch = torch.tensor(S_all.values, dtype=torch.float64)
    D_all_torch = torch.tensor(D_all.values, dtype=torch.float64).view(-1, 1)
    G_all_torch = torch.tensor(G_all.values, dtype=torch.float64).view(-1, 1)
    return {"Y_obsorved": Y_obsorved,"Y_experimental": Y_experimental, "S_obs": S_obs, "S_exp":S_exp,
            "D_exp": D_exp, "D_ops":D_obs, "X_obs": X_obs, "X_exp": X_exp, "Y_all" : Y_all_torch, "X_all": X_all_torch, "S_all": S_all_torch, "D_all": D_all_torch, "G_all": G_all_torch,
            "names_x":covariates, "names_z":s_colums }

lasso_cv_settings = {
    'b_degree' : 1,
    'cv_folds' : 5,
    'random_state' : 42
}

lasso_a_settings = {
    'lambda_val' : 0,
    'beta_start' : None,
    'D_LB' : 0,
    'D_add' : 0.2,
    'c1' : 0.1, # CHANGED FROM "CV"
    'c2' : 0.1,
    'tol' : 1e-5,
    'max_iter' : 100,
    'b_degree' : 1,
    'control' : {'maxIter': 1000, 'optTol': 1e-5, 'zeroThreshold': 1e-6}
}

lasso_f_settings = {
    'lambda_val' : 0,
    'beta_start' : None,
    'D_LB' : 0,
    'D_add' : 0.2,
    'c1' :  0.1, # CHANGED FROM "CV"
    'c2' : 0.1,
    'tol' : 1e-5,
    'max_iter' : 100,
    'b_degree' : 1,
    'control' : {'maxIter': 1000, 'optTol': 1e-5, 'zeroThreshold': 1e-6}
}

rf_a_settings= {
    'poly_degree' : 0,
    'l2' : 0,
    'n_estimators' : 100,
    'criterion' : "mse",
    'max_depth' : None,
    'min_samples_split' : 10,
    'min_samples_leaf' : 5,
    'min_weight_fraction_leaf' : 0.,
    'min_var_fraction_leaf' : None,
    'min_var_leaf_on_val' : False,
    'max_features' : "auto",
    'min_impurity_decrease' : 0.,
    'max_samples' : .45,
    'min_balancedness_tol' : .45,
    'honest' : True,
    'inference' : True,
    'fit_intercept' : True,
    'subforest_size' : 4,
    'n_jobs' : 1,
    'random_state' : None,
    'verbose' : 0,
    'warm_start' : False
}
rf_f_settings = {
    'poly_degree' : 1,
    'l2' : 0,
    'n_estimators' : 100,
    'criterion' : "mse",
    'max_depth' : None,
    'min_samples_split' : 10,
    'min_samples_leaf' : 5,
    'min_weight_fraction_leaf' : 0.,
    'min_var_fraction_leaf' : None,
    'min_var_leaf_on_val' : False,
    'max_features' : "auto",
    'min_impurity_decrease' : 0.,
    'max_samples' : .45,
    'min_balancedness_tol' : .45,
    'honest' : True,
    'inference' : True,
    'fit_intercept' : True,
    'subforest_size' : 4,
    'n_jobs' : 1,
    'random_state' : None,
    'verbose' : 0,
    'warm_start' : False
}

net_a_settings = {
    'test_split' : 0,
    'learner_lr' : 1e-4,
    'learner_l2' : 1e-3,
    'learner_l1' : 0,
    'n_epochs' : 100,
    'earlystop_rounds' : 20,
    'earlystop_delta' : 1e-3,
    'bs' : 64,
    'optimizer' : 'adam',
    'warm_start' : False,
    'logger' : None,
    'model_dir' : '.',
    'device' : torch.cuda.current_device() if torch.cuda.is_available() else None,
    'n_hidden' : 100,
    'drop_prob' : 0,
    'degree' : 2,
    'interaction_only' : True,
    'n_common' : 200,
    'act_func' : 'elu'
}
net_f_settings = {
    'test_split' : 0,
    'learner_lr' : 1e-4,
    'learner_l2' : 1e-3,
    'learner_l1' : 0,
    'n_epochs' : 100,
    'earlystop_rounds' : 20,
    'earlystop_delta' : 1e-3,
    'bs' : 64,
    'optimizer' : 'adam',
    'warm_start' : False,
    'logger' : None,
    'model_dir' : '.',
    'device' : torch.cuda.current_device() if torch.cuda.is_available() else None,
    'n_hidden' : 100,
    'drop_prob' : 0,
    'degree' : 2,
    'interaction_only' : True,
    'n_common' : 200,
    'act_func' : 'elu'
}

def thetahat(learners_f, learners_a, Y, G, X, D, S):
    """
    This function gives the estimate for theta. 
    
    trainer (Trainer class) : the trainer from which the a_t and f_t functions will be used
    Y : Outcome data
    X: Covariates
    D : Treatment 
    d : Counterfactual treatment
    X_index : Denotes where period t covariates start

    Can delete the RR
    """
    predictors_2 = torch.hstack((X, S))
    predictors_1 = X

    rr_variables_1 = torch.hstack((G, D))
    rr_variables_2 = G
    d1_1 = torch.tensor([0, 1]).repeat(rr_variables_1.shape[0], 1)
    d2_1 = torch.ones(rr_variables_2.shape)
    d1_0 = torch.zeros(rr_variables_1.shape)
    d2_0 = torch.ones(rr_variables_2.shape)

    pi = (1 - G) / torch.mean((1 - G.float()))

    # First the estimate of Yi(1)
    theta = pi * (learners_f[0].predict(predictors_1, torch.ones(D.shape))) 
    # theta = pi * (learners_f[0].predict(predictors_1, torch.ones(D.shape))) 
    with torch.no_grad():                         # optional if predict already no-grad
        f2 = learners_f[1].predict(predictors_2)  # tensor
    f2 = f2.detach().view(-1, 1)

    theta -= pi * learners_a[0].predict(predictors_1, rr_variables_1) * (f2 - learners_f[0].predict(predictors_1, D))
    # theta -= learners_a[0].predict(predictors_1, rr_variables_1) * (learners_f[1].predict(predictors_2, d2_1) - learners_f[0].predict(predictors_1, D))

    theta -= learners_a[1].predict(predictors_2, rr_variables_2) * (Y - f2)

    # Now add the estimate of Yi(0)
    theta -= pi * (learners_f[0].predict(predictors_1, torch.zeros(D.shape)))      
    # theta -= pi * (learners_f[0].predict(predictors_1, torch.zeros(D.shape)))      

    theta += pi*learners_a[2].predict(predictors_1, rr_variables_1) * (f2 - learners_f[0].predict(predictors_1, D))
    # theta += learners_a[2].predict(predictors_1, rr_variables_1) * (learners_f[1].predict(predictors_2, d2_0) - learners_f[0].predict(predictors_1, D))

    theta += learners_a[3].predict(predictors_2, rr_variables_2) * (Y - f2)

    # pdb.set_trace()
    return theta.float()


def estimateDynamicRiesz_subsetting_net(Y, G, X, D, S, folds,
                          method_a = "LASSO", method_f = "LASSO",
                            lasso_a_settings = lasso_a_settings_global,
                                lasso_f_settings = lasso_f_settings_global,
                                        rf_a_settings = rf_a_settings_global,
                                            rf_f_settings = rf_f_settings_global,
                                                net_a_settings = net_a_settings_global,
                                                    net_f_settings = net_f_settings_global, seed = None):
        
    fold_results = torch.zeros(Y.shape, dtype=torch.float64)

    # Iterate through folds
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(kf.split(Y)):

        # Splitting data
        Y_train, Y_test = Y[train_index], Y[test_index]
        G_train, G_test = G[train_index], G[test_index]
        X_train, X_test = X[train_index], X[test_index]
        D_train, D_test = D[train_index], D[test_index]
        S_train, S_test = S[train_index], S[test_index]

        # Train data for fold
        trainer = Trainer(Y_train, G_train, X_train, D_train, S_train,
                          method_a, method_f,
                            lasso_a_settings = lasso_a_settings,
                                lasso_f_settings = lasso_f_settings,
                                        rf_a_settings = rf_a_settings,
                                            rf_f_settings = rf_f_settings,
                                                net_a_settings = net_a_settings,
                                                    net_f_settings = net_f_settings, seed = seed)
        trainer.train()

        # Predict theta for every observation in the test data
        fold_results[test_index] = thetahat(trainer.learners_f, trainer.learners_a, Y_test, G_test, X_test, D_test, S_test ).double()
        
    point = torch.mean(fold_results)
    sigma2 = torch.mean( (fold_results - point) ** 2 )
    sigma = torch.sqrt(sigma2)
    
    return point, sigma

class Trainer: 
    def __init__(self, Y, G, X, D, S,
                  method_a = 'LASSO', method_f = "LASSO",
                    lasso_a_settings = lasso_a_settings_global,
                      lasso_f_settings = lasso_f_settings_global,
                            rf_a_settings = rf_a_settings_global,
                                rf_f_settings = rf_f_settings_global,
                                    net_a_settings = net_a_settings_global,
                                        net_f_settings = net_f_settings_global, 
                                        seed = None):
                
        """
        Dictionary mapping function 

        Y (nX) : Vector of outcomes
        X (nx(p1...pT)) : Full matrix of covariates
        D (nxT) : Full matrix of treatment assignments 
        d (nxT) : Full matrix of counterfactual assignments 
        
        X_index (T,1) : Shows the index where each new set of covariates starts. E.g. (X,X,X,X2,X2,X3,X3,X3) -> [0,3,5]. 
        """
        self.Y = Y
        self.G = G
        self.X = X
        self.D = D
        self.S = S
        self.T = 2
        self.method_a = method_a
        self.method_f = method_f
        # initalise trainers for f functions
        if method_f == 'LASSO':
            lasso_f_settings["seed"] = seed
            self.learners_f = [utils.dynamicRieszLASSO.Learner_f_LASSO(lasso_f_settings = lasso_f_settings) for _ in range(self.T)]
        if method_f == 'RF':
            rf_f_settings["random_state"] = seed
            self.learners_f = [utils.dynamicRieszRF.Learner_f_RF(rf_f_settings = rf_f_settings) for _ in range(self.T)]
        if method_f == 'Net':
            self.learners_f = [utils.dynamicRieszNet.Learner_f_Net(net_f_settings=net_f_settings) for _ in range(self.T)]
        
        # initalise trainers for a functions
        if method_a == "LASSO":
            lasso_a_settings["seed"] = seed
            self.learners_a = [utils.dynamicRieszLASSO.Learner_a_LASSO(lasso_a_settings = lasso_a_settings) for _ in range(self.T * 2)]
        if method_a == "RF":
            rf_a_settings["random_state"] = seed
            self.learners_a = [utils.dynamicRieszRF.Learner_a_RF(rf_a_settings = rf_a_settings) for _ in range(self.T * 2)]
        if method_a == "Net":
            self.learners_a = [utils.dynamicRieszNet.Learner_a_Net(net_a_settings = net_a_settings) for _ in range(self.T * 2)]

    def train(self):
        predictors_2 = torch.hstack((self.X, self.S))
        predictors_1 = self.X

        rr_variables_1 = torch.hstack((self.G, self.D))
        rr_variables_2 = self.G
        d1_1 = torch.tensor([0, 1]).repeat(rr_variables_1.shape[0], 1)
        d2 = torch.ones(rr_variables_2.shape)
        d1_0 = torch.zeros(rr_variables_1.shape)

        # Estimate f2, only on the dataset with  G=1,#
        mask_G0 = (1 - self.G).bool().squeeze()
        mask_G1 =  self.G.bool().squeeze()

        if  self.method_f == "Net":
            self.learners_f[1].fit(self.Y[mask_G1].ravel(), predictors_2[mask_G1])
            f2_hat = self.learners_f[1].predict(predictors_2)

        # Estimate f1
        else:
            f2_hat = torch.tensor(self.learners_f[1].predict(predictors_2))  
        
        self.learners_f[0].fit(f2_hat[mask_G0].view(-1,1), predictors_1[mask_G0], self.D[mask_G0])
        # mask = (1 - self.G).bool().squeeze()

        # self.learners_f[0].fit(f2_hat[mask], predictors_1[mask], self.D[mask])

        # Estimate a1^1
        a_prev_1 = (1 - self.G) / torch.mean(1 - self.G.float())
        self.learners_a[0].fit(predictors_1, rr_variables_1, d1_1, a_prev_1)

        # Estimate a2^1
        a_prev_21 = self.learners_a[0].predict(predictors_1, rr_variables_1)
        self.learners_a[1].fit(predictors_2, rr_variables_2, d2, a_prev_21)

        # Estimate a1^0
        a_prev_1 = (1 - self.G) / torch.mean(1 - self.G.float())
        self.learners_a[2].fit(predictors_1, rr_variables_1, d1_0, a_prev_1)

        # Estimate a2^0
        a_prev_20 = self.learners_a[2].predict(predictors_1, rr_variables_1)
        self.learners_a[3].fit(predictors_2, rr_variables_2, d2, a_prev_20)

folds = 5
q = 6
application = os.environ.get("SURROGACY_APPLICATION", "earn")
ds = create_dataset(q, application)
Y_all = ds["Y_all"]
X_all = ds["X_all"]
S_all = ds["S_all"]
D_all = ds["D_all"]
G_all = ds["G_all"]

D_changed = D_all.clone()
Y_changed = Y_all.clone()

###################
D_changed[G_all.bool()] = 0
Y_changed[(1 - G_all).bool()] = 0
torch.manual_seed(0)
print("Starting Net at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
ATT_6_Net, std_6_net= estimateDynamicRiesz_subsetting_net(
    Y_changed, G_all, X_all, D_changed, S_all, folds,
    method_a="Net", net_a_settings=net_a_settings,
    method_f="Net", net_f_settings=net_f_settings, seed=0
)
print("Finished Net at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))




torch.manual_seed(0)
print("Starting LASSO at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
ATT_6_LS, std_6_ls = estimateDynamicRiesz(
    Y_changed, G_all, X_all, D_changed, S_all, folds,
    method_a="LASSO", lasso_a_settings=lasso_a_settings,
    method_f="LASSO", lasso_f_settings=lasso_f_settings, seed=0
)
print("Finished LASSO at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))



torch.manual_seed(0)
print("Starting RF at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
ATT_6_RF, std_6_rf = estimateDynamicRiesz(
    Y_changed, G_all, X_all, D_changed, S_all, folds,
    method_a="RF", rf_a_settings=rf_a_settings,
    method_f="RF", rf_f_settings=rf_f_settings, seed=0
)
print("Finished RF at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


results = {
    "ATT_LASSO": ATT_6_LS,
    "Std_LASSO": std_6_ls,
    "ATT_RF": ATT_6_RF,
    "Std_RF": std_6_rf,
    "ATT_Net": ATT_6_Net,
    "Std_Net": std_6_net,
}


os.makedirs("results", exist_ok=True)
df = pd.DataFrame([results])
df.to_csv(f"results/auto_{application}.csv", index=False)
