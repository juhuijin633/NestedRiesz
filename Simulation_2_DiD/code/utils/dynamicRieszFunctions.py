import utils.dynamicRieszLASSO
import utils.dynamicRieszRF
import utils.dynamicRieszNet
import torch
import numpy as np
from sklearn.model_selection import KFold
import pdb

lasso_a_settings_global = {
    'lambda_val' : 0,
    'beta_start' : None,
    'D_LB' : 0,
    'D_add' : 0.2,
    'c1' : "CV",
    'c2' : 0.1,
    'tol' : 1e-5,
    'max_iter' : 100,
    'b_degree' : 1,
    'control' : {'maxIter': 1000, 'optTol': 1e-5, 'zeroThreshold': 1e-6}
}
lasso_f_settings_global = {
    'lambda_val' : 0,
    'beta_start' : None,
    'D_LB' : 0,
    'D_add' : 0.2,
    'c1' : "CV",
    'c2' : 0.1,
    'tol' : 1e-5,
    'max_iter' : 100,
    'b_degree' : 1,
    'control' : {'maxIter': 1000, 'optTol': 1e-5, 'zeroThreshold': 1e-6}
}
rf_a_settings_global = {
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
rf_f_settings_global = {
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
net_a_settings_global = {
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
net_f_settings_global = {
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

def thetahat(learners_f, learners_a, Y1, Y2, D, Z, X1, X2):
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
        
    if torch.all(Z == 0):
        predictors_2 = torch.hstack((X1, X2))
        predictors_1 = torch.hstack((X1, Y1))
    else:
        predictors_1 = torch.hstack((Z, X1, Y1))
        predictors_2 = torch.hstack((Z, X1, X2))

    RR1 = torch.zeros(D.shape[0],1)
    RR2 = torch.zeros(D.shape[0],1)

    pi = D / torch.mean(D.float())

    theta = pi * (Y2 - Y1)
    theta -= pi * (learners_f[0].predict(predictors_1, D * 0))    

    RR1 = learners_a[0].predict(predictors_1, D)
    theta -= RR1 * (learners_f[1].predict(predictors_2, D * 0) - learners_f[0].predict(predictors_1, D))

    RR2 = learners_a[1].predict(predictors_2, D)
    theta -= RR2 * ( (Y2 - Y1) - learners_f[1].predict(predictors_2, D))

    # pdb.set_trace()

    return theta, RR1, RR2

def estimateDynamicRiesz(Y1, Y2, D, Z, X1, X2, folds,
                          method_a = "LASSO", method_f = "LASSO",
                            lasso_a_settings = lasso_a_settings_global,
                                lasso_f_settings = lasso_f_settings_global,
                                        rf_a_settings = rf_a_settings_global,
                                            rf_f_settings = rf_f_settings_global,
                                                net_a_settings = net_a_settings_global,
                                                    net_f_settings = net_f_settings_global, seed = None):
        
    fold_results = torch.zeros(Y1.shape)
    RR1 = torch.zeros(Y1.shape)
    RR2 = torch.zeros(Y1.shape)

    # Iterate through folds
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_index, test_index) in enumerate(kf.split(Y1)):

        # Splitting data
        X1_train, X1_test = X1[train_index], X1[test_index]
        X2_train, X2_test = X2[train_index], X2[test_index]
        D_train, D_test = D[train_index], D[test_index]
        Z_train, Z_test = Z[train_index], Z[test_index]
        Y1_train, Y1_test = Y1[train_index], Y1[test_index]
        Y2_train, Y2_test = Y2[train_index], Y2[test_index]

        # Train data for fold
        trainer = Trainer(Y1_train, Y2_train, D_train, Z_train, X1_train, X2_train,
                          method_a, method_f,
                            lasso_a_settings = lasso_a_settings,
                                lasso_f_settings = lasso_f_settings,
                                        rf_a_settings = rf_a_settings,
                                            rf_f_settings = rf_f_settings,
                                                net_a_settings = net_a_settings,
                                                    net_f_settings = net_f_settings, seed = seed)
        trainer.train()

        # Predict theta for every observation in the test data
        fold_results[test_index], RR1[test_index], RR2[test_index] = thetahat(trainer.learners_f, trainer.learners_a, Y1_test, Y2_test, D_test, Z_test, X1_test, X2_test)   
        
    point = torch.mean(fold_results)
    sigma2 = torch.mean( (fold_results - point) ** 2 )
    sigma = torch.sqrt(sigma2)
    
    return point, sigma, RR1, RR2

class Trainer: 
    def __init__(self, Y1, Y2, D, Z, X1, X2,
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

        Y (nx1) : Vector of outcomes
        X (nx(p1...pT)) : Full matrix of covariates
        D (nxT) : Full matrix of treatment assignments 
        d (nxT) : Full matrix of counterfactual assignments 
        
        X_index (T,1) : Shows the index where each new set of covariates starts. E.g. (X1,X1,X1,X2,X2,X3,X3,X3) -> [0,3,5]. 
        """
        self.Y1 = Y1
        self.Y2 = Y2
        self.delta_Y = Y2 - Y1
        self.X1 = X1
        self.X2 = X2
        self.D = D
        self.Z = Z
        self.T = 2

        # initalise trainers for f functions
        if method_f == 'LASSO':
            lasso_f_settings["seed"] = seed
            self.learners_f = [utils.dynamicRieszLASSO.Learner_f_LASSO(lasso_f_settings = lasso_f_settings) for _ in range(self.T)]
        if method_f == 'RF':
            rf_f_settings["random_state"] = seed
            self.learners_f = [utils.dynamicRieszRF.Learner_f_RF(rf_f_settings = rf_f_settings) for _ in range(self.T)]
        if method_f == 'Net':
            torch.manual_seed(seed)
            self.learners_f = [utils.dynamicRieszNet.Learner_f_Net(net_f_settings=net_f_settings) for _ in range(self.T)]
        
        # initalise trainers for a functions
        if method_a == "LASSO":
            lasso_a_settings["seed"] = seed
            self.learners_a = [utils.dynamicRieszLASSO.Learner_a_LASSO(lasso_a_settings = lasso_a_settings) for _ in range(self.T)]
        if method_a == "RF":
            rf_a_settings["random_state"] = seed
            self.learners_a = [utils.dynamicRieszRF.Learner_a_RF(rf_a_settings = rf_a_settings) for _ in range(self.T)]
        if method_a == "Net":
            torch.manual_seed(seed)
            self.learners_a = [utils.dynamicRieszNet.Learner_a_Net(net_a_settings = net_a_settings) for _ in range(self.T)]

    def train(self):
        if torch.all(self.Z == 0):
            predictors_2 = torch.hstack((self.X1, self.X2))
            predictors_1 = torch.hstack((self.X1, self.Y1))

            # Estimate f2
            self.learners_f[1].fit(self.delta_Y, predictors_2, self.D)
            
            # Estimate f1
            f2_hat = self.learners_f[1].predict(predictors_2, self.D * 0)
            self.learners_f[0].fit(f2_hat, predictors_1, self.D)

            # Estimate a1
            a_prev_1 = self.D / torch.mean(self.D.float())
            self.learners_a[0].fit(predictors_1, self.D, a_prev_1)

            # Estimate a2
            a_prev_2 = self.learners_a[0].predict(predictors_1, self.D)
            self.learners_a[1].fit(predictors_2, self.D, a_prev_2)
        else:
            predictors_2 = torch.hstack((self.Z, self.X1, self.X2))
            predictors_1 = torch.hstack((self.Z, self.X1, self.Y1))

            # Estimate f2
            self.learners_f[1].fit(self.delta_Y, predictors_2, self.D)
            
            # Estimate f1
            f2_hat = self.learners_f[1].predict(predictors_2, self.D * 0)
            self.learners_f[0].fit(f2_hat, predictors_1, self.D)

            # Estimate a1
            a_prev_1 = self.D / torch.mean(self.D.float())
            self.learners_a[0].fit(predictors_1, self.D, a_prev_1)

            # Estimate a2
            a_prev_2 = self.learners_a[0].predict(predictors_1, self.D)
            self.learners_a[1].fit(predictors_2, self.D, a_prev_2)



def estimateDynamicRiesz_all(Y1, Y2, D, X1, X2, Z, folds,
                            lasso_a_settings = lasso_a_settings_global,
                                lasso_f_settings = lasso_f_settings_global,
                                        rf_a_settings = rf_a_settings_global,
                                            rf_f_settings = rf_f_settings_global,
                                                net_a_settings = net_a_settings_global,
                                                    net_f_settings = net_f_settings_global):
        
    fold_results = torch.zeros(Y2.shape[0],9)
    RR1 = torch.zeros(Y2.shape[0],9)
    RR2 = torch.zeros(Y2.shape[0],9)

    # Iterate through folds
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(kf.split(Y)):

        # Splitting data
        X1_train, X1_test = X1[train_index], X1[test_index]
        X2_train, X2_test = X2[train_index], X2[test_index]
        D_train, D_test = D[train_index], D[test_index]
        Z_train, Z_test = Z[train_index], Z[test_index]
        Y1_train, Y1_test = Y1[train_index], Y1[test_index]
        Y2_train, Y2_test = Y2[train_index], Y2[test_index]

        # Train data for fold
        trainer = Trainer(Y1_train, Y2_train, D_train, Z_train, X1_train, X2_train,
                            lasso_a_settings = lasso_a_settings,
                                lasso_f_settings = lasso_f_settings,
                                        rf_a_settings = rf_a_settings,
                                            rf_f_settings = rf_f_settings,
                                                net_a_settings = net_a_settings,
                                                    net_f_settings = net_f_settings)
        trainer.train()

        # Predict theta for every observation in the test data
        fold_results[test_index,:1], RR1[test_index,:1], RR2[test_index,:1] = thetahat(trainer.learners_f_LASSO, trainer.learners_a_LASSO, Y1_test, Y2_test, D_test, Z_test, X1_test, X2_test)   
        fold_results[test_index,1:2], RR1[test_index,1:2], RR2[test_index,1:2] = thetahat(trainer.learners_f_LASSO, trainer.learners_a_RF, Y1_test, Y2_test, D_test, Z_test, X1_test, X2_test)   
        fold_results[test_index,2:3], RR1[test_index,2:3], RR2[test_index,2:3] = thetahat(trainer.learners_f_LASSO, trainer.learners_a_Net, Y1_test, Y2_test, D_test, Z_test, X1_test, X2_test)   
        fold_results[test_index,3:4], RR1[test_index,3:4], RR2[test_index,3:4] = thetahat(trainer.learners_f_RF, trainer.learners_a_LASSO, Y1_test, Y2_test, D_test, Z_test, X1_test, X2_test)   
        fold_results[test_index,4:5], RR1[test_index,4:5], RR2[test_index,4:5] = thetahat(trainer.learners_f_RF, trainer.learners_a_RF, Y1_test, Y2_test, D_test, Z_test, X1_test, X2_test)   
        fold_results[test_index,5:6], RR1[test_index,5:6], RR2[test_index,5:6] = thetahat(trainer.learners_f_RF, trainer.learners_a_Net, Y1_test, Y2_test, D_test, Z_test, X1_test, X2_test)   
        fold_results[test_index,6:7], RR1[test_index,6:7], RR2[test_index,6:7] = thetahat(trainer.learners_f_Net, trainer.learners_a_LASSO, Y1_test, Y2_test, D_test, Z_test, X1_test, X2_test)   
        fold_results[test_index,7:8], RR1[test_index,7:8], RR2[test_index,7:8] = thetahat(trainer.learners_f_Net, trainer.learners_a_RF, Y1_test, Y2_test, D_test, Z_test, X1_test, X2_test)   
        fold_results[test_index,8:9], RR1[test_index,8:9], RR2[test_index,8:9] = thetahat(trainer.learners_f_Net, trainer.learners_a_Net, Y1_test, Y2_test, D_test, Z_test, X1_test, X2_test)   

    point = torch.mean(fold_results,0)
    sigma2 = torch.mean( (fold_results - point) ** 2 ,0)
    sigma = torch.sqrt(sigma2)
    
    return point, sigma, RR1, RR2

class Trainer_all: 
    def __init__(self, Y, X, D, d, X_index,
                    lasso_a_settings = lasso_a_settings_global,
                      lasso_f_settings = lasso_f_settings_global,
                            rf_a_settings = rf_a_settings_global,
                                rf_f_settings = rf_f_settings_global,
                                    net_a_settings = net_a_settings_global,
                                        net_f_settings = net_f_settings_global):
                
        """
        Dictionary mapping function 

        Y (nx1) : Vector of outcomes
        X (nx(p1...pT)) : Full matrix of covariates
        D (nxT) : Full matrix of treatment assignments 
        d (nxT) : Full matrix of counterfactual assignments 
        
        X_index (T,1) : Shows the index where each new set of covariates starts. E.g. (X1,X1,X1,X2,X2,X3,X3,X3) -> [0,3,5]. 
        """
        self.Y = Y
        self.X = X
        self.D = D
        self.d = d
        self.X_index = X_index
        self.T = 2

        # initalise trainers for f functions
        self.learners_f_LASSO = [utils.dynamicRieszLASSO.Learner_f_LASSO(lasso_f_settings = lasso_f_settings) for _ in range(self.T)]
        self.learners_f_RF = [utils.dynamicRieszRF.Learner_f_RF(rf_f_settings = rf_f_settings) for _ in range(self.T)]
        self.learners_f_Net = [utils.dynamicRieszNet.Learner_f_Net(net_f_settings=net_f_settings) for _ in range(self.T)]

        # initalise trainers for a functions
        self.learners_a_LASSO = [utils.dynamicRieszLASSO.Learner_a_LASSO(lasso_a_settings = lasso_a_settings) for _ in range(self.T)]
        self.learners_a_RF = [utils.dynamicRieszRF.Learner_a_RF(rf_a_settings = rf_a_settings) for _ in range(self.T)]
        self.learners_a_Net = [utils.dynamicRieszNet.Learner_a_Net(net_a_settings = net_a_settings) for _ in range(self.T)]


    def train(self):

        predictors_2 = torch.hstack((self.Z, self.X1, self.X2))
        predictors_1 = torch.hstack((self.Z, self.X1, self.Y1))

        # LASSO
        # Estimate f2
        self.learners_f_LASSO[1].fit(self.delta_Y, predictors_2, self.D)
        # Estimate f1
        f2_hat = self.learners_f_LASSO[1].predict(predictors_2, self.D * 0)
        self.learners_f_LASSO[0].fit(f2_hat, predictors_1, self.D)

        # RF
        # Estimate f2
        self.learners_f_RF[1].fit(self.delta_Y, predictors_2, self.D)
        # Estimate f1
        f2_hat = self.learners_f_RF[1].predict(predictors_2, self.D * 0)
        self.learners_f_RF[0].fit(f2_hat, predictors_1, self.D)

        # Net
        # Estimate f2
        self.learners_f_Net[1].fit(self.delta_Y, predictors_2, self.D)
        # Estimate f1
        f2_hat = self.learners_f_Net[1].predict(predictors_2, self.D * 0)
        self.learners_f_Net[0].fit(f2_hat, predictors_1, self.D)

        # LASSO
        # Estimate a1
        a_prev_1 = self.D / torch.mean(self.D.float())
        self.learners_a_LASSO[0].fit(predictors_1, self.D, a_prev_1)
        # Estimate a2
        a_prev_2 = self.learners_a_LASSO[0].predict(predictors_1, self.D)
        self.learners_a_LASSO[1].fit(predictors_2, self.D, a_prev_2)

        # RF
        # Estimate a1
        a_prev_1 = self.D / torch.mean(self.D.float())
        self.learners_a_RF[0].fit(predictors_1, self.D, a_prev_1)
        # Estimate a2
        a_prev_2 = self.learners_a_RF[0].predict(predictors_1, self.D)
        self.learners_a_RF[1].fit(predictors_2, self.D, a_prev_2)
        
        # Net
        # Estimate a1
        a_prev_1 = self.D / torch.mean(self.D.float())
        self.learners_a_Net[0].fit(predictors_1, self.D, a_prev_1)
        # Estimate a2
        a_prev_2 = self.learners_a_Net[0].predict(predictors_1, self.D)
        self.learners_a_Net[1].fit(predictors_2, self.D, a_prev_2)