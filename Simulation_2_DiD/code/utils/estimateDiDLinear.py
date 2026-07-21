import utils.dynamicRieszLASSO
import torch
import numpy as np
import pdb
from sklearn.linear_model import LassoCV, LinearRegression

lasso_f_settings_global = {
    'lambda_val' : 0,
    'beta_start' : None,
    'D_LB' : 0,
    'D_add' : 0.2,
    'c1' : 0,
    'c2' : 0,
    'tol' : 1e-5,
    'max_iter' : 100,
    'b_degree' : 1,
    'control' : {'maxIter': 1000, 'optTol': 1e-5, 'zeroThreshold': 1e-6}
}

def estimateDiDLinear(Y1, Y2, D, Z, X1, X2, seed =None):

    fold_results = torch.zeros(Y1.shape)

    trainer = Trainer(Y1, Y2, D, Z, X1, X2, seed = seed)
    trainer.train()

    fold_results = trainer.theta0  
        
    point = torch.mean(fold_results)
    sigma2 = torch.mean( (fold_results - point) ** 2 )
    sigma = torch.sqrt(sigma2)
    
    return point, sigma

class Trainer:

    def __init__(self, Y1, Y2, D, Z, X1, X2, seed = None):

        self.Y1 = Y1
        self.Y2 = Y2
        self.delta_Y = Y2 - Y1
        self.X1 = X1
        self.X2 = X2
        self.delta_X = X2 - X1
        self.D = D
        self.Z = Z
        self.T = 2
        self.seed = seed
        lasso_f_settings_global["seed"] = seed

        #self.learner_x = utils.dynamicRieszLASSO.Learner_f_LASSO(lasso_f_settings = lasso_f_settings_global)
        
        #self.learner_f = utils.dynamicRieszLASSO.Learner_f_LASSO(lasso_f_settings = lasso_f_settings_global)
        self.learner_x = LinearRegression()
        self.learner_f = LinearRegression()

    def train(self):
        if torch.all(self.Z == 0):
            # For delta X2(0)

            #predictors_1 = torch.hstack((self.X1, self.Y1))
            predictors_1 = torch.hstack((self.X1, self.Y1, self.D))
            predictors_1_D0 = torch.hstack((self.X1, self.Y1, self.D * 0))
            predictors_2 = self.X1
            for i in range(self.X2.shape[1]):
                # Estimate delta X equation by LASSO
                
                #self.learner_x.fit(self.delta_X[:,i:i+1], predictors_1, self.D)
                self.learner_x.fit(predictors_1, self.delta_X[:,i:i+1])
                #  Estimate delta X2(0)
                #dX2_0_hat = self.learner_x.predict(predictors_1, self.D * 0)
                dX2_0_hat = self.learner_x.predict(predictors_1_D0)
                predictors_2 = torch.hstack((predictors_2, dX2_0_hat))

            # For delta Y(0)
            predictors_2_merge = torch.hstack((predictors_2, self.D))
            predictors_2_D0_merge = torch.hstack((predictors_2, self.D*0))
            # Estimate delta Y equation by LASSO
            #self.learner_f.fit(self.delta_Y, predictors_2, self.D)
            self.learner_f.fit(predictors_2_merge, self.delta_Y)
            # Estimate delta Y(0)
            dY_0_hat = self.learner_f.predict(predictors_2_D0_merge)

            # Estimate ATT
            self.theta0 = (self.delta_Y[self.D.squeeze() == 1]) - (dY_0_hat[self.D.squeeze() == 1])

        else:
            # For delta X2(0)

            #predictors_1 = torch.hstack((self.Z, self.X1, self.Y1))
            predictors_1 = torch.hstack((self.Z, self.X1, self.Y1, self.D))
            predictors_1_D0 = torch.hstack((self.Z, self.X1, self.Y1, self.D * 0))
            predictors_2 = torch.hstack((self.Z, self.X1))
            for i in range(self.X2.shape[1]):
               # Estimate delta X equation by LASSO
                
                #self.learner_x.fit(self.delta_X[:,i:i+1], predictors_1, self.D)
                self.learner_x.fit(predictors_1, self.delta_X[:,i:i+1])
                #  Estimate delta X2(0)
                #dX2_0_hat = self.learner_x.predict(predictors_1, self.D * 0)
                dX2_0_hat = torch.tensor(self.learner_x.predict(predictors_1_D0))
                predictors_2 = torch.hstack((predictors_2, dX2_0_hat))


            # For delta Y(0)
            predictors_2_merge = torch.hstack((predictors_2, self.D))
            predictors_2_D0_merge = torch.hstack((predictors_2, self.D*0))
            # Estimate delta Y equation by LASSO
            self.learner_f.fit(predictors_2_merge, self.delta_Y)
            # Estimate delta Y(0)
            #dY_0_hat = self.learner_f.predict(predictors_2, self.D * 0)
            dY_0_hat =  torch.tensor(self.learner_f.predict(predictors_2_D0_merge))

            # Estimate ATT
            self.theta0 = (self.delta_Y[self.D.squeeze() == 1]) - (dY_0_hat[self.D.squeeze() == 1])



        # # With LASSO package
        # if self.Z is None:
        #     # For delta X2(0)

        #     predictors_1 = torch.hstack((self.X1, self.Y1))
        #     predictors_2 = torch.hstack((self.X1))

        #     for i in range(self.X2.shape[1]):

        #         lasso_x = LassoCV(cv=5).fit(predictors_1[self.D.squeeze() == 0,:], self.delta_X[self.D.squeeze() == 0,i:i+1].squeeze())
        #         dX2_0_hat = torch.tensor(lasso_x.predict(predictors_1).reshape(-1, 1))
        #         predictors_2 = torch.hstack((predictors_2, dX2_0_hat))

        #     # For delta Y(0)

        #     lasso_y = LassoCV(cv=5).fit(predictors_2[self.D.squeeze() == 0,:], self.delta_Y[self.D.squeeze() == 0,i:i+1].squeeze())
        #     dY_0_hat = torch.tensor(lasso_y.predict(predictors_1).reshape(-1, 1))

        #     # Estimate ATT
        #     self.theta0 = (self.delta_Y[self.D.squeeze() == 1]) - (dY_0_hat[self.D.squeeze() == 1])

        # # With LASSO package
        # if self.Z is not None:
        #     # For delta X2(0)

        #     predictors_1 = torch.hstack((self.Z, self.X1, self.Y1))
        #     predictors_2 = torch.hstack((self.Z, self.X1))

        #     for i in range(self.X2.shape[1]):

        #         lasso_x = LassoCV(cv=5).fit(predictors_1[self.D.squeeze() == 0,:], self.delta_X[self.D.squeeze() == 0,i:i+1].squeeze())
        #         dX2_0_hat = torch.tensor(lasso_x.predict(predictors_1).reshape(-1, 1))
        #         predictors_2 = torch.hstack((predictors_2, dX2_0_hat))

        #     # For delta Y(0)

        #     lasso_y = LassoCV(cv=5).fit(predictors_2[self.D.squeeze() == 0,:], self.delta_Y[self.D.squeeze() == 0,i:i+1].squeeze())
        #     dY_0_hat = torch.tensor(lasso_y.predict(predictors_1).reshape(-1, 1))

        #     # Estimate ATT
        #     self.theta0 = (self.delta_Y[self.D.squeeze() == 1]) - (dY_0_hat[self.D.squeeze() == 1])