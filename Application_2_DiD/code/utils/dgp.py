import torch
from tqdm import tqdm

class DiD_DGP:
    def __init__(self, dim_X=3, dim_Z = 3, beta_1=2.0, beta_2=1.5, c_1=1.0, delta_1 = 1 , delta_2 =1 , delta_3=1 , alpha_1 =1, gamma_1 = None, gamma_2= None, g  = None):
        self.dim_X = dim_X
        self.dim_Z = dim_Z
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.alpha_1 = alpha_1
        self.c_1 = c_1
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.delta_3 = delta_3
        self.gamma_1 = gamma_1 # dimension dim_Z
        self.gamma_2 = gamma_2 # dimension dim_Z
        if gamma_1 is None: # if not specified, Z does not affect the propensity score
            self.gamma_1 = torch.zeros(dim_Z)
        if gamma_2 is None: # if not specified, Z does not affect the outcome
            self.gamma_2 = torch.zeros(dim_Z)
        self.g = g # this function specifies the propensity model
        self.ATT = None
       

    def generate(self, n, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        X1 = torch.randn(n, self.dim_X)
        X11 = X1[:, 0]
        eta = torch.randn(n)
        eps_x = torch.randn(n, self.dim_X)
        Z = torch.randn(n, self.dim_Z)
        Y1 = self.beta_1 * (X11 > 0).float() + self.delta_1 * X11 + Z @ self.gamma_2 + torch.randn(n) 
        prob_D = self.g(self.delta_2* X11 + Z @ self.gamma_1 + self.alpha_1 * Y1 + eta) # propensity score
        D = torch.bernoulli(prob_D)
        X2 = X1 + torch.ones(n, self.dim_X) *  (D *(1 + eta)).unsqueeze(1)  + eps_x
        X21 = X2[:, 0]
        Y2 = Y1 + self.c_1 * D.squeeze() + self.beta_2 * (X21 > 0).float() + self.delta_3* D.squeeze() * X21 + torch.randn(n)
        return {
            "X1": X1,
            "X2": X2,
            "Y1": Y1,
            "Y2": Y2,
            "D": D,
            "Z": Z,
        }    
    def simulate_ATT(self, n = 1000000):
        X1 = torch.randn(n, self.dim_X)
        X11 = X1[:, 0]
        eta = torch.randn(n)
        eps_x = torch.randn(n, self.dim_X)
        Z = torch.randn(n, self.dim_Z)
        Y1 = self.beta_1 * (X11 > 0).float() + self.delta_1 * X11 + Z @ self.gamma_2 + torch.randn(n) 
        prob_D = self.g(self.delta_2* X11 + Z @ self.gamma_1 + self.alpha_1 * Y1 + eta) 
        D = torch.bernoulli(prob_D)
        X2_0 = X1 + eps_x
        X2_1 = X1 + torch.ones(n, self.dim_X) *  (1 + eta).unsqueeze(1)  + eps_x
        X21_0 = X2_0[:, 0]
        X21_1 = X2_1[:, 0]
        # Calculate ATT
        E_X1D1 = torch.mean(X11[D == 1])  # E[X11 | D = 1]
        E_ETAD1 = torch.mean(eta[D == 1]) # E[ETA | D = 1]
        I_X21D1 = (X21_1 >0).float()
        I_X20D1 = (X21_0 > 0).float()
        E_X21D1 = torch.mean(I_X21D1[D == 1]) # P(X21(1) > 0 | D = 1)
        E_X20D1 = torch.mean(I_X20D1[D == 1])  # P(X21(0) > 0 | D = 1)
        ATT = self.c_1 + self.beta_2 * E_X21D1 - self.beta_2 * E_X20D1 + self.delta_3 * E_X1D1 + self.delta_3* E_ETAD1 + self.delta_3*1
        del X1, X11, eta, eps_x, Z, Y1, prob_D, D, X2_0, X2_1, X21_0, X21_1, I_X21D1, I_X20D1
        return {"ATT": ATT, 
                "E_X1D1": E_X1D1, "E_ETAD1": E_ETAD1, 
                "E_X21D1": E_X21D1, "E_X20D1": E_X20D1,}
        
class DiD_DGP_v2:
    def __init__(self, dim_X=3, dim_Z = 3, beta_1=2.0, beta_2=1.5, c_1=1.0, delta_1 = 1 , delta_2 =1 , delta_3=1 , alpha_1 =1, gamma_1 = None, gamma_2= None, g  = None, gamma_10 = 0,gamma_11=0 ):
        self.dim_X = dim_X
        self.dim_Z = dim_Z
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.alpha_1 = alpha_1
        self.c_1 = c_1
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.delta_3 = delta_3
        self.gamma_1 = gamma_1 # dimension dim_Z
        self.gamma_2 = gamma_2 # dimension dim_Z
        self.gamma_10 = gamma_10
        self.gamma_11 = gamma_11
        if gamma_1 is None: # if not specified, Z does not affect the propensity score
            self.gamma_1 = torch.zeros(dim_Z)
        if gamma_2 is None: # if not specified, Z does not affect the outcome
            self.gamma_2 = torch.zeros(dim_Z)
        self.g = g # this function specifies the propensity model
        self.ATT = None
       

    def generate(self, n, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        X1 = torch.randn(n, self.dim_X)
        X11 = X1[:, 0]
        eta = torch.randn(n)
        eps_x = torch.randn(n, self.dim_X)
        Z = torch.randn(n, self.dim_Z)
        Y1 = self.beta_1 * (X11 > 0).float() + self.delta_1 * X11 + Z @ self.gamma_2 + torch.randn(n) 
        prob_D = self.g(self.delta_2* X11 + Z @ self.gamma_1 + self.alpha_1 * Y1 + eta) # propensity score
        D = torch.bernoulli(prob_D)
        I_x11 = (X11 > 0)
        X2 = ((D*self.gamma_11 + (1-D)*self.gamma_10)*I_x11).unsqueeze(1)  +X1 + torch.ones(n, self.dim_X) *  (D *(1 + eta)).unsqueeze(1)  + eps_x
        X21 = X2[:, 0]
        Y2 = Y1 + self.c_1 * D.squeeze() + self.beta_2 * (X21 > 0).float() + self.delta_3* D.squeeze() * X21 + torch.randn(n)
        return {
            "X1": X1,
            "X2": X2,
            "Y1": Y1,
            "Y2": Y2,
            "D": D,
            "Z": Z,
        }    
    def simulate_att_population(self, n=int(5e6), batch_size=int(5e5), seed=None):
        
        if seed is not None:
            torch.manual_seed(seed)

        #
        diff_sum_treated = 0.0
        n_treated = 0


        ones_template = None  


        num_full = n // batch_size
        remainder = n - num_full * batch_size

        def run_batch(m):
            nonlocal diff_sum_treated, n_treated, ones_template


            X1 = torch.randn(m, self.dim_X)
            X11 = X1[:, 0]
            Z  = torch.randn(m, self.dim_Z)
            eta = torch.randn(m)
            eps_x  = torch.randn(m, self.dim_X)
            eps_y1 = torch.randn(m)
            eps_y2 = torch.randn(m)   # reused for both potentials


            Y1 = self.beta_1 * (X11 > 0).float() + self.delta_1 * X11 + Z @ self.gamma_2 + eps_y1


            score = self.delta_2 * X11 + Z @ self.gamma_1 + self.alpha_1 * Y1 + eta
            p = self.g(score)
            D = torch.bernoulli(p)

            # === Build X2(d) under do(D=d) with same shocks
            I_x11 = (X11 > 0).float()
            if ones_template is None:
                ones_template = torch.ones(1, self.dim_X)

            def X2_given(d):
                d = float(d)
                shift = (d * (1 + eta)).unsqueeze(1)
                gate  = ((d * self.gamma_11 + (1 - d) * self.gamma_10) * I_x11).unsqueeze(1)
                return gate + X1 + ones_template.expand(m, -1) * shift + eps_x

            X2_0 = X2_given(0.0)
            X2_1 = X2_given(1.0)
            X21_0 = X2_0[:, 0]
            X21_1 = X2_1[:, 0]

            # === Potential outcomes with identical eps_y2
            Y2_0 = Y1 + self.beta_2 * (X21_0 > 0).float()  + eps_y2
            Y2_1 = Y1 + self.c_1 + self.beta_2 * (X21_1 > 0).float() + self.delta_3 * 1.0 * X21_1 + eps_y2

            # === ATT accumulator
            treated = (D == 1)
            if treated.any():
                diff_sum_treated += (Y2_1 - Y2_0)[treated].sum().item()
                n_treated += treated.sum().item()

        # Run full batches
        for _ in tqdm(range(num_full), desc = "Running diferent batches to simualte the true ATT"):
            run_batch(batch_size)

        # Run remainder if needed
        if remainder > 0:
            run_batch(remainder)

        # Compute ATT
        att = diff_sum_treated / n_treated

        # Store and return
        self.ATT = att

        return att
                
            