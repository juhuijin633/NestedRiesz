import torch


def estimateDiD_OLS(Y1, Y2, D, Z, X1, X2, seed=None):
    """
    Two-step OLS estimator for DiD with time-varying covariates.

    Step 1: Impute X2(0) for treated units by regressing X2 on (X1, Z)
            among control units.
    Step 2: Regress (Y2 - Y1) on D, X1, Z, and imputed X2(0).
            The coefficient on D estimates the ATT.

    Returns:
        theta: ATT point estimate (coefficient on D)
        sigma: HC0 robust standard error for theta
    """
    n = X1.shape[0]
    D_flat = D.view(-1)
    ctrl = (D_flat == 0)
    treated = (D_flat == 1)

    ones = torch.ones(n, 1, dtype=X1.dtype)

    # --- Step 1: Impute X2(0) ---
    W = torch.cat([ones, X1, Z], dim=1)
    W_ctrl = W[ctrl]
    X2_ctrl = X2[ctrl]

    beta_x = torch.linalg.lstsq(W_ctrl, X2_ctrl).solution
    X2_hat = W @ beta_x

    # Use actual X2 for controls, imputed X2(0) for treated
    X2_tilde = X2.clone()
    X2_tilde[treated] = X2_hat[treated]

    # --- Step 2: Regress dY on [1, D, X1, Z, X2_tilde] ---
    dY = (Y2 - Y1).view(-1)
    design = torch.cat([ones, D.view(-1, 1), X1, Z, X2_tilde], dim=1)

    beta = torch.linalg.lstsq(design, dY.unsqueeze(1)).solution.squeeze()

    # theta is the coefficient on D (column index 1)
    theta = beta[1]

    # --- HC0 robust standard errors ---
    residuals = dY - design @ beta
    bread = torch.linalg.inv(design.T @ design)
    meat = (design * residuals.unsqueeze(1)).T @ (design * residuals.unsqueeze(1))
    vcov = bread @ meat @ bread
    sigma = torch.sqrt(vcov[1, 1])

    return theta, sigma