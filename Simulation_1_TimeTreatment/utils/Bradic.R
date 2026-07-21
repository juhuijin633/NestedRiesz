require(glmnet);require(itertools);require(foreach);require(doParallel)

num_piest = 3
num_mu1est = 3
eps <- 1e-6
grid=data.frame(est_mu1=1:num_mu1est,est_pi=1:num_piest)

N = nrow(S1)
M=floor(N/K)

pred_pi1_1=matrix(0,nrow=num_piest,ncol=N)
pred_pi2_0=matrix(0,nrow=num_piest,ncol=N);pred_pi2_1=matrix(0,nrow=num_piest,ncol=N)
pred_mu2_1=matrix(0,nrow=num_mu1est,ncol=N);pred_mu1_1=matrix(0,nrow=num_mu1est,ncol=N)
pred_mu2_0=matrix(0,nrow=num_mu1est,ncol=N);pred_mu1_0=matrix(0,nrow=num_mu1est,ncol=N)

#estimation of pi1, pi2, mu1, mu2
for (k in 1:K){
  index=(1:M)+(k-1)*M
  index_nk=(1:N)[-index]
  
  index_nk1=index_nk[1:floor(length(index_nk)/2)]
  index1_nk1=index_nk1[which(A1[index_nk1]==1)];index0_nk1=index_nk1[which(A1[index_nk1]==0)]
  index11_nk1=index_nk1[which(A1[index_nk1]*A2[index_nk1]==1)];index00_nk1=index_nk1[which(A1[index_nk1]+A2[index_nk1]==0)]
  
  index_nk2=index_nk[-(1:floor(length(index_nk)/2))]
  index1_nk2=index_nk2[which(A1[index_nk2]==1)];index0_nk2=index_nk2[which(A1[index_nk2]==0)]
  index11_nk2=index_nk2[which(A1[index_nk2]*A2[index_nk2]==1)];index00_nk2=index_nk2[which(A1[index_nk2]+A2[index_nk2]==0)]
  
  index1_nk=index_nk[which(A1[index_nk]==1)];index0_nk=index_nk[which(A1[index_nk]==0)]
  index11_nk=index_nk[which(A1[index_nk]*A2[index_nk]==1)];index00_nk=index_nk[which(A1[index_nk]+A2[index_nk]==0)]
  #pi1
  fit_Log1=cv.glmnet(x=S1[index_nk,],y=A1[index_nk],family="binomial")
  pred_pi1_1[1,index]=predict(fit_Log1,newx=S1[index,],type="response",s="lambda.min") #logistic+lasso
  pred_pi1_1[2,index]=pred_pi1_1[1,index]
  pred_pi1_1[3,index]=pred_pi1_1[1,index]
  #pi2_1
  temp=try({fit_Log2_1=cv.glmnet(x=cbind(S1[index1_nk,],S2[index1_nk,]),y=A2[index1_nk],family="binomial")},TRUE)
  if ('try-error' %in% class(temp)){
    pred_pi2_1[1,index]=rep(mean(A2[index1_nk]),length(index))
  } else {
    pred_pi2_1[1,index]=predict(fit_Log2_1,newx=cbind(S1[index,],S2[index,]),type="response",s="lambda.min") #logistic
  }
  pred_pi2_1[2,index]=pred_pi2_1[1,index]
  pred_pi2_1[3,index]=pred_pi2_1[1,index]
  #pi2_0
  temp=try({fit_Log2_0=cv.glmnet(x=cbind(S1[index0_nk,],S2[index0_nk,]),y=A2[index0_nk],family="binomial")},TRUE)
  if ('try-error' %in% class(temp)){
    pred_pi2_0[1,index]=rep(mean(A2[index0_nk]),length(index))
  } else {
    pred_pi2_0[1,index]=predict(fit_Log2_0,newx=cbind(S1[index,],S2[index,]),type="response",s="lambda.min") #logistic
  }
  pred_pi2_0[2,index]=pred_pi2_0[1,index]
  pred_pi2_0[3,index]=pred_pi2_0[1,index]
  #mu2_1
  fit_lasso1=cv.glmnet(x=cbind(S1[index11_nk,],S2[index11_nk,]),y=Y[index11_nk],family="gaussian",alpha=1)
  pred_mu2_1[1,index]=predict(fit_lasso1,newx=cbind(S1[index,],S2[index,]),s="lambda.min") #lasso
  pred_mu2_1[2,index]=pred_mu2_1[1,index]
  pred_mu2_1[3,index]=pred_mu2_1[1,index]
  #mu2_0
  fit_lasso0=cv.glmnet(x=cbind(S1[index00_nk,],S2[index00_nk,]),y=Y[index00_nk],family="gaussian",alpha=1)
  pred_mu2_0[1,index]=predict(fit_lasso0,newx=cbind(S1[index,],S2[index,]),s="lambda.min") #lasso
  pred_mu2_0[2,index]=pred_mu2_0[1,index]
  pred_mu2_0[3,index]=pred_mu2_0[1,index]
  #DTL
  #mu1_1
  pred_mu2_1_nocross=predict(fit_lasso1,newx=cbind(S1[index1_nk,],S2[index1_nk,]),s="lambda.min")
  fit_lasso2_nocross=cv.glmnet(x=S1[index1_nk,],y=pred_mu2_1_nocross,family="gaussian",alpha=1)
  pred_mu1_1[2,index]=predict(fit_lasso2_nocross,newx=S1[index,],s="lambda.min")
  #mu1_0
  pred_mu2_0_nocross=predict(fit_lasso0,newx=cbind(S1[index0_nk,],S2[index0_nk,]),s="lambda.min")
  fit_lasso2_nocross=cv.glmnet(x=S1[index0_nk,],y=pred_mu2_0_nocross)
  pred_mu1_0[2,index]=predict(fit_lasso2_nocross,newx=S1[index,],s="lambda.min")
  #D-DRL
  #mu1_1
  fit_lasso1_new1=cv.glmnet(x=cbind(S1[index11_nk1,],S2[index11_nk1,]),y=Y[index11_nk1],family="gaussian",alpha=1)
  temp=try({fit_Log2_1_new1=cv.glmnet(x=cbind(S1[index1_nk1,],S2[index1_nk1,]),y=A2[index1_nk1],family="binomial")},TRUE)
  if ('try-error' %in% class(temp)){
    pred_pi2_1_new1=rep(mean(A2[index1_nk1]),length(index1_nk2))
  } else {
    pred_pi2_1_new1=predict(fit_Log2_1_new1,newx=cbind(S1[index1_nk2,],S2[index1_nk2,]),type="response",s="lambda.min")
  }
  pred_mu2_1_new1=predict(fit_lasso1_new1,newx=cbind(S1[index1_nk2,],S2[index1_nk2,]),s="lambda.min")
  pred_pi2_1_new1 <- pmax(pmin(pred_pi2_1_new1, 1-eps), eps)
  Y1_new1=as.numeric(pred_mu2_1_new1+A2[index1_nk2]*(Y[index1_nk2]-pred_mu2_1_new1)/pred_pi2_1_new1)
  fit_lasso2_nocross1_new1=cv.glmnet(x=S1[index1_nk2,],y=Y1_new1,family="gaussian",alpha=1)
  pred_mu1_1_new1=predict(fit_lasso2_nocross1_new1,newx=S1[index,],s="lambda.min")
  
  fit_lasso1_new2=cv.glmnet(x=cbind(S1[index11_nk2,],S2[index11_nk2,]),y=Y[index11_nk2],family="gaussian",alpha=1)
  temp=try({fit_Log2_1_new2=cv.glmnet(x=cbind(S1[index1_nk2,],S2[index1_nk2,]),y=A2[index1_nk2],family="binomial")},TRUE)
  if ('try-error' %in% class(temp)){
    pred_pi2_1_new2=rep(mean(A2[index1_nk2]),length(index1_nk1))
  } else {
    pred_pi2_1_new2=predict(fit_Log2_1_new2,newx=cbind(S1[index1_nk1,],S2[index1_nk1,]),type="response",s="lambda.min")
  }
  pred_mu2_1_new2=predict(fit_lasso1_new2,newx=cbind(S1[index1_nk1,],S2[index1_nk1,]),s="lambda.min")
  pred_pi2_1_new2 <- pmax(pmin(pred_pi2_1_new2, 1-eps), eps)
  Y1_new2=as.numeric(pred_mu2_1_new2+A2[index1_nk1]*(Y[index1_nk1]-pred_mu2_1_new2)/pred_pi2_1_new2)
  fit_lasso2_nocross1_new2=cv.glmnet(x=S1[index1_nk1,],y=Y1_new2,family="gaussian",alpha=1)
  pred_mu1_1_new2=predict(fit_lasso2_nocross1_new2,newx=S1[index,],s="lambda.min")
  pred_mu1_1[1,index]=(pred_mu1_1_new1+pred_mu1_1_new2)/2
  
  #mu1_0
  fit_lasso0_new1=cv.glmnet(x=cbind(S1[index00_nk1,],S2[index00_nk1,]),y=Y[index00_nk1],family="gaussian",alpha=1)
  temp=try({fit_Log2_0_new1=cv.glmnet(x=cbind(S1[index0_nk1,],S2[index0_nk1,]),y=A2[index0_nk1],family="binomial")},TRUE)
  if ('try-error' %in% class(temp)){
    pred_pi2_0_new1=rep(mean(A2[index0_nk1]),length(index0_nk2))
  } else {
    pred_pi2_0_new1=predict(fit_Log2_0_new1,newx=cbind(S1[index0_nk2,],S2[index0_nk2,]),type="response",s="lambda.min")
  }
  pred_mu2_0_new1=predict(fit_lasso0_new1,newx=cbind(S1[index0_nk2,],S2[index0_nk2,]),s="lambda.min")
  pred_pi2_0_new1 <- pmax(pmin(pred_pi2_0_new1, 1-eps), eps)
  Y0_new1=as.numeric(pred_mu2_0_new1+(1-A2[index0_nk2])*(Y[index0_nk2]-pred_mu2_0_new1)/(1-pred_pi2_0_new1))
  fit_lasso2_nocross0_new1=cv.glmnet(x=S1[index0_nk2,],y=Y0_new1,family="gaussian",alpha=1)
  pred_mu1_0_new1=predict(fit_lasso2_nocross0_new1,newx=S1[index,],s="lambda.min")
  
  fit_lasso0_new2=cv.glmnet(x=cbind(S1[index00_nk2,],S2[index00_nk2,]),y=Y[index00_nk2],family="gaussian",alpha=1)
  temp=try({fit_Log2_0_new2=cv.glmnet(x=cbind(S1[index0_nk2,],S2[index0_nk2,]),y=A2[index0_nk2],family="binomial")},TRUE)
  if ('try-error' %in% class(temp)){
    pred_pi2_0_new2=rep(mean(A2[index0_nk2]),length(index0_nk1))
  } else {
    pred_pi2_0_new2=predict(fit_Log2_0_new2,newx=cbind(S1[index0_nk1,],S2[index0_nk1,]),type="response",s="lambda.min")
  }
  pred_mu2_0_new2=predict(fit_lasso0_new2,newx=cbind(S1[index0_nk1,],S2[index0_nk1,]),s="lambda.min")
  pred_pi2_0_new2 <- pmax(pmin(pred_pi2_0_new2, 1-eps), eps)
  Y0_new2=as.numeric(pred_mu2_0_new2+(1-A2[index0_nk1])*(Y[index0_nk1]-pred_mu2_0_new2)/(1-pred_pi2_0_new2))
  fit_lasso2_nocross0_new2=cv.glmnet(x=S1[index0_nk1,],y=Y0_new2,family="gaussian",alpha=1)
  pred_mu1_0_new2=predict(fit_lasso2_nocross0_new2,newx=S1[index,],s="lambda.min")
  pred_mu1_0[1,index]=(pred_mu1_0_new1+pred_mu1_0_new2)/2
  #D-DRL'
  #mu1_1
  fit_lasso1_new=cv.glmnet(x=cbind(S1[index11_nk,],S2[index11_nk,]),y=Y[index11_nk],family="gaussian",alpha=1)
  temp=try({fit_Log2_1_new=cv.glmnet(x=cbind(S1[index1_nk,],S2[index1_nk,]),y=A2[index1_nk],family="binomial")},TRUE)
  if ('try-error' %in% class(temp)){
    pred_pi2_1_new=rep(mean(A2[index1_nk]),length(index1_nk))
  } else {
    pred_pi2_1_new=predict(fit_Log2_1_new,newx=cbind(S1[index1_nk,],S2[index1_nk,]),type="response",s="lambda.min")
  }
  pred_mu2_1_new=predict(fit_lasso1_new,newx=cbind(S1[index1_nk,],S2[index1_nk,]),s="lambda.min")
  pred_pi2_1_new <- pmax(pmin(pred_pi2_1_new, 1-eps), eps)
  Y1_new=as.numeric(pred_mu2_1_new+A2[index1_nk]*(Y[index1_nk]-pred_mu2_1_new)/pred_pi2_1_new)
  fit_lasso2_nocross1_new=cv.glmnet(x=S1[index1_nk,],y=Y1_new,family="gaussian",alpha=1)
  pred_mu1_1[3,index]=predict(fit_lasso2_nocross1_new,newx=S1[index,],s="lambda.min")
  
  #mu1_0
  fit_lasso0_new=cv.glmnet(x=cbind(S1[index00_nk,],S2[index00_nk,]),y=Y[index00_nk],family="gaussian",alpha=1)
  temp=try({fit_Log2_0_new=cv.glmnet(x=cbind(S1[index0_nk,],S2[index0_nk,]),y=A2[index0_nk],family="binomial")},TRUE)
  if ('try-error' %in% class(temp)){
    pred_pi2_0_new=rep(mean(A2[index0_nk]),length(index0_nk))
  } else {
    pred_pi2_0_new=predict(fit_Log2_0_new,newx=cbind(S1[index0_nk,],S2[index0_nk,]),type="response",s="lambda.min")
  }
  pred_mu2_0_new=predict(fit_lasso0_new,newx=cbind(S1[index0_nk,],S2[index0_nk,]),s="lambda.min")
  pred_pi2_0_new <- pmax(pmin(pred_pi2_0_new, 1-eps), eps)
  Y0_new=as.numeric(pred_mu2_0_new+(1-A2[index0_nk])*(Y[index0_nk]-pred_mu2_0_new)/(1-pred_pi2_0_new))
  fit_lasso2_nocross0_new=cv.glmnet(x=S1[index0_nk,],y=Y0_new,family="gaussian",alpha=1)
  pred_mu1_0[3,index]=predict(fit_lasso2_nocross0_new,newx=S1[index,],s="lambda.min")
}

#clip propensity matrices before computing IPW weights
pred_pi1_1 <- pmax(pmin(pred_pi1_1, 1-eps), eps)
pred_pi2_1 <- pmax(pmin(pred_pi2_1, 1-eps), eps)
pred_pi2_0 <- pmax(pmin(pred_pi2_0, 1-eps), eps)

#estimation of theta
pred_theta_t=c()
pred_sig2_t=c()
pred_psi1_t=c()
pred_psi0_t=c()
pred_sig2_psi1_t=c()
pred_sig2_psi0_t=c();

for (j in 1:nrow(grid)){
  x=as.numeric(grid[j,])
  gamma2_1=A1*A2/(pred_pi1_1[x[2],]*pred_pi2_1[x[2],]);gamma1_1=A1/pred_pi1_1[x[2],]
  psi_1=gamma2_1*Y-(gamma1_1-1)*pred_mu1_1[x[1],]-(gamma2_1-gamma1_1)*pred_mu2_1[x[1],]
  gamma2_0=(1-A1)*(1-A2)/((1-pred_pi1_1[x[2],])*(1-pred_pi2_0[x[2],]));gamma1_0=(1-A1)/(1-pred_pi1_1[x[2],])
  psi_0=gamma2_0*Y-(gamma1_0-1)*pred_mu1_0[x[1],]-(gamma2_0-gamma1_0)*pred_mu2_0[x[1],]
  pred_theta_t=c(pred_theta_t,mean(psi_1-psi_0))
  pred_psi1_t=c(pred_psi1_t,mean(psi_1))
  pred_psi0_t=c(pred_psi0_t,mean(psi_0))
}

#asymptotic variance estimator
for (j in 1:nrow(grid)){
  x=as.numeric(grid[j,])
  gamma2_1=A1*A2/(pred_pi1_1[x[2],]*pred_pi2_1[x[2],]);gamma1_1=A1/pred_pi1_1[x[2],]
  psi_1=gamma2_1*Y-(gamma1_1-1)*pred_mu1_1[x[1],]-(gamma2_1-gamma1_1)*pred_mu2_1[x[1],]
  gamma2_0=(1-A1)*(1-A2)/((1-pred_pi1_1[x[2],])*(1-pred_pi2_0[x[2],]));gamma1_0=(1-A1)/(1-pred_pi1_1[x[2],])
  psi_0=gamma2_0*Y-(gamma1_0-1)*pred_mu1_0[x[1],]-(gamma2_0-gamma1_0)*pred_mu2_0[x[1],]
  psi=psi_1-psi_0-pred_theta_t[j]
  pred_sig2_t=c(pred_sig2_t,mean(psi^2))

  psi1 = psi_1 - pred_psi1_t[j]
  psi0 = psi_0 - pred_psi0_t[j]
  pred_sig2_psi1_t=c(pred_sig2_psi1_t,mean(psi1^2))
  pred_sig2_psi0_t=c(pred_sig2_psi0_t,mean(psi0^2))
}
# pred_theta_t=c(mean(Y[which(A1*A2==1)])-mean(Y[which(A1+A2==0)]),pred_theta_t)
# pred_theta_t=c(mean(Y[which(A1*A2==1)])-mean(Y[which(A1+A2==0)]),pred_theta_t)
# pred_sig2_t=c(var(Y[which(A1*A2==1)])*sum(A1*A2==1)/N+var(Y[which(A1+A2==0)])*sum(A1+A2==0)/N,pred_sig2_t)
c(pred_theta=pred_theta_t,pred_sig2=pred_sig2_t, pred_psi1_t, pred_sig2_psi1_t, pred_psi0_t, pred_sig2_psi0_t)  
