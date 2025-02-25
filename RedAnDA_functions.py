


# !/usr/bin/env python

""" AnDA_stat_functions.py: Collection of statistical functions used in AnDA. """

__author__ = "Pierre Tandeo and Phi Huynh Viet"
__version__ = "1.0"
__date__ = "2016-10-16"
__maintainer__ = "Pierre Tandeo"
__email__ = "pierre.tandeo@telecom-bretagne.eu"

import numpy as np
from sklearn.neighbors import KDTree
from numpy.linalg import pinv


def AnDA_RMSE(a,b):
    """ Compute the Root Mean Square Error between 2 n-dimensional vectors. """
 
    return np.sqrt(np.mean((a-b)**2))

def normalise(M):
    """ Normalize the entries of a multidimensional array sum to 1. """

    c = np.sum(M);
    # Set any zeros to one before dividing
    d = c + 1*(c==0);
    M = M/d;
    return M;

def mk_stochastic(T):
    """ Ensure the matrix is stochastic, i.e., the sum over the last dimension is 1. """

    if len(T.shape) == 1:
        T = normalise(T);
    else:
        n = len(T.shape);
        # Copy the normaliser plane for each i.
        normaliser = np.sum(T,n-1);
        normaliser = np.dstack([normaliser]*T.shape[n-1])[0];
        # Set zeros to 1 before dividing
        # This is valid since normaliser(i) = 0 iff T(i) = 0

        normaliser = normaliser + 1*(normaliser==0);
        T = T/normaliser.astype(float);
    return T;

def sample_discrete(prob, r, c):
    """ Sampling from a non-uniform distribution. """

    # this speedup is due to Peter Acklam
    cumprob = np.cumsum(prob);
    n = len(cumprob);
    R = np.random.rand(r,c);
    M = np.zeros([r,c]);
    for i in range(0,n-1):
        M = M+1*(R>cumprob[i]);    
    return int(M)

def resampleMultinomial(w):
    """ Multinomial resampler. """

    M = np.max(w.shape);
    Q = np.cumsum(w,0);
    Q[M-1] = 1; # Just in case...
    i = 0;
    indx = [];
    while (i<=(M-1)):
        sampl = np.random.rand(1,1);
        j = 0;
        while (Q[j]<sampl):
            j = j+1;
        indx.append(j);
        i = i+1
    return indx

def inv_using_SVD(Mat, eigvalMax):
    """ SVD decomposition of Matrix. """
    
    U,S,V = np.linalg.svd(Mat, full_matrices=True);
    eigval = np.cumsum(S)/np.sum(S);
    # search the optimal number of eigen values
    i_cut_tmp = np.where(eigval>=eigvalMax)[0];
    S = np.diag(S);
    V = V.T;
    i_cut = np.min(i_cut_tmp)+1;
    U_1 = U[0:i_cut,0:i_cut];
    U_2 = U[0:i_cut,i_cut:];
    U_3 = U[i_cut:,0:i_cut];
    U_4 = U[i_cut:,i_cut:];
    S_1 = S[0:i_cut,0:i_cut];
    S_2 = S[0:i_cut,i_cut:];
    S_3 = S[i_cut:,0:i_cut];
    S_4 = S[i_cut:,i_cut:];
    V_1 = V[0:i_cut,0:i_cut];
    V_2 = V[0:i_cut,i_cut:];
    V_3 = V[i_cut:,0:i_cut];
    V_4 = V[i_cut:,i_cut:];
    tmp1 = np.dot(np.dot(V_1,np.linalg.inv(S_1)),U_1.T);
    tmp2 = np.dot(np.dot(V_1,np.linalg.inv(S_1)),U_3.T);
    tmp3 = np.dot(np.dot(V_3,np.linalg.inv(S_1)),U_1.T);
    tmp4 = np.dot(np.dot(V_3,np.linalg.inv(S_1)),U_3.T);
    inv_Mat = np.concatenate((np.concatenate((tmp1,tmp2),axis=1),np.concatenate((tmp3,tmp4),axis=1)),axis=0);
    tmp1 = np.dot(np.dot(U_1,S_1),V_1.T);
    tmp2 = np.dot(np.dot(U_1,S_1),V_3.T);
    tmp3 = np.dot(np.dot(U_3,S_1),V_1.T);
    tmp4 = np.dot(np.dot(U_3,S_1),V_3.T);
    hat_Mat = np.concatenate((np.concatenate((tmp1,tmp2),axis=1),np.concatenate((tmp3,tmp4),axis=1)),axis=0);
    det_inv_Mat = np.prod(np.diag(S[0:i_cut,0:i_cut]));   
    return inv_Mat;






def AnDA_analog_forecasting(x, AF):
    """ Apply the analog method on catalog of historical data to generate forecasts. """
    
    # initializations
    N, n = x.shape
    xf = np.zeros([N,n])
    xf_mean = np.zeros([N,n])
    stop_condition = 0
    i_var = np.array([0])
    
    # local or global analog forecasting
    while (stop_condition !=1):

        # in case of global approach
        if np.all(AF.neighborhood == 1):

            i_var_neighboor = np.arange(n,dtype=np.int64)
            i_var = np.arange(n, dtype=np.int64)
            stop_condition = 1

        # in case of local approach
        else:
            i_var_neighboor = np.where(AF.neighborhood[int(i_var),:]==1)[0]
            
        # find the indices and distances of the k-nearest neighbors (knn)
        kdt = KDTree(AF.catalog.analogs[:,i_var_neighboor], leaf_size=50, metric='euclidean')

        dist_knn, index_knn = kdt.query(x[:,i_var_neighboor], AF.k)
        
        #print(index_knn,end='\r')
        # parameter of normalization for the kernels
        lambdaa = np.median(dist_knn)

        # compute weights
        if AF.k == 1:
            weights = np.ones([N,1])
        else:
            weights = mk_stochastic(np.exp(-np.power(dist_knn,2)/lambdaa**2))
        
        # for each member/particle
        for i_N in range(0,N):
            
            # initialization
            xf_tmp = np.zeros([AF.k,np.max(i_var)+1])
            
            # method "locally-constant"
            if (AF.regression == 'locally_constant'):
                
                # compute the analog forecasts
                xf_tmp[:,i_var] = AF.catalog.successors[np.ix_(index_knn[i_N,:],i_var)]
                
                # weighted mean and covariance
                xf_mean[i_N,i_var] = np.sum(xf_tmp[:,i_var]*np.repeat(weights[i_N,:][np.newaxis].T,len(i_var),1),0)
                E_xf = (xf_tmp[:,i_var]-np.repeat(xf_mean[i_N,i_var][np.newaxis],AF.k,0)).T
                cov_xf = 1.0/(1.0-np.sum(np.power(weights[i_N,:],2)))*np.dot(np.repeat(weights[i_N,:][np.newaxis],len(i_var),0)*E_xf,E_xf.T)

            # method "locally-incremental"
            elif (AF.regression == 'increment'):
                
                # compute the analog forecasts
                xf_tmp[:,i_var] = np.repeat(x[i_N,i_var][np.newaxis],AF.k,0) + AF.catalog.successors[np.ix_(index_knn[i_N,:],i_var)]-AF.catalog.analogs[np.ix_(index_knn[i_N,:],i_var)]
                
                # weighted mean and covariance
                xf_mean[i_N,i_var] = np.sum(xf_tmp[:,i_var]*np.repeat(weights[i_N,:][np.newaxis].T,len(i_var),1),0)
                E_xf = (xf_tmp[:,i_var]-np.repeat(xf_mean[i_N,i_var][np.newaxis],AF.k,0)).T
                cov_xf = 1.0/(1-np.sum(np.power(weights[i_N,:],2)))*np.dot(np.repeat(weights[i_N,:][np.newaxis],len(i_var),0)*E_xf,E_xf.T)

            # method "locally-linear"
            elif (AF.regression == 'local_linear'):
         
                # define analogs, successors and weights
                X = AF.catalog.analogs[np.ix_(index_knn[i_N,:],i_var_neighboor)]                
                Y = AF.catalog.successors[np.ix_(index_knn[i_N,:],i_var)]                
                w = weights[i_N,:][np.newaxis]
                
                # compute centered weighted mean and weighted covariance
                Xm = np.sum(X*w.T, axis=0)[np.newaxis]
                Xc = X - Xm
                
                # regression on principal components
                Xr   = np.c_[np.ones(X.shape[0]), Xc]
                Cxx  = np.dot(w    * Xr.T,Xr)
                Cxx2 = np.dot(w**2 * Xr.T,Xr)
                Cxy  = np.dot(w    * Y.T, Xr)
                inv_Cxx = pinv(Cxx, rcond=0.01) # in case of error here, increase the number of analogs (AF.k option)
                beta = np.dot(inv_Cxx,Cxy.T)
                X0 = x[i_N,i_var_neighboor]-Xm
                X0r = np.c_[np.ones(X0.shape[0]),X0]
                 
                # weighted mean
                xf_mean[i_N,i_var] = np.dot(X0r,beta)
                pred = np.dot(Xr,beta)
                res = Y-pred
                xf_tmp[:,i_var] = xf_mean[i_N,i_var] + res
    
                # weigthed covariance
        
                cov_xfc = np.dot(w * res.T,res)/(1-np.trace(np.dot(Cxx2,inv_Cxx)))
                cov_xf = cov_xfc*(1+np.trace(Cxx2@inv_Cxx@X0r.T@X0r@inv_Cxx))
                
                if (cov_xf==np.inf).any():
                    cov_xf = np.zeros_like(cov_xf)
#                 elif (cov_xf==np.nan).any():
#                     cov_xf = np.zeros_like(cov_xf)

#                 cov_xf=np.where(cov_xf!=np.inf,cov_xf,0)
#                 if (1-np.trace(np.dot(Cxx2,inv_Cxx)))==0:
#                     np.save('unbug',[X,Y,w,x,i_N,i_var_neighboor],allow_pickle=True)


                # constant weights for local linear
                weights[i_N,:] = 1.0/len(weights[i_N,:])

            # method "Dynamical Mode Decomposition"
            elif (AF.regression == 'DMD'):             
            
                # define analogs, successors and weights
                X = AF.catalog.analogs[np.ix_(index_knn[i_N,:],i_var_neighboor)]                
                Y = AF.catalog.successors[np.ix_(index_knn[i_N,:],i_var)]                
                w = weights[i_N,:][np.newaxis]
                
                Oper = (Y.dot(np.linalg.pinv(X)))
                
                xf_mean[i_N,i_var] = (Y.T.dot(np.linalg.pinv(X.T))).dot(x[i_N,i_var_neighboor])
                
                
            # error
            else:
                raise ValueError("""\
                    Error: choose AF.regression between \
                    'locally_constant', 'increment', 'local_linear' """)
            
            '''
            # method "globally-linear" (to finish)
            elif (AF.regression == 'global_linear'):
                ### REMARK: USE i_var_neighboor IN THE FUTURE! ####
                xf_mean[i_N,:] = AF.global_linear.predict(np.array([x[i_N,:]]))
                if n==1:
                    cov_xf = np.cov((AF.catalog.successors - AF.global_linear.predict(AF.catalog.analogs)).T)[np.newaxis][np.newaxis]
                else:
                    cov_xf = np.cov((AF.catalog.successors - AF.global_linear.predict(AF.catalog.analogs)).T)
            
            # method "locally-forest" (to finish)
            elif (AF.regression == 'local_forest'):
                ### REMARK: USE i_var_neighboor IN THE FUTURE! #### 
                xf_mean[i_N,:] = AF.local_forest.predict(np.array([x[i_N,:]]))
                if n==1:
                    cov_xf = np.cov(((AF.catalog.successors - np.array([AF.local_forest.predict(AF.catalog.analogs)]).T).T))[np.newaxis][np.newaxis]
                else:
                    cov_xf = np.cov((AF.catalog.successors - AF.local_forest.predict(AF.catalog.analogs)).T)
                # weighted mean and covariance
                #xf_tmp[:,i_var] = AF.local_forest.predict(AF.catalog.analogs[np.ix_(index_knn[i_N,:],i_var)]);
                #xf_mean[i_N,i_var] = np.sum(xf_tmp[:,i_var]*np.repeat(weights[i_N,:][np.newaxis].T,len(i_var),1),0)
                #E_xf = (xf_tmp[:,i_var]-np.repeat(xf_mean[i_N,i_var][np.newaxis],AF.k,0)).T;
                #cov_xf = 1.0/(1.0-np.sum(np.power(weights[i_N,:],2)))*np.dot(np.repeat(weights[i_N,:][np.newaxis],len(i_var),0)*E_xf,E_xf.T);
            '''
            
            # Gaussian sampling
            if (AF.sampling =='gaussian'):
                # random sampling from the multivariate Gaussian distribution
                xf[i_N,i_var] = np.random.multivariate_normal(xf_mean[i_N,i_var],cov_xf)
                
            
            # Multinomial sampling
            elif (AF.sampling =='multinomial'):
                # random sampling from the multinomial distribution of the weights
                i_good = sample_discrete(weights[i_N,:],1,1)
                xf[i_N,i_var] = xf_tmp[i_good,i_var]
            
            # error
            else:
                raise ValueError("""\
                    Error: choose AF.sampling between 'gaussian', 'multinomial' 
                """)

        # stop condition
        if (np.array_equal(i_var,np.array([n-1])) or (len(i_var) == n)):
            stop_condition = 1;
             
        else:
            i_var = i_var + 1
            
    return xf, xf_mean; # end




