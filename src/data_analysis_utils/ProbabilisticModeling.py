import scipy
from scipy import special
import numpy as np
import pandas as pd




#SCIPY_ARRAY_API=1


class ProbabilisticModeling:
    def __init__(self):
        self.fitted_normal_pdf=None     #(mean,std)
        self.fitted_uniform_pdf=None    #(lower,upper,mean,var,std)
        self.fitted_sparse_bernoulli=None  #(p_if_1,q)
        self.fitted_binomial_pmf=None   #(n, p_if_1, variance, q, mu) 
        self.fitted_poisson_pmf=None    #(lam,q)


    #this would be better if it was statistic driven: mean, std, range, distribution etc. as apposed to "values"
    def generate_vector(self,size:int,values:list,probas:list):
        """
        values should be mapped to values
        """
        return np.random.choice(values, size=size, p=probas)

    #===================================================
    # a vecorized application of normal PDF 
    def fit_normal(self,vector):
        vector=np.asarray(vector,'float')
        self.fitted_normal_pdf=(vector.mean(),vector.std(ddof=0))#       
        
    def predict_normal_pdf(self,vector):
        vector=np.asarray(vector,'float')
        mean,std=self.fitted_normal_pdf[0],self.fitted_normal_pdf[1]
        res = (1/(std*np.sqrt(2*np.pi))) * (np.e**( -1*(((np.asarray(vector)-mean)**2)/(2*std**2)) )) 
        return res
    
    def predict_normal_cdf(self,vector_):
        """
        """
        vector=np.asarray(vector_.copy(),'float')
        calculation=(vector - self.fitted_normal_pdf[0]) / (self.fitted_normal_pdf[1] * np.sqrt(2))
        return 0.5 * (1 + scipy.special.erf(calculation))
    
    #===================================================    
    def fit_uniform(self,vector):
        vector=np.asarray(vector,'float')
        lower,upper=vector.min(),vector.max()
        mean=(lower+upper)/2
        var,std=((upper-lower)**2)/12,  (upper-lower)/np.sqrt(12)
        self.fitted_uniform_pdf=(lower,upper,mean,var,std)

    def predict_uniform_pdf(self,vector_):
        vector=np.asarray(vector_.copy(),'float')  

        #identify indices for targeted vectorized operations
        # indices to identify P=0 
        indices_that_are_out_of_bounds = (vector < self.fitted_uniform_pdf[0]) | (vector > self.fitted_uniform_pdf[1])
        # indices to calculate P
        indices_that_are_in_bounds = (vector >= self.fitted_uniform_pdf[0]) & (vector <= self.fitted_uniform_pdf[1])

        try:
            global jax_
            if jax_==True:
                # set the indices to the P
                vector=vector.at[indices_that_are_out_of_bounds].set(0)
                vector=vector.at[indices_that_are_in_bounds].set(1 / (self.fitted_uniform_pdf[1][indices_that_are_in_bounds] - self.fitted_uniform_pdf[0][indices_that_are_in_bounds]))
            else:
                # set the indices to the P
                vector[indices_that_are_out_of_bounds]=0
                vector[indices_that_are_in_bounds]=1 / (self.fitted_uniform_pdf[1][indices_that_are_in_bounds] - self.fitted_uniform_pdf[0][indices_that_are_in_bounds])
        except:
            # set the indices to the P
            vector[indices_that_are_out_of_bounds]=0
            vector[indices_that_are_in_bounds]=1 / (self.fitted_uniform_pdf[1][indices_that_are_in_bounds] - self.fitted_uniform_pdf[0][indices_that_are_in_bounds])
        return vector
    
    def predict_uniform_cdf(self,vector_):
        """
        predicts P of a random variable less than vector
        """
        vector = np.asarray(vector_.copy(),'float')
        lower, upper, _, _, _ =self.fitted_uniform_pdf    #(lower,upper,mean,var,std)

        # zero probability indices
        zero_proba_indices = vector < lower
        # 1 probability indices
        one_proba_indices = vector >= upper
        # 0<P<1 indices
        inbounds_proba_indices = (vector<upper)&(vector>=lower)

        try:
            global jax_
            if jax_==True:
                # set indices to P
                vector=vector.at[zero_proba_indices].set(0)
                vector=vector.at[one_proba_indices].set(1)
                vector=vector.at[inbounds_proba_indices].set((vector[inbounds_proba_indices] - lower) / (upper - lower))
            else:
                # set indices to P
                vector[zero_proba_indices]=0
                vector[one_proba_indices]=1
                vector[inbounds_proba_indices]= (vector[inbounds_proba_indices] - lower) / (upper - lower)
        except:
            # set indices to P
            vector[zero_proba_indices]=0
            vector[one_proba_indices]=1
            vector[inbounds_proba_indices]= (vector[inbounds_proba_indices] - lower) / (upper - lower)
        
        """vector = np.where(vector < lower, 0,
                            np.where(vector < upper, (vector - lower) / (upper - lower), 1))"""
        return vector

    
    #===================================================     
    def fit_sparse_bernoulli(self,vector_):
        """ where vector is an array of ones and zeros"""
        vector=np.asarray(vector_.copy(),'float') 
        #################################################   hadle classification into ones and 0s, and raise errors for >2 classes
        p_if_1=np.sum(vector)/len(vector)
        p_if_0=1-p_if_1
        q=np.sqrt( p_if_1*p_if_0 )
        self.fitted_sparse_bernoulli=(p_if_1,q)
        
    def predict_sparse_bernoulli_pmf(self,vector_):
        """ where vector is an array of ones and zeros or True/False"""
        vector=np.asarray(vector_.copy(),'float')
        # edge cases
        invalid=((vector!=0)&(vector!=1))|((vector!=0)&(vector!=1))
        if invalid.any():
            raise ValueError(f"Values must be either in (1,0) or (True,False)")  
        # identify positive and negative instances
        positive_instance = (vector==1)|(vector==True)
        negative_instance = (vector==0 ) | (vector==False)      

        try:
            global jax_
            if jax_==True:
                # retrieve probabilities
                vector=vector.at[positive_instance].set(self.fitted_sparse_bernoulli[0][positive_instance])
                vector=vector.at[negative_instance].set(1-self.fitted_sparse_bernoulli[0][negative_instance])
            else:
                # retrieve probabilities
                vector[positive_instance]=self.fitted_sparse_bernoulli[0][positive_instance]
                vector[negative_instance]=1-self.fitted_sparse_bernoulli[0][negative_instance]
        except:
            # retrieve probabilities
            vector[positive_instance]=self.fitted_sparse_bernoulli[0][positive_instance]
            vector[negative_instance]=1-self.fitted_sparse_bernoulli[0][negative_instance]
        return vector
    
    #===================================================  
    def fit_binomial(self,n_trials_vector,k_successes_vector):
        """
        First input vector == n_number of trials 
        Second == k_number of successes
        vectors can be of shape (-1,1) or (-1,) and contain numeric data
        """     
        n_trials_vector,k_successes_vector=np.array(n_trials_vector,dtype='float'),np.asarray(k_successes_vector, dtype='float')
        if (n_trials_vector<k_successes_vector).any():
            raise ValueError(f"Number of Trials must be <= Number of Successes")
        n=n_trials_vector.reshape(-1,1)
        successes=k_successes_vector.reshape(-1,1)
        p_if_1=successes/n
        p_if_1=p_if_1.reshape(-1,1)
        p_if_0=1-p_if_1
        p_if_0=p_if_0.reshape(-1,1)
        variance=n*p_if_1*p_if_0
        q=np.sqrt( n*p_if_1*p_if_0 )
        mu=n*p_if_1   
        self.fitted_binomial_pmf=(n, p_if_1, variance, q, mu)  
 
    def predict_binomial_pmf(self,vector_):
        """
        where the input is k number of successes for each element in the input vector
        and output is probability of k number of successes
        for mean, call self.fitted_binomial_pmf[5]
        """
        vector = np.asarray(vector_.copy(),'float')
        vector = vector.reshape(-1,1)
        n, p_if_1= self.fitted_binomial_pmf[:2]
        p_if_0=1-p_if_1
    
        eps=1e-15 
        try:
            global jax_
            if jax_ is True:
                log_coeff = jsp.scipy.special.gammaln(n + 1) - jsp.scipy.special.gammaln(vector + 1) - jsp.scipy.special.gammaln(n - vector + 1)
            else:
                log_coeff = scipy.special.gammaln(n + 1) - scipy.special.gammaln(vector + 1) - scipy.special.gammaln(n - vector + 1)
        except:
                log_coeff = scipy.special.gammaln(n + 1) - scipy.special.gammaln(vector + 1) - scipy.special.gammaln(n - vector + 1)

        log_pmf = log_coeff + vector * np.log(np.clip(p_if_1,eps,1)) + (n - vector) * np.log(np.clip(1 - p_if_1,eps,1))
        return np.exp(log_pmf)
        


    def predict_binomial_cdf(self,num_successes_vector_,return_as_steps_list=False,zero_padded=True):
        """
        takes a vector of hypothetical k_number of successes, and returns P(x<=k) based on fitted parameters of the binomial distribution
        return_as_steps_list=False,zero_padded=t/f --> for cases when P@ each sep is preffered over sum of all P. 
        if zero_padded, valuse after k are padded, esle lsits are sliced iteratively. zero_padded is set to True for speed, but would be more memory efficient as False
        """

        if self.fitted_binomial_pmf is None:
            raise ValueError("This model instance is not fitted yet. Call fit_binomial(observed_trials,  observed_successes).")  

        #retrieve class objects
        n, p_if_1 =self.fitted_binomial_pmf[:2]   #(n, p_if_1, variance, q, mu) 
        p_if_0 = 1- p_if_1    

        # reshape input
        num_successes_vector_=np.asarray(num_successes_vector_).reshape(-1,1)        

        if ((n-num_successes_vector_)<0).any():
            raise ValueError(f"Instance(s) of number of successes > trials detected.")
        # create padded matrix

        try:
            global jax_
            if jax_ is True: 
                max_successes=num_successes_vector_.max()        
                padded_matrix_of_cdf_k_successes=np.zeros((n.shape[0],int(max_successes+1)))
                for i in range(0,int(max_successes+1)):
                    padded_matrix_of_cdf_k_successes.at[:,i].set(i)
                padded_matrix_of_cdf_k_successes=padded_matrix_of_cdf_k_successes.astype(float)
                mask = padded_matrix_of_cdf_k_successes>num_successes_vector_               
                padded_matrix_of_cdf_k_successes.at[mask].set(np.nan)
            else:
                max_successes=num_successes_vector_.max()        
                padded_matrix_of_cdf_k_successes=np.zeros((n.shape[0],int(max_successes+1)))
                for i in range(0,int(max_successes+1)):
                    padded_matrix_of_cdf_k_successes[:,i]=i
                padded_matrix_of_cdf_k_successes=padded_matrix_of_cdf_k_successes.astype(float)
                mask = padded_matrix_of_cdf_k_successes>num_successes_vector_
                padded_matrix_of_cdf_k_successes[mask]=np.nan
        except:
                max_successes=num_successes_vector_.max()        
                padded_matrix_of_cdf_k_successes=np.zeros((n.shape[0],int(max_successes+1)))
                for i in range(0,int(max_successes+1)):
                    padded_matrix_of_cdf_k_successes[:,i]=i
                padded_matrix_of_cdf_k_successes=padded_matrix_of_cdf_k_successes.astype(float)
                mask = padded_matrix_of_cdf_k_successes>num_successes_vector_
                padded_matrix_of_cdf_k_successes[mask]=np.nan

        eps=1e-15
        try:
            if jax_ is True:
                log_coeff = jsp.scipy.special.gammaln(n + 1) - jsp.scipy.special.gammaln(padded_matrix_of_cdf_k_successes + 1) - jsp.scipy.special.gammaln(n - padded_matrix_of_cdf_k_successes + 1)              
            else:
                log_coeff = scipy.special.gammaln(n + 1) - scipy.special.gammaln(padded_matrix_of_cdf_k_successes + 1) - scipy.special.gammaln(n - padded_matrix_of_cdf_k_successes + 1)
        except:
                log_coeff = scipy.special.gammaln(n + 1) - scipy.special.gammaln(padded_matrix_of_cdf_k_successes + 1) - scipy.special.gammaln(n - padded_matrix_of_cdf_k_successes + 1)
        
        log_pmf = log_coeff + padded_matrix_of_cdf_k_successes * np.log(np.clip(p_if_1,eps,1)) + (n - padded_matrix_of_cdf_k_successes) * np.log(np.clip(1 - p_if_1,eps,1))
        result_vector= np.exp(log_pmf)  

        if return_as_steps_list==True:            
            if zero_padded==True: 
                nan_values = np.isnan(result_vector)            
                try:
                    if jax_ is True:
                        result_vector=result_vector.at[nan_values].set(0)
                    else:
                        result_vector[nan_values] = 0
                except:
                    result_vector[nan_values] = 0
                return result_vector.tolist()
            else:
                lists=result_vector.tolist() 
                slice_points=num_successes_vector_.ravel()
                return [row[:int(round(length))+1] for row,length in zip(lists,slice_points)]
        return np.sum(result_vector,axis=1) 

       


    #====================================================================================
    def fit_poisson(self, units_of_space, occurances_trhought_all_units ):
        """ args: (total number of periods,     total number of occurances)"""
        units_of_space, occurances_trhought_all_units = np.asarray(units_of_space,'float'),np.asarray(occurances_trhought_all_units,'float')
        lam=occurances_trhought_all_units/units_of_space
        lam=lam.reshape(-1,1)
        q=np.sqrt(lam)
        self.fitted_poisson_pmf=(lam,q)


    def predict_poisson_pmf(self, vector_):
        lam=self.fitted_poisson_pmf[0]
        vector = np.asarray(vector_.copy(), dtype='float').reshape(-1,1)
        try:
            global jax_
            if jax_ is True:
                result= np.exp ( vector * np.log(lam) - lam - jsp.scipy.special.gammaln(vector + 1) )         
            else:
                result= np.exp ( vector * np.log(lam) - lam - scipy.special.gammaln(vector + 1) )
        except:
                result= np.exp ( vector * np.log(lam) - lam - scipy.special.gammaln(vector + 1) )
        return result
    

    def predict_poisson_cdf(self, upper_bound_vector_, return_as_steps_list=False,zero_padded=True,lower_bound_vector_=None):
        """
        lower bound default is 0
        upper bound vectors is inclusive    
        return_as_steps_list=False,zero_padded=t/f --> for cases when P@ each sep is preffered over sum of all P. 
        if zero_padded, valuse after k are padded, esle lsits are sliced iteratively. zero_padded is set to True for speed, but would be more memory efficient as False    
        """
        lam=self.fitted_poisson_pmf[0]

        upper_bound_vector  =  np.asarray(upper_bound_vector_.copy(),float).reshape(-1,1)
        if lower_bound_vector_ is not None:
            lower_bound_vector = np.asarray(lower_bound_vector_.copy(),float).reshape(-1,1)
        else:
            lower_bound_vector=np.zeros(upper_bound_vector.shape)

        #create a matrix where there may be values over the upper_bound
        max_range= (upper_bound_vector - lower_bound_vector).max()
        if max_range%1>0:
            max_range=(max_range//1)+1
        max_range=int(max_range)
        for interval in range(max_range):
            lower_bound_vector=np.hstack((lower_bound_vector,lower_bound_vector[:,-1:]+1))
        # cap values at upper bound by casting to np.nan
        maxes = lower_bound_vector>upper_bound_vector
        global jax_
        try:            
            if jax_==True:
                # retrieve probabilities
                lower_bound_vector=lower_bound_vector.at[maxes].set(np.nan)
            else:
                # retrieve probabilities
                lower_bound_vector[maxes]=np.nan
        except:
            # retrieve probabilities
            lower_bound_vector[maxes]=np.nan

        try:
            if jax_ is True:
                result= np.exp ( lower_bound_vector * np.log(lam) - lam - jsp.scipy.special.gammaln(lower_bound_vector + 1) )          
            else:
                result= np.exp ( lower_bound_vector * np.log(lam) - lam - scipy.special.gammaln(lower_bound_vector + 1) )
        except:
                result= np.exp ( lower_bound_vector * np.log(lam) - lam - scipy.special.gammaln(lower_bound_vector + 1) )

        if return_as_steps_list==True:            
            if zero_padded==True: 
                nan_values = np.isnan(result)            
                try:
                    if jax_ is True:
                        result=result.at[nan_values].set(0)
                    else:
                        result[nan_values] = 0
                except:
                    result[nan_values] = 0
                return result.tolist()
            else:
                lists=result.tolist() 
                slice_points=upper_bound_vector.ravel()
                return [row[:int(round(length))+1] for row,length in zip(lists,slice_points)]
        return np.sum(result,axis=1) 
        


        