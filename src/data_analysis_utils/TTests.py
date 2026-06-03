import pandas as pd
import numpy as np
import scipy
from scipy import stats
from scipy.stats import t

import itertools
from itertools import combinations



class TTests:
    def __init__(self):
        self.two_col_t_test_overview=None

    """"
    #students two sample t test:
    t=(xbar1-xbar2) / sqrt(s2p((1/n1)+(1/n2)))    
    s2p is pooled variance
    s2p=( (n1-1)s1**2 + (n2-2)s2**2 ) / dof       
    dof = n1+n2-2
    stats.ttest_ind(vector1,vector2)  

    #two sample z  
    z= ((xbar1-xbar2)-(mu1-mu2)) / sqrt( (q1**2/n1)+(q2**2/n2) 

    #welch's t
    t=(x1bar - x2bar)  /  sqrt(s12/n1 + s22/n2)
    dof = (s12/n1 + s22/n2)2 / { [ (s12 / n1)2 / (n1 - 1) ] + [ (s22 / n2)2 / (n2 - 1) ] }
    """


    def students_t_test(self, two_col_numnum_df: pd.DataFrame):
        """
        takes a dataframe with two numeric columns
        Perform Student's two-tailed t-test (equal variances).
        Returns t_stat, dof, p_value.
        """
        x1, x2 = two_col_numnum_df.columns[0], two_col_numnum_df.columns[1]
        x1s2 = two_col_numnum_df[x1].var(ddof=1)
        x2s2 = two_col_numnum_df[x2].var(ddof=1)
        x1n = two_col_numnum_df[x1].count()
        x2n = two_col_numnum_df[x2].count()    
        dof = x1n + x2n - 2
        sp2 = ( ((x1n - 1) * x1s2) +  ((x2n - 1) * x2s2)  ) / dof
        t_stat = (two_col_numnum_df[x1].mean() - two_col_numnum_df[x2].mean()) / np.sqrt( sp2 * (1 / x1n + 1 / x2n) )
        p_val = 2 * scipy.stats.t.sf(abs(t_stat), dof)
        return p_val


    def students_t_all_pairs(self, df: pd.DataFrame):
        """
        Vectorized all-vs-all Student's t-test (equal variances).

        Returns
        -------
        t_stat : (p, p) ndarray
        dof    : (p, p) ndarray
        p_val  : (p, p) ndarray
        """
        X = df.to_numpy(dtype=float)

        # column-wise stats
        means = np.nanmean(X, axis=0)
        vars_ = np.nanvar(X, axis=0, ddof=1)
        ns = np.sum(~np.isnan(X), axis=0)

        # broadcasted differences
        mean_diff = means[:, None] - means[None, :]

        # pooled variance
        sp2 = (
            (ns[:, None] - 1) * vars_[:, None] +
            (ns[None, :] - 1) * vars_[None, :]
        ) / (ns[:, None] + ns[None, :] - 2)

        denom = np.sqrt(sp2 * (1 / ns[:, None] + 1 / ns[None, :]))

        t_stat = mean_diff / denom

        dof = ns[:, None] + ns[None, :] - 2

        p_val = 2 * scipy.stats.t.sf(np.abs(t_stat), dof)

        p_frame = pd.DataFrame(p_val,columns=df.columns,index=df.columns).unstack().reset_index(drop=False).rename(columns={'level_0':'numeric_1','level_1':'numeric_2',0:'P-value'})
        p_frame = p_frame.loc[p_frame['numeric_1']!=p_frame['numeric_2']].reset_index(drop=True)
        return p_frame



    def welchs_t_test(self,two_col_numnum_df:pd.DataFrame):
        """
        takes a df with two numeric columns
        perform welch's two-tailed welches t_test on x1,x2
        return a p-value
        """
        x1, x2 = two_col_numnum_df.columns[0], two_col_numnum_df.columns[1]
        x1s12,x2s22 = two_col_numnum_df[x1].var(ddof=1), two_col_numnum_df[x2].var(ddof=1)
        x1n, x2n = two_col_numnum_df[x1].count(), two_col_numnum_df[x2].count()
        x1var_over_n,  x2var_over_n = x1s12/x1n, x2s22/x2n
        x1var_over_n_plus_x2var_over_n = x1var_over_n  +  x2var_over_n
        t_stat = (two_col_numnum_df[x1].mean()-two_col_numnum_df[x2].mean()) / np.sqrt(x1var_over_n_plus_x2var_over_n)
        dof = x1var_over_n_plus_x2var_over_n**2 / ( ( ((x1s12/x1n)**2) / (x1n-1) )+( ((x2s22/x2n)**2) / (x2n-1) ) )
        p_val = 2 * scipy.stats.t.sf( abs(t_stat),  dof  )
        return p_val

    def welchs_t_all_pairs(self,df: pd.DataFrame):
        """
        takes a dataframe of numeric values and returns p-valuse for every column combination
        """
        X = df.to_numpy(dtype=float)

        means = np.nanmean(X, axis=0)
        vars_ = np.nanvar(X, axis=0, ddof=1)
        ns = np.sum(~np.isnan(X), axis=0)  # sums bool vals, like .shape[0], only doesn't include NaNs

        mu_diff = means[:, None] - means[None, :]
        v_over_n = vars_ / ns

        denom = np.sqrt(v_over_n[:, None] + v_over_n[None, :])  # swhere None is like np.newaxis and impacts broadcasting rulse such that shapes (n,1) and (1,n) become (n,n)
        t_stat = mu_diff / denom

        dof = (v_over_n[:, None] + v_over_n[None, :]) ** 2 / (
            (v_over_n[:, None] ** 2) / (ns[:, None] - 1) +
            (v_over_n[None, :] ** 2) / (ns[None, :] - 1)
        )

        p_val = 2 * scipy.stats.t.sf(np.abs(t_stat), dof)
        p_frame = pd.DataFrame(p_val,columns=df.columns,index=df.columns).unstack().reset_index(drop=False).rename(columns={'level_0':'numeric_1','level_1':'numeric_2',0:'P-value'})
        p_frame = p_frame.loc[p_frame['numeric_1']!=p_frame['numeric_2']].reset_index(drop=True)
        return p_frame
            

    def target_features_t_test_to_rest(self,data:pd.DataFrame,target:str|list,method:str='welch'):
        """
        where method can be of ('welch','student')
        Iterates through all input dataframe columns and caculates the t test between target column(s)
        returns an unsorted dataframe data[number_1, number_2, Correlation]
        """
        if isinstance(target,str):
            target=[target]
        cols=data.columns
        p_vals=[]
        targs=[]
        y_vars=[]
        for targ in target:
            for col in cols:            
                if col==targ:
                    continue
                if method=='welch':
                    pval=self.welchs_t_test(data[[targ,col]])
                elif method=='student':
                    pval=self.students_t_test(data[[targ,col]])
                else:
                    raise ValueError("Unrecognized method. Acceptable are method='welch' or method='student'.")                
                p_vals.append(pval)
                targs.append(targ)
                y_vars.append(col)
        if len(p_vals)<1:
            return pd.DataFrame(columns=['numeric_1','numeric_2','P-value'])
        return pd.DataFrame({'numeric_1':targs,'numeric_2':y_vars,'P-value':p_vals}) 

    def test_all_num_num_t_test(self,data, numeric_columns:str|list|None=None,target:list|str|None=None,t_test_method:str='welch'):
        """
        t_test_method can be of ('sudent','welch')
        """
        # determine base columns based on numeric_columns
        if numeric_columns == None: 
            numeric_columns=list(data.select_dtypes('number').columns)
        elif isinstance(numeric_columns,str):
            numeric_columns=[numeric_columns]     
        # initialize a dataframe to pass to other functions
        num_data=pd.DataFrame(data.copy())
        # look at target(s)
        if target is not None:
            if isinstance(target,str):
                target=[target]
            #pass the dataframe with targets and numeric_columns
            num_data = num_data[list(set(numeric_columns+target))].astype(float)
            # return a target based result
            return self.target_features_t_test_to_rest(num_data,target,t_test_method)
        # if not target
        else:
            #pass only numeric columns
            num_data = num_data[numeric_columns].astype(float)
            if t_test_method=='welch':
                num_data = self.welchs_t_all_pairs(num_data)
            elif t_test_method=='student':
                num_data = self.students_t_all_pairs(num_data)
            else:
                raise ValueError("Unrecognized t_test_method. Acceptable are t_test_method='welch' or t_test_method='student'.")  
        #map_base=pd.DataFrame(list(itertools.combinations(columns, 2)),columns=['numeric_1','numeric_2'])
        #num_data=map_base.merge(num_data,how='inner',on=['numeric_1','numeric_2'])
        return num_data
    
    
    def num_num_column_t_test_comparison(self,data, alpha:float=0.05,keep_above_p:bool|None=False,numeric_columns:str|list|None=None,target:list|str|None=None, t_test_method:str='welch'):
        """
        t_test_method can be of ('sudent','welch')
        takes alpha pvalue as a paremeter
        if keep similar==True, only p values>=alpha are returned if False, <alpha are returned
        """
        num_data=self.test_all_num_num_t_test(data=data,numeric_columns=numeric_columns,target=target,t_test_method=t_test_method)
        if keep_above_p==True:
            num_data=num_data.loc[np.abs(num_data['P-value'])>=alpha].reset_index(drop=True)
        elif keep_above_p==False: 
            num_data = num_data.loc[np.abs(num_data['P-value'])<alpha].reset_index(drop=True)
        #else don't filter
        return num_data
    
