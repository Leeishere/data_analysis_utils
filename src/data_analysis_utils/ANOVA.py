import pandas as pd
import scipy.stats
import warnings
import itertools
from itertools import combinations
import numpy as np
import inspect
from joblib import Parallel, delayed

#these use f.survival_function or t.survival_function) to calculate the p value
# scipy.stats.f.sf(f_score,dfn,dfd) where dfn is the numerator--mean_square(ie variance) and dfd is denominator-->error# this returns a p value for the f stat
#     #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html#scipy.stats.t
#     #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html#scipy.stats.f
#  AT THIS TIME:  tests that are specific to scipy.stats are used. Specifically in the case of assumption checking 
# DEBUG TOOL TO CHECK BOTTLE NECKS

class ANOVA:
    def __init__(self):
        self.two_way_interaction_columns=None
        self.two_way_interaction_sizes=None 

    # =========================================================================================================================================================================
    # 5 functions that check assumptions for ONE-WAY-ANOVA
    # 4 function that check assumptions for Kruskal Wallis tests
    # they can be called with parameter retrieve_meta==True, to produce in-depth meta results 
    # otherwise they return True/False/'Pseudo'
    # anova_assumption_checks() wraps multiple helper functions
    # kruskal_wallis_assumptions() wraps multiple helper as well
    # they are called/located in test_all_num_cat_kruskal_wallis() and test_all_num_cat_ANOVA() 
    # scipy.stats.shapiro() and scipy.stats.normaltest() are called in _anova_check_normality() and tests for normality, 
    # scipy.stats.shapiro() is exclusive to scipy.stats
    # scipy.stats.normaltest() is supported in other libraries and on gpu. 
    # scipy.stats.normaltest() impliments D'Agostino's K2
    # scipy.stats.ks_2samp() is called in _kruskal_assumptions_strict() and kruskal_assumptions_meta(), it is exclusive to scipy.stats
    # scipy.stats.levene() is supported in libraries such as cupy, jax, pytorch, but not for gpu

    
    # 1) Group size check
    def _anova_assumptions_check_group_sizes(self,
                                             group, 
                                             y, 
                                             min_n:int|None=None, 
                                             retrieve_meta:bool|None=None,
                                             dropna:bool|None=None):

        """
        Evaluate whether each group has sufficient sample size.
        Parameters
        ----------
        group : array-like (pd.Series)
            Categorical grouping variable.
        y : array-like (pd.Series)
            Numeric dependent variable.
        min_n : int, default=5
            Minimum required observations per group.

        Returns
        -------
        T/F Bool

        Notes
        -----
        Small group sizes reduce statistical power and can undermine
        ANOVA reliability. Recommended minimum is typically 5–10 per group.
        """
        if dropna is None:
            dropna=True
        if retrieve_meta is None:
            retrieve_meta=False
        if min_n is None:
            min_n=5

        df = pd.DataFrame({'group': group, 'y': y})

        if dropna:
            df = df.dropna()
        else:
            df = df.dropna(subset=['y'])
            df['group'] = df['group'].where(~df['group'].isna(), 'NaN')

        if df.empty:
            return False
        
        dropna=False

        counts = df.groupby('group',dropna=dropna,observed=True)['y'].size()
        if retrieve_meta == True:
            return counts.rename('count_obs')
        result =bool((counts >= min_n).all())
        return result  

    # 2) Normality per group (Shapiro-Wilk)
    def _anova_check_normality(self,
                               group, 
                               y, 
                               alpha:float|None=None,
                               retrieve_meta:bool|None=None,
                               dropna:bool|None=None):

        """
        Test normality of the dependent variable within each group
        using the Shapiro-Wilk  or D'Agustino K2 : scipy.stats.shapiro or scipy.stats.normaltest

        Parameters
        ----------
        group : array-like (pd.Series)
            Categorical grouping variable.
        y : array-like (pd.Series)
            Numeric dependent variable.
        alpha : float, default=0.05
            Significance level for hypothesis testing.

        Returns
        -------
        T/F Bool

        Notes
        -----
        The null hypothesis is that the data are normally distributed.
        ANOVA assumes approximate normality within groups. This test
        may be sensitive to large sample sizes.
        """

        if dropna==None:
            dropna=True
        if retrieve_meta==None:
            retrieve_meta=False
        if alpha==None:
            alpha=0.05
            
        if dropna==True:
            df = pd.DataFrame({'group': group, 'y': y}).dropna()
        else:
            df = pd.DataFrame({'group': group, 'y': y})
        dropna=False
 
        if retrieve_meta == True:
            def norm_test(vals):
                vals=vals.dropna()
                if len(vals) < 3:
                    return pd.Series({
                        'test': np.nan,
                        'stat': np.nan,
                        'p': np.nan,
                        'normal': False,
                        'note': 'n<3'
                    })
                if len(vals) < 300:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")  
                        stat, p = scipy.stats.shapiro(vals)   
                    """if w:
                        print(w[-1].message) """
                    return  pd.Series({
                            'test':'shapiro',
                            'stat': stat,
                            'p': p,
                            'normal': p>alpha,
                            'note': 'n>=3'
                        })
                else:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always") 
                        stat, p = scipy.stats.normaltest(vals)   
                    """if w:
                        print(w[-1].message) """
                    return  pd.Series({
                            'test':"dagostino",
                            'stat': stat,
                            'p': p,
                            'normal': p>alpha,
                            'note': 'n>=3'})
            return df.groupby('group',dropna=dropna,observed=True)['y'].apply(norm_test).rename('is_normal')
        
        def norm_test(vals):
                vals=vals.dropna()
                if len(vals) < 3:
                    return False
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always") 
                    if len(vals) < 300:
                        stat, p = scipy.stats.shapiro(vals)  # perfom test on a sample to avoid compute. computing corr in scipy.stats.shapiro() seems be an issue that slows perfomance
                        """if w:
                            print(w[-1].message)"""
                        return  p > alpha  
                    else:
                        stat, p = scipy.stats.normaltest(vals)  # perfom test on a sample to avoid compute. computing corr in scipy.stats.shapiro() seems be an issue that slows perfomance
                """if w:
                    print(w[-1].message)"""
                return  p > alpha             

        result = (df.groupby('group',dropna=dropna,observed=True)['y'].apply(norm_test)).all() 
        return result
    
    # 3) Homogeneity of variance (Levene – more robust than Bartlett)
    def _anova_check_homogeneity(self,
                                 group, 
                                 y, 
                                 alpha:float|None=None, 
                                 center:str|None=None, 
                                 retrieve_meta:bool|None=None,
                                 dropna:bool|None=None):
        """
        Test equality of variances across groups using Levene's test.

        Parameters
        ----------
        group : array-like (pd.Series)
            Categorical grouping variable.
        y : array-like (pd.Series)
            Numeric dependent variable.
        alpha : float, default=0.05
            Significance level.
        center : {'mean', 'median', 'trimmed'}, default='median'
            Method for centering in Levene's test. 'median' is most robust.

        Returns
        -------
        T/F bool

        Notes
        -----
        The null hypothesis is equal variances across groups.
        Violation suggests using Welch’s ANOVA instead of standard ANOVA.
        """
        if dropna is None:
            dropna=True
        if alpha is None:
            alpha=0.05
        if center is None: 
            center='median' 
        if retrieve_meta is None:
            retrieve_meta=False
            

        df = pd.DataFrame({'group': group, 'y': y})

        if dropna:
            df = df.dropna()
        else:
            df = df.dropna(subset=['y'])
            df['group'] = df['group'].where(~df['group'].isna(), 'NaN')

        if df.empty:
            return False
        
        dropna=False

        grouped = [vals.values for _, vals in df.groupby('group',dropna=dropna,observed=True)['y']]
        if len(grouped) < 2:
            return False   # can't test homogeneity with only 1 group
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always") 
            stat, p = scipy.stats.levene(*grouped, center=center)
        """if w:
            print(w[-1].message)  """
        meta = {
            'stat': stat,
            'p': p,
            'equal_variance': p > alpha,
            'method': f'Levene (center={center})'
        }

        if retrieve_meta == True:
            return meta
        return meta['equal_variance']


    # 4) Outlier detection (IQR method) + optional removal
    def _anova_detect_outliers(self,
                               group, 
                               y, 
                               iqr_multiplier:float|None=None, 
                               retrieve_meta:bool|None=None,
                               dropna:bool|None=None):
        """
        Detect (and optionally remove) outliers within each group using the IQR method.

        Parameters
        ----------
        group : array-like (pd.Series)
            Categorical grouping variable.
        y : array-like (pd.Series)
            Numeric dependent variable.
    iqr_multiplier : float, default=1.5
            Multiplier for IQR to define outlier thresholds.

        Returns
        -------
        a df w removed outliers
        -----
        Outliers are defined as values outside:
            [Q1 - k*IQR, Q3 + k*IQR]
        where k = iqr_multiplier.
        Outliers can strongly influence ANOVA results.
        """
        if dropna is None:
            dropna=True
        if iqr_multiplier is None:
            iqr_multiplier=1.5
        if retrieve_meta is None:
            retrieve_meta=False
            

        df = pd.DataFrame({'group': group, 'y': y})

        if dropna:
            df = df.dropna()
        else:
            df = df.dropna(subset=['y'])
            df['group'] = df['group'].where(~df['group'].isna(), 'NaN')

        if df.empty:
            return False
        
        dropna=False

        df['is_outlier'] = False
        #############################################################################################
        # add a how parameter here to either identify outliers based on z score or on relation to iqr
        #############################################################################################
        q1 = df.groupby('group',dropna=dropna,observed=True)['y'].transform(lambda x: x.quantile(0.25))
        q3 = df.groupby('group',dropna=dropna,observed=True)['y'].transform(lambda x: x.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        df['is_outlier'] = (df['y'] < lower) | (df['y'] > upper)

        if retrieve_meta == True:
            return df.groupby('group',dropna=dropna,observed=True)['is_outlier'].sum().rename('count_outliers')

        return df.loc[~df['is_outlier']].drop(columns='is_outlier')


    # 5) Wrapper function
    def anova_assumption_checks(self,
                                group, 
                                y, 
                                retrieve_meta:bool|None=None,
                                normality_alpha:bool|None=None, 
                                homogeneity_alpha:bool|None=None,
                                min_n:int|None=None,
                                iqr_multiplier:float|None=None,
                                dropna:bool|None=None):
        """
        Run a full suite of ANOVA assumption checks.

        Parameters
        ----------
        group : array-like (pd.Series)
            Categorical grouping variable.
        y : array-like (pd.Series)
            Numeric dependent variable.
        alpha : float, default=0.05
            Significance level for statistical tests.
        min_n : int, default=5
            Minimum required observations per group.
        remove_outliers : bool, default=False
            Whether to remove outliers before running tests.
        iqr_multiplier : float, default=1.5
            IQR multiplier for outlier detection.

        Returns
        -------
        bool if retrieve_meta==False
        esle df w meta

        Notes
        -----
        This function consolidates key ANOVA assumptions:
        - Adequate group sizes
        - Within-group normality
        - Homogeneity of variances
        - Outlier presence

        Use results to decide whether standard ANOVA is appropriate
        or if alternatives (e.g., Welch ANOVA, Kruskal-Wallis) are needed.
        """
        if retrieve_meta is None:
            retrieve_meta=True
        if normality_alpha is None:
            normality_alpha=0.02 
        if homogeneity_alpha is None:
            homogeneity_alpha=0.05
        if min_n is None:
            min_n=5
        if iqr_multiplier is None:
            iqr_multiplier=2
        if dropna is None:
            dropna=True
            

      

        # use pandas data frame in case of numpy array input to ensure NaN handling is consistent

        df = pd.DataFrame({'group': group, 'y': y})

        if dropna:
            df = df.dropna()
        else:
            df = df.dropna(subset=['y'])
            df['group'] = df['group'].where(~df['group'].isna(), 'NaN')

        if df.empty:
            return False
        
        #create panda Series
        group, y = df['group'], df['y']
        del df
        # avoid unnecessary compute
        dropna=False

        if retrieve_meta == True:
            is_outlier = self._anova_detect_outliers(group, 
                                                    y,
                                                    iqr_multiplier=iqr_multiplier, 
                                                    retrieve_meta=True,
                                                    dropna=dropna)
            group_sizes = self._anova_assumptions_check_group_sizes(group, 
                                                        y,
                                                        min_n=min_n, 
                                                        retrieve_meta=True,
                                                    dropna=dropna)
            result_df=pd.merge(is_outlier,
                            group_sizes,
                            how='inner',
                            right_index=True,
                            left_index=True)
            is_normal = self._anova_check_normality(group, 
                                                    y, 
                                                    alpha=normality_alpha, 
                                                    retrieve_meta=True,
                                                    dropna=dropna)
            result_df=pd.merge(result_df,
                            is_normal,
                            how='inner',
                            right_index=True,
                            left_index=True)
            equal_varaice_between_groups = self._anova_check_homogeneity(group, 
                                                                        y, 
                                                                        alpha=homogeneity_alpha, 
                                                                        retrieve_meta=True,
                                                                        dropna=dropna)
            return equal_varaice_between_groups, result_df[['is_normal','count_outliers',  'count_obs']]




        # Step 1: outliers (optionally clean first)
        outlier_result = self._anova_detect_outliers(group, y,
                                                    iqr_multiplier=iqr_multiplier,
                                                    dropna=dropna)

        df = outlier_result

        
        # Step 2: checks
        res_1 = self._anova_assumptions_check_group_sizes(df['group'], 
                                                    df['y'], 
                                                    min_n=min_n,
                                                    dropna=dropna)
        if res_1 == False:
            return False

        res_2 = self._anova_check_homogeneity(df['group'], 
                                                    df['y'], 
                                                    alpha=homogeneity_alpha,
                                                    dropna=dropna)
        
        if res_2==False:
            return False
    
        res_3 = self._anova_check_normality(df['group'], 
                                                    df['y'], 
                                                    alpha=normality_alpha,
                                                    dropna=dropna)
        if res_3==False:
            return False
        return True
    

    def _kruskal_assumptions_pseudo(self, 
                                    y,
                                    max_global_ties_ratio:float|None=None,
                                    dropna:bool|None=None,
                                    retrieve_meta:bool=None):
        """
        y : array-like (pd.Series)
                    Numeric dependent variable
        
        returns
        -------
        bool if retrieve_meta == False: (ties_ratio<=max_global_ties_ratio)
        else ties_ratio

        What it tests:
                Without the equal-shape assumption ( test of median ), a significant result means:
            At least one group differs in distribution
                That difference could come from:  location shifts, variance differences, skewness, tail behavior
            Conover, W. J. (1999). Practical Nonparametric Statistics (3rd ed.). John Wiley & Sons.
            Hollander, M., Wolfe, D. A., & Chicken, E. (2013). Nonparametric Statistical Methods (3rd ed.). John Wiley & Sons.
            Zar, J. H. (2010). Biostatistical Analysis (5th ed.). Pearson.
        """



        if retrieve_meta==None:
            retrieve_meta=False 
        if max_global_ties_ratio==None:
            max_global_ties_ratio=0.5
        if dropna==None:
            dropna=True

        ties_ratio = y.value_counts(normalize=True,dropna=dropna).max()
        if retrieve_meta==True:
            return ties_ratio
        return "Psuedo" if ties_ratio < max_global_ties_ratio else False

    def _ks_pairwise_parallel(self,
                            df:pd.DataFrame, 
                            x_col:str, 
                            y_col:str, 
                            n_jobs:int=None,
                            alpha:float=None,
                            return_meta:bool=None,
                            guesstimate:dict|None=None):
        """
        This assumes no group size is <3
        as per levene checks that happen before this is called
        if guesstimate is not False or None:
            rej_max_pct_in_group and/or max_num_outlier_all_reject can be passed as a dict in place of True for guesstimate
            either can be exceded for the result to output False
            rej_max_pct_in_group is max percentage of reject in combos the given group is involved in. Such that the group can be in left or right column
            max_num_outlier_all_reject is the z score of number of reject counts given to any given group. all groups are included. including ones with zero rejects
            max_pct_reject_total is total pct of combinations that rejected
            default guesstimate = {'rej_max_pct_in_group':0.2,'max_num_outlier_all_reject':3, 'max_pct_reject_total':0.2}

            two ways to fail: (the 1st is 2 parts)
            1) if one or more varaibles exced percentage failure for num combos it is involved with |or| any one variable excedes the z score threshold
            2) max_pct_reject_total is exceded. In this case, it can be one or many varaibles that contribute. Even if none were caught in previous
        """
        if n_jobs is None:
            n_jobs=3         
        if return_meta is None:
            return_meta=False
        if alpha is None:
            alpha=0.05
        if guesstimate is None:
            guesstimate=False
        if guesstimate!=False:
            default_guesstimate = {'rej_max_pct_in_group':0.2,'max_num_outlier_all_reject':3, 'max_pct_reject_total':0.2}
            if guesstimate==True:
                guesstimate = default_guesstimate
            else:
                default_guesstimate.update(guesstimate)
            rej_max_pct_in_group = default_guesstimate.get('rej_max_pct_in_group',0.2)
            max_num_outlier_all_reject    = default_guesstimate.get('max_num_outlier_all_reject',3)
            max_pct_reject_total = default_guesstimate.get('max_pct_reject_total',0.2)
        
        grouped = (
            df[[x_col, y_col]]
            .groupby(x_col,as_index=True,observed=True)[y_col]
            .apply(np.asarray)
        )

        keys = list(grouped.index)

        grouped = grouped.to_dict()

        def compute(i, j):
            a, b = grouped[i], grouped[j]
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                stat, p = scipy.stats.ks_2samp(a,b)
            """if w:
                print(w[-1].message)   """ 
            return i, j, stat, p

        try:   # catch edge cases where n_jobs may be out of range
            results = Parallel(n_jobs=n_jobs)(
                delayed(compute)(i, j)
                for i, j in combinations(keys, 2)
            )
        except Exception as e:
            warnings.warn(f'Problem encountered during parallel computing of similarity scores for Kruskal-Wallis assumption checks: \n{e}. \nTrying n_jobs = -1.')
            results = Parallel(n_jobs=-1)(
                            delayed(compute)(i, j)
                            for i, j in combinations(keys, 2)
                        )

        base_df = pd.DataFrame(
                            results,
                            columns=[x_col + "_1", x_col + "_2", "ks_stat", "p_value"]
                            )
        if return_meta == True:
            return base_df

        pairwise_difference_detected = base_df['p_value'] < alpha
        if guesstimate==False:
            return not pairwise_difference_detected.any()

        total = base_df.shape[0]
        if total == 0:
            return True

        pct_reject = pairwise_difference_detected.sum()/total
        group_sizes = 2*total/len(keys)  # the group can be in left or right column
        sums_df = pd.DataFrame(np.zeros(len(keys)),index=keys,columns=['totals'])
        rejected_pairs = base_df.loc[pairwise_difference_detected]
        m1 = rejected_pairs.groupby(x_col+"_1",as_index=True,observed=True).size().rename('count_1')
        m2 = rejected_pairs.groupby(x_col+"_2",as_index=True,observed=True).size().rename('count_2')
        sums_df = pd.merge(sums_df, m1, how='left',left_index=True, right_index=True)
        sums_df = pd.merge(sums_df, m2, how='left',left_index=True, right_index=True).fillna(0)
        sums_df['totals'] = sums_df['count_1']+sums_df['count_2']
        mu, q = sums_df['totals'].mean(), sums_df['totals'].std(ddof=0)
        if q == 0:
            sums_df['z'] = 0.0
        else:
            sums_df['z'] = (sums_df['totals']-mu) / q
        sums_df['pct_of_group'] = sums_df['totals']/group_sizes
        # True means this group exceeds a configured failure threshold.
        sums_df['result'] = (sums_df['pct_of_group'] > rej_max_pct_in_group) | (sums_df['z'] > max_num_outlier_all_reject)

        return not (sums_df['result'].any() or (pct_reject > max_pct_reject_total))

    def _kruskal_assumptions_strict(self,
                        group, 
                        y, 
                        levene_alpha:float|None=None,
                        ks_alpha:float|None=None,
                        dropna:bool|None=None,
                        return_pseudo:bool|None=None,
                        pseudo_test_max_global_ties_ratio:float|None=None,
                        guesstimate:dict|bool|None=None,
                        n_jobs:int|None=None):
        """
        where return_pseudo returns "Pseudo-Ratio: "+ ratio if the test of median fails. 
        Without the equal-shape assumption ( test of median ), a significant result means:
            At least one group differs in distribution
                That difference could come from:  location shifts, variance differences, skewness, tail behavior
            Conover, W. J. (1999). Practical Nonparametric Statistics (3rd ed.). John Wiley & Sons.
            Hollander, M., Wolfe, D. A., & Chicken, E. (2013). Nonparametric Statistical Methods (3rd ed.). John Wiley & Sons.
            Zar, J. H. (2010). Biostatistical Analysis (5th ed.). Pearson.
        Because rank-based tests assume continuous data and ties reduce rank resolution and test power, 
        we monitor the proportion of repeated values. While no formal cutoff exists, high tie concentration indicates 
        reduced sensitivity and motivates alternative methods. 
            assumptions parameter should be set w rule of thumb ties_ratio>.5 bad, >.7 raise caution. 
            if check_assumptions==True: default is assumption_check_params={'max_global_ties_ratio' : 0.5}
        calls _ks_pairwise_parallel(). it's guesstimate dict params should be explained in a higher leve function
            ELSE: """

        if levene_alpha is None:
            levene_alpha = 0.05
        if ks_alpha is None:
            ks_alpha = 0.05
        if dropna is None:
            dropna = True
        if return_pseudo is None:
            return_pseudo=False
        if pseudo_test_max_global_ties_ratio is None:
            pseudo_test_max_global_ties_ratio = 0.5
        


        df = pd.DataFrame({'group': group, 'y': y})

        if dropna:
            df = df.dropna()
        else:
            df = df.dropna(subset=['y'])
            df['group'] = df['group'].where(~df['group'].isna(), 'NaN')

        if df.empty:
            return False

        dropna = False
            
        # --- 1. spread similarity (Levene) ---
        groups = [vals.values  for _, vals in df.groupby('group',dropna=dropna,observed=True)['y']]
        if len(groups)<2:
            return False
        if any([len(i)<3 for i in groups]):
            if return_pseudo==True:
                return self._kruskal_assumptions_pseudo(df['y'],
                                                            dropna=dropna,
                                                            max_global_ties_ratio=pseudo_test_max_global_ties_ratio)
            return False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _, levene_p = scipy.stats.levene(*groups, center='median')            
        """if w:
            print(w[-1].message) """               
        if levene_p < levene_alpha:
            if return_pseudo==True:
                return self._kruskal_assumptions_pseudo(df['y'],
                                                            dropna=dropna,
                                                            max_global_ties_ratio=pseudo_test_max_global_ties_ratio)
            return False
        
        # --- 2. shape similarity (pairwise KS) ---
        shape_assumption_met = self._ks_pairwise_parallel(df[['group','y']],
                            'group', 
                            'y', 
                            n_jobs=n_jobs,
                            alpha=ks_alpha,
                            return_meta=False,
                            guesstimate=guesstimate)
        """
        # use self._ks_pairwise_parallel() instead
        for i in range(len(groups)):  
            for j in range(i + 1, len(groups)):  # ensures every combo is tested
                g1 = groups[i]
                g2 = groups[j]
                _, p = scipy.stats.ks_2samp(g1, g2)
        """
        if not shape_assumption_met:
            if return_pseudo==True:
                return self._kruskal_assumptions_pseudo(df['y'],
                                                            dropna=dropna,
                                                            max_global_ties_ratio=pseudo_test_max_global_ties_ratio)
            return False  # distributions differ → not just medians
 
        return True




    def _kruskal_assumptions_meta(self,
                                        group, 
                                        y,
                                        dropna:bool|None=None,
                                        guesstimate:dict|bool|None=None,
                                        n_jobs:int|None=None,
                                        jupyter_output:bool|None=None):
        """
        returns meta as a dict
        if jupyter_ouput==True: 
            returns:
                print(f"x={group.name}, y={y.name}")
                print('kruskal: ',meta_dict['kruskal'])
                print('levene: ',meta_dict['levene'])
                display(round(meta_dict['summary'],6))
                display(round(meta_dict['ks_pairwise'],6))
        """

        if dropna is None:
            dropna = True
        if jupyter_output is None:
            jupyter_output=False

        df = pd.DataFrame({'group': group, 'y': y})

        if dropna:
            df = df.dropna()
        else:
            df = df.dropna(subset=['y'])
            df['group'] = df['group'].where(~df['group'].isna(), 'NaN')

        if df.empty:
            raise ValueError('No valid observations available for Kruskal-Wallis checks.')

        dropna = False
        keys = df['group'].unique()

        summary = (
            df.groupby('group', dropna=dropna, observed=True)['y']
            .agg(
                n='size',
                median='median',
                mean='mean',
                std='std',
                iqr=lambda x: np.subtract(*np.percentile(x, [75, 25])),
                skew='skew'
            )
        )

        # --- variance equality (robust) ---
        groups = [g['y'].values for _, g in df.groupby('group',dropna=dropna,observed=True)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            levene_stat, levene_p = scipy.stats.levene(*groups, center='median')            
        """if w:
            print(w[-1].message)   """
        

        # --- distribution equality (stronger check) ---
        ks_df = self._ks_pairwise_parallel(df[['group','y']], 
                            'group', 
                            'y', 
                            n_jobs=n_jobs,
                            alpha=None,
                            return_meta=True,
                            guesstimate=guesstimate)
        """
        # pairwise KS tests
        ks_results = []
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                g1 = df.loc[df['group'] == keys[i], 'y']
                g2 = df.loc[df['group'] == keys[j], 'y']
                if g1.empty or g2.empty:
                    continue
                stat, p = scipy.stats.ks_2samp(g1, g2)
                ks_results.append((keys[i], keys[j], stat, p))

        ks_df = pd.DataFrame(ks_results, columns=['g1','g2','ks_stat','p']) if ks_results else pd.DataFrame(columns=['g1','g2','ks_stat','p'])
        """

        # --- kruskal-wallis ---
        kw_stat, kw_p = scipy.stats.kruskal(*groups)

        meta_dict =  {
            'summary': summary,
            'levene': {'stat': levene_stat, 'p': levene_p},
            'ks_pairwise': ks_df,
            'kruskal': {'stat': kw_stat, 'p': kw_p}
        }
        if jupyter_output==True:
            try:
                from IPython.display import display as notebook_display
            except ImportError:
                notebook_display = print
            print(f"x={group.name}, y={y.name}")
            print('kruskal: ',meta_dict['kruskal'])
            print('levene: ',meta_dict['levene'])
            notebook_display(round(meta_dict['summary'],6))
            notebook_display(round(meta_dict['ks_pairwise'],6))
        return meta_dict
    
    def kruskal_wallis_assumptions(self,
                        group, 
                        y, 
                        levene_alpha:float|None=None,
                        ks_alpha:float|None=None,
                        dropna:bool|None=None,
                        retrieve_meta:bool|None=None,
                        return_pseudo:bool|None=None,
                        pseudo_test_max_global_ties_ratio:float|None=None,
                        full_pseudo:bool|None=None,
                        jupyter_output:bool|None=None,
                        guesstimate:dict|bool|None=None,
                        n_jobs:int|None=None):
        """
        if retrieve_meta==False
            calls _kruskal_assumptions_strict() w optional parameter return_pseudo to call  _kruskal_assumptions_pseudo() only if assumptions arent met
            _kruskal_assumptions_pseudo() is much less computationaly expensive and can be called on it's own: full_pseudo==True
            in either case, it returns 'Pseudo' instead of a True or False bool
        else
            calls _kruskal_assumptions_meta()
        
        more on return_pseudo:
            where return_pseudo returns "Pseudo-Ratio: "+ ratio if the test of median fails. 
            Without the equal-shape assumption ( test of median ), a significant result means:
                At least one group differs in distribution
                    That difference could come from:  location shifts, variance differences, skewness, tail behavior
                Conover, W. J. (1999). Practical Nonparametric Statistics (3rd ed.). John Wiley & Sons.
                Hollander, M., Wolfe, D. A., & Chicken, E. (2013). Nonparametric Statistical Methods (3rd ed.). John Wiley & Sons.
                Zar, J. H. (2010). Biostatistical Analysis (5th ed.). Pearson.
            Because rank-based tests assume continuous data and ties reduce rank resolution and test power, 
            we monitor the proportion of repeated values. While no formal cutoff exists, high tie concentration indicates 
            reduced sensitivity and motivates alternative methods. 
                assumptions parameter should be set w rule of thumb ties_ratio>.5 bad, >.7 raise caution. 
                if check_assumptions==True: default is assumption_check_params={'max_global_ties_ratio' : 0.5}
        
        if guesstimate is not False or None:
            it controls how pairwise grouped vars are tested for similar distributions
            elements in {'rej_max_pct_in_group':0.2,'max_num_outlier_all_reject':3, 'max_pct_reject_total':0.2} can be passed as a dict in place of True for guesstimate
            any one of them needs to be exceded for the result to output False
            rej_max_pct_in_group is max percentage of reject in combos the given group is involved in. Such that the group can be in left or right column
            max_num_outlier_all_reject is the z score of number of reject counts given to any given group. all groups are included. including ones with zero rejects
            max_pct_reject_total is total pct of combinations that rejected
            default guesstimate = {'rej_max_pct_in_group':0.2,'max_num_outlier_all_reject':3, 'max_pct_reject_total':0.2}

            two ways to fail: (the 1st is 2 parts)
            1) if one or more varaibles exced percentage failure for num combos it is involved with |or| any one variable excedes the z score threshold
            2) max_pct_reject_total is exceded. In this case, it can be one or many varaibles that contribute. Even if none were caught in previous
        """

        if levene_alpha is None:
            levene_alpha = 0.05
        if ks_alpha is None:
            ks_alpha = 0.05
        if dropna is None:
            dropna = True
        if retrieve_meta is None:
            retrieve_meta = True
        if full_pseudo is None:
            full_pseudo=False
        if jupyter_output is None:
            jupyter_output = False

        if retrieve_meta==False:

            if full_pseudo==True:
                return self._kruskal_assumptions_pseudo(y,
                                            max_global_ties_ratio=pseudo_test_max_global_ties_ratio,
                                            dropna=dropna,
                                            retrieve_meta=False)
            return self._kruskal_assumptions_strict(group, 
                                            y, 
                                            levene_alpha=levene_alpha,
                                            ks_alpha=ks_alpha,
                                            dropna=dropna,
                                            return_pseudo=return_pseudo,
                                            pseudo_test_max_global_ties_ratio=pseudo_test_max_global_ties_ratio,
                                            guesstimate=guesstimate,
                                            n_jobs=n_jobs)
        return self._kruskal_assumptions_meta(group, 
                                                        y,
                                                        dropna=dropna,
                                                        guesstimate=guesstimate,
                                                        n_jobs=n_jobs,
                                                        jupyter_output = jupyter_output)
    


    # ========================================================================================================================================================================= 

    # PREPROCESS
    # for uniform 2-way-ANOVA interaction sizes
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html 
    
    def create_uniform_interaction_sizes(self, 
                                    three_col_XX_y:pd.DataFrame,
                                    min_size:None|int, 
                                    override_min:None|bool=None, 
                                    ntile:None|int=None, 
                                    stratify:None|pd.Series=None):
        """
        override_min will sample with replacement from small datasets
        if ntile is not none, the interseciton size will be the ntile of sizes that are greater than min, 
            otherwise smallest acceptable interaciton's size will be used
                use: n_rows=np.percentile(sizes,[*ntile*])[0].astype(int)
                    where ntile is an int in range [0,100]
        where stratify can be None, and sampling will be random, or a pd.series of the binned y variable with an index that aligns with the xx_y_prep_df pd.dataframe index
        """

        min_size = 5 if min_size is None else min_size
        override_min = False if override_min is None else override_min
        cols=three_col_XX_y.columns
        x1 = cols[0]
        x2 = cols[1]
        y = cols[2]
        binned_y = None 
        if stratify!=None:
            try:
                binned_y = stratify.name
            except:
                if 'binned_y' not in (x1,x2,y):
                    binned_y = 'binned_y'
                else: 
                    binned_y = 'binned_y-'
                    while binned_y in (x1,x2,y):
                        binned_y = binned_y + "I"
        #unique values for interactions
        x1_vars=three_col_XX_y[x1].unique()
        x2_vars=three_col_XX_y[x2].unique()

        #record interactions and sizes
        grouped=three_col_XX_y.groupby([x1,x2],as_index=False,observed=False)[y].agg('size')
        sufficient=grouped.loc[(grouped['size']>=min_size)]
        sizes=sufficient['size'].to_list()
        interactions=list(sufficient[[x1,x2]].itertuples(index=False,name=None))
        too_small_interactions=list(grouped.loc[(grouped['size']>0)&(grouped['size']<min_size)][[x1,x2]].itertuples(index=False,name=None))
        non_interactions=list(grouped.loc[(grouped['size']<=0)][[x1,x2]].itertuples(index=False,name=None))

        # concatenate a new dataframe with uniform interaction sizes
        result=[]
        if ntile is not None:
            n_rows=np.percentile(sizes,[ntile])[0].astype(int)
            #
            #warning
            for interaction in interactions:
                weights=None
                if stratify!=None:
                    three_col_XX_y[binned_y] = stratify
                    weights = three_col_XX_y.loc[(three_col_XX_y[x1]==interaction[0])&(three_col_XX_y[x2]==interaction[1])].groupby([x1,x2,binned_y],observed=True).transform('size')
                data=three_col_XX_y[[x1,x2]].loc[(three_col_XX_y[x1]==interaction[0])&(three_col_XX_y[x2]==interaction[1])].sample(n_rows,replace=False, weights=weights)
                result.append(data)
            if override_min==True:
                #
                #warning
                for interaction in too_small_interactions:
                    weights=None
                    if stratify!=None:
                        three_col_XX_y[binned_y] = stratify
                        weights = three_col_XX_y.loc[(three_col_XX_y[x1]==interaction[0])&(three_col_XX_y[x2]==interaction[1])].groupby([x1,x2,binned_y],observed=True).transform('size')
                    data=three_col_XX_y[[x1,x2]].loc[(three_col_XX_y[x1]==interaction[0])&(three_col_XX_y[x2]==interaction[1])].sample(n_rows,replace=True)
                    result.append(data)
        elif ntile is None:
            n_rows=min(sizes)
            for interaction in interactions:
                weights=None
                if stratify!=None:
                    three_col_XX_y[binned_y] = stratify
                    weights = three_col_XX_y.loc[(three_col_XX_y[x1]==interaction[0])&(three_col_XX_y[x2]==interaction[1])].groupby([x1,x2,binned_y],observed=True).transform('size')
                data=three_col_XX_y[[x1,x2]].loc[(three_col_XX_y[x1]==interaction[0])&(three_col_XX_y[x2]==interaction[1])].sample(n_rows, replace=False)
                result.append(data)
            if override_min==True:
                #
                #warning
                for interaction in too_small_interactions:
                    weights=None
                    if stratify!=None:
                        three_col_XX_y[binned_y] = stratify
                        weights = three_col_XX_y.loc[(three_col_XX_y[x1]==interaction[0])&(three_col_XX_y[x2]==interaction[1])].groupby([x1,x2,binned_y],observed=True).transform('size')
                    data=three_col_XX_y[[x1,x2]].loc[(three_col_XX_y[x1]==interaction[0])&(three_col_XX_y[x2]==interaction[1])].sample(n_rows,replace=True)
                    result.append(data)
        if override_min==True:
            interactions=interactions+too_small_interactions
            too_small_interactions=[]
        
        #metrics
        self.two_way_interaction_sizes=grouped
        #over_under_samplimng_interaciton_size=n_rows # not stratified, but random sampling
        self.two_way_interaction_columns={'interactions':interactions,'insufficient':too_small_interactions,'non_interactive': non_interactions }
        return pd.concat(result)
       
   
    # 2 way ANOVA
    
    def two_way_ANOVA(self,
                      three_col_XX_y:pd.DataFrame,
                      unbalanced_interaction_sizes:None|bool=None,
                      verbose:None|bool=None):
        """
        Where col in positions [:2] are catigorical and [2] is numeric
        data is preprocessed to have uniform observations counts across interaction
        """
        unbalanced_interaction_sizes = True if unbalanced_interaction_sizes is None else unbalanced_interaction_sizes
        verbose = True if verbose is None else verbose
        cols=three_col_XX_y.columns
        a = cols[0]
        b = cols[1]
        y = cols[2]

        #create an interaction dataframe (tempararily keep non interaction groups)
        interaction = three_col_XX_y.groupby([a,b],as_index=False,observed=False)[y].agg(['mean','size'])

        # track interacting and non interacting factors
        interaction_factors=list(interaction.loc[interaction['size']>=5][[a,b]].itertuples(index=False,name=None))
        non_interacting_factors=list(interaction.loc[interaction['size']<5][[a,b]].itertuples(index=False,name=None))
        if verbose:
            print(f"Interacting factors: {interaction_factors}\nNon interacting: where count < 5 interaction factors: {non_interacting_factors}")

        # filter out non interaction factor groups
        interaction=interaction.loc[interaction['size']>0]

        if not unbalanced_interaction_sizes:
            nij=interaction['size'].mean()# this is okay because this is not a wieghted anova

        if not unbalanced_interaction_sizes:
            pass
            #assert ... f""
        if unbalanced_interaction_sizes==True:
            pass
            #assert

        #number of groups in a and in b
        a_number_of_groups = three_col_XX_y[a].nunique()
        b_number_of_groups = three_col_XX_y[b].nunique()

        #multiply num of groups by num_interact_ai_and_bj for groups in a & b
        if not unbalanced_interaction_sizes: 
            nij_multiply_a = nij*a_number_of_groups
            nif_multiply_b = nij*b_number_of_groups
        elif unbalanced_interaction_sizes==True:
            interaction['nij_product_a']=interaction['size']*a_number_of_groups
            interaction['nij_product_b']=interaction['size']*b_number_of_groups

        #grand mean
        overall_mean = three_col_XX_y[y].mean()

        #calculate sums of squares for t and e
        SST=((three_col_XX_y[y] - overall_mean)**2).sum()
        ### while broadcast subraction is unsuported, this merge is used as a middle step
        SSE_dataframe = three_col_XX_y.merge(interaction[[a,b,'mean']],on=[a,b],how='left',validate='m:1')
        SSE = ((SSE_dataframe[y]-SSE_dataframe['mean'])**2).sum()
        del SSE_dataframe

        #calculate sums of squares for groups of factors a & b
        if not unbalanced_interaction_sizes:
            mean_y_ai = three_col_XX_y.groupby(a,observed=True)[y].mean()#--------------------------------i changed observed to True in both of these
            interaction=interaction.merge(mean_y_ai.rename('mean_a'),left_on=a,right_index=True)
            SSA = ((mean_y_ai-overall_mean)**2).sum() * nif_multiply_b  # nb * a sum of centered squares 
            mean_y_bj = three_col_XX_y.groupby(b,observed=True)[y].mean()
            interaction=interaction.merge(mean_y_bj.rename('mean_b'),left_on=b,right_index=True)
            SSB = ((mean_y_bj-overall_mean)**2).sum() * nij_multiply_a  # na * b sum of centered squares
        if unbalanced_interaction_sizes==True:
            mean_and_count_y_ai = three_col_XX_y.groupby(a, observed=True)[y].agg(['mean','size'])
            mean_and_count_y_ai.columns=['mean_a','size_a']
            SSA = ((mean_and_count_y_ai['mean_a'] - overall_mean)**2 * mean_and_count_y_ai['size_a']).sum()
            mean_and_count_y_bj = three_col_XX_y.groupby(b, observed=True)[y].agg(['mean','size'])
            mean_and_count_y_bj.columns=['mean_b','size_b']
            SSB = ((mean_and_count_y_bj['mean_b'] - overall_mean)**2 * mean_and_count_y_bj['size_b']).sum()

        # sum of squares for interaction between a & b    
        if not unbalanced_interaction_sizes:
            SSAB = ((interaction['mean'] - interaction['mean_a'] - interaction['mean_b'] + overall_mean)**2).sum() * nij
        if unbalanced_interaction_sizes==True:
            interaction=interaction.merge(mean_and_count_y_ai['mean_a'],left_on=a,right_index=True)
            interaction=interaction.merge(mean_and_count_y_bj['mean_b'],left_on=b,right_index=True)
            SSAB = ((interaction['mean'] - interaction['mean_a'] - interaction['mean_b'] + overall_mean)**2 * interaction['size']).sum()

        if verbose==True:  
            print(f"SST = SSA+SSB+SSAB+SSE: {SST == SSA+SSB+SSAB+SSE}\nnp.isclose(SST, SSA+SSB+SSAB+SSE): {np.isclose(SST, SSA+SSB+SSAB+SSE)}")
            print("SST:", SST)
            print("SSA + SSB + SSAB + SSE:", SSA + SSB + SSAB + SSE)
            print("Residual:", SST - (SSA + SSB + SSAB + SSE))
            print("If the residual error is large, try another model \ntwo_way_ANOVA_for_un_balanced_data(), which implements statsmodels.api\nor over/under sample the data.") 
        # degrees of freedom
        dof_a, dof_b, dof_ab, dof_e = a_number_of_groups-1, b_number_of_groups-1, (a_number_of_groups-1)*(b_number_of_groups-1), three_col_XX_y.shape[0]-(a_number_of_groups * b_number_of_groups)
        # mean squares
        MSA,MSB,MSAB,MSE =  SSA/dof_a, SSB/dof_b, SSAB/dof_ab, SSE/dof_e
        # F-statistics 
        FA,FB,FAB = MSA/MSE, MSB/MSE, MSAB/MSE
        # p values
        p_value_A, p_value_B, p_value_AB = scipy.stats.f.sf(FA,dof_a,dof_e), scipy.stats.f.sf(FB,dof_b,dof_e), scipy.stats.f.sf(FAB,dof_ab,dof_e)

        res= {a:p_value_A,b:p_value_B,'interaction':p_value_AB}
        return res
    
    # =========================================================================================================================================================================

    # A function that pairs combos based on inputs
    #it is used in kruskal_wallis and anova funtions that multiple many p-vlues based on the the combos
    def determine_column_combinations(self,
                                      data:pd.DataFrame, 
                                      numeric_columns:str|list|None=None,
                                      categoric_columns:str|list|None=None,
                                      categoric_target:str|list|None=None,
                                      numeric_target:str|list|None=None,
                                      cols_to_exclude_from_targets:str|list|None=None):
    
        """
        where categoric_columns and numeric_columns default to auto_detect if not [] or column(s) entered as str or list
        categoric_target and numeric_target can both be entered as str or list(s) of strings
        in case wehn neither catigoric_target or numeric_target is None:
            P-Value is eveluated when one OR the other is present in any combination, not one AND the the other
        cols_to_exclude_from_targets do override columns passed in as targets, 
            [IN THE CASE OF AnalyzeDataset MODULE, THAT MEANS RESULTS FOR EXCLUDED COLUMNS SHOULD ALREADY BE STORED AS CLASS ATRIBUTES]
        """

        # DETERMINE THE CATEGORIC COLUMNS
        # base categoric w/o looking at target
        if categoric_columns == None : 
            categoric_columns=list(set(list(data.select_dtypes('object').columns)+list(data.select_dtypes('category').columns)))
        elif  isinstance(categoric_columns,str):
            categoric_columns=[categoric_columns]
        # look at categoric target
        if isinstance(categoric_target,str):
            categoric_target=[categoric_target]
        # ensure the categoric_target is in the categoric columns
        if categoric_target is not None:
            categoric_columns = list( set( categoric_columns + categoric_target ) )
        
        # DETERMINE THE NUMERIC COLUMNS
        # base numeric w/o looking at target
        if numeric_columns == None : 
            numeric_columns = list(data.select_dtypes('number').columns)
        elif isinstance(numeric_columns,str):
            numeric_columns=[numeric_columns]
        # look at numeric target
        if isinstance(numeric_target,str):
            numeric_target=[numeric_target]
        # ensure the numeric_target is in the numeric columns
        if numeric_target is not None:
            numeric_columns = list( set( numeric_columns + numeric_target ) )

        # filter columns to exclude from targets as per parameter
        if (cols_to_exclude_from_targets is not None):
            if isinstance(cols_to_exclude_from_targets,str):
                cols_to_exclude_from_targets = [cols_to_exclude_from_targets]
            categoric_target = [targ for targ in categoric_target if targ not in cols_to_exclude_from_targets]
            numeric_target = [targ for targ in numeric_target if targ not in cols_to_exclude_from_targets]
            
        #conditional statements to consider target columns when defined or every valid combinanation otehrwise
        # no targets
        if (categoric_target is None) and (numeric_target is None):
            combinations=[(cat_col,num_col) for num_col in numeric_columns for cat_col in categoric_columns if cat_col!=num_col] 
        # numeric target only
        elif (categoric_target is None) and (numeric_target is not None):
            combinations=[(cat_col,num_col) for num_col in numeric_columns for cat_col in categoric_columns if ((cat_col!=num_col) and (num_col in numeric_target))]
        # categoric target only
        elif (categoric_target is not None) and (numeric_target is None):
            combinations=[(cat_col,num_col) for num_col in numeric_columns for cat_col in categoric_columns if ((cat_col!=num_col) and (cat_col in categoric_target))]
        # categoric and numeric targets
        elif (categoric_target is not None) and (numeric_target is not None):
            combinations=[(cat_col,num_col) for num_col in numeric_columns for cat_col in categoric_columns if ((cat_col!=num_col) and ((cat_col in categoric_target) or (num_col in numeric_target)))]
        # raise error if no condition is met
        else:
            raise ValueError("There seems to be an error in one or both of numeric_target and categoric_target parameters")
        #return list of combinations: [(cat,num),(cat2,num2), ..., (catn,numn)]
        return combinations

    # ========================================================================================================================================================================= 
    # one way tests: ANOVA and Kruskal Wallis
    # =========================================================================================================================================================================
    # ANOVA
    def one_way_ANOVA(self,
                      two_col_cat_num_df:pd.DataFrame,
                      dropna:bool|None=None):
        """
        Where col in position [0] is catigorical and [1] is numeric
        returns np.nan when there arent enough observations or when there is no within-group variance
        """

        two_col_cat_num_df=pd.DataFrame(two_col_cat_num_df) 
        cols=two_col_cat_num_df.columns
        x= cols[0]
        y= cols[1]

        if dropna is None:
            dropna=True
        if dropna==False:
            two_col_cat_num_df = two_col_cat_num_df.dropna(subset=[y])
            two_col_cat_num_df[x] = two_col_cat_num_df[x].where(~two_col_cat_num_df[x].isna(), 'NaN')
        else:
            two_col_cat_num_df = two_col_cat_num_df.dropna()


        number_of_groups=two_col_cat_num_df[x].nunique()
        overall_mean=two_col_cat_num_df[y].mean()

        #Sum of Squares Between
        SB_df=two_col_cat_num_df.groupby(x,observed=True)[y].agg(['size','mean'])
        SSB=(((SB_df['mean']-overall_mean)**2)*SB_df['size']).sum()

        #Sum of Squares Within
        #untill broadcast level is supported @ https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/api/cudf.dataframe.subtract/#cudf.DataFrame.subtract
        #join is used instead of direct subtraction
        #map might be faster on smaller datasets than join
        SW_df=two_col_cat_num_df.join(SB_df,on=x,validate='m:1')
        SW_df['observed-group_mean']=SW_df[y]-SW_df['mean']
        SW_df['observed-group_mean']=SW_df['observed-group_mean']**2
        SW_df=SW_df.groupby(x,observed=True)['observed-group_mean'].sum()
        #SW_df=SW_df**2
        SSW=SW_df.sum()
        del SW_df, SB_df

        #Degrees of Freedom
        dof_W=two_col_cat_num_df.shape[0] - number_of_groups
        # return np.nan if there aren't enough observations for ANOVA
        if dof_W<=0: return np.nan
        dof_B=number_of_groups - 1

        #Mean Squares
        MSB=SSB/dof_B
        MSW=SSW/dof_W
        #return np.nan if MSW is 0: all values in groups are identical, ANOVA is meaningless
        if MSW == 0 or np.isclose(MSW, 0): return np.nan

        #F-statistic
        f_statistic = MSB/MSW 

        #P_value      https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html#scipy.stats.f
        p_value = scipy.stats.f.sf(f_statistic,dof_B,dof_W)

        return p_value

    def test_all_num_cat_ANOVA(self,
                               data:pd.DataFrame, 
                               numeric_columns:str|list|None=None,
                               categoric_columns:str|list|None=None,
                               categoric_target:str|list|None=None,
                               numeric_target:str|list|None=None,
                               cols_to_exclude_from_targets:str|list|None=None,
                               check_assumptions:bool|None=None,
                               assumption_check_params:dict|None=None):
        """
        where categoric_columns and numeric_columns default to auto_detect if not None or column(s) entered as str or list
        categoric_target and numeric_target can both be entered as str or list(s) of strings
        in case when neither categoric_target or numeric_target is None:
            P-Value is evaluated when one OR the other is present in any combination, not one AND the the other
        cols_to_exclude_from_targets do override columns passed in as targets, 
            [IN THE CASE OF AnalyzeDataset MODULE, THAT MEANS RESULTS FOR EXCLUDED COLUMNS SHOULD ALREADY BE STORED AS CLASS ATTRIBUTES]
        """
        if assumption_check_params is None:
            assumption_check_params = {}
        if check_assumptions is None:
            check_assumptions=True
        if check_assumptions==True:
            if assumption_check_params:
                normality_alpha=assumption_check_params.get('normality_alpha', 0.02)
                homogeneity_alpha=assumption_check_params.get('homogeneity_alpha', 0.05)
                min_n=assumption_check_params.get('min_n', 5) # min obs per group
                iqr_multiplier=assumption_check_params.get('iqr_multiplier', 2)
            else:
                normality_alpha= 0.02
                homogeneity_alpha= 0.02
                min_n= 5
                iqr_multiplier= 2
        if assumption_check_params:
                dropna=assumption_check_params.get('dropna', True)
        else:
            dropna= True

        combinations = self.determine_column_combinations(data, 
                                                          numeric_columns=numeric_columns,
                                                          categoric_columns=categoric_columns,
                                                          categoric_target=categoric_target,
                                                          numeric_target=numeric_target,
                                                          cols_to_exclude_from_targets=cols_to_exclude_from_targets)
        if len(combinations)<1:
            return pd.DataFrame(columns=['category','numeric','P-value'] if check_assumptions==False else ['category','numeric','P-value','assumptions_met'])
        res_dict={'category':[],'numeric':[],'P-value':[]}
        if check_assumptions==True:
            res_dict['assumptions_met']=[]
        for combo in combinations:
            p=self.one_way_ANOVA(data[[*combo]],
                                 dropna=dropna)
            res_dict['category'].append(combo[0])
            res_dict['numeric'].append(combo[1])
            res_dict['P-value'].append(p)
            if check_assumptions==True:
                anova_assumptions_met = self.anova_assumption_checks(data[combo[0]],
                                                                      data[combo[1]],
                                                                      retrieve_meta=False,
                                                                      normality_alpha=normality_alpha, 
                                                                      homogeneity_alpha=homogeneity_alpha,
                                                                      min_n=min_n,
                                                                      iqr_multiplier=iqr_multiplier,
                                                                      dropna=dropna)
                res_dict['assumptions_met'].append(anova_assumptions_met)
        res = pd.DataFrame(res_dict)
        res = res[[i for i in ['category','numeric','P-value','assumptions_met'] if i in res.columns]]
        return res

    # =========================================================================================================================================================================
    #ONE WAY KRUSKUL WALLIS TEST, similar to a onw way ANOVA but the assumptions make it more robust to non-normally distributed data
    #https://library.virginia.edu/data/articles/getting-started-with-the-kruskal-wallis-test
    #H = (12 / N(N + 1)) / sum(Ri² / ni) - 3(N + 1)
    # assumptions 1) all groups have the same distribution (else larger sample size), more groups require larger sample sizes, 
    # ========================================================================================================================================================================= 
    
    # Kruskal Wallis
    def one_way_kruskal_wallis(self,
                               two_col_cat_num_df:pd.DataFrame,
                               dropna:bool|None=None):
        """
        Where col in positions [0] is catigorical and [1] is numeric
        """
        
        data=pd.DataFrame(two_col_cat_num_df.copy())
        cols=data.columns
        x = cols[0]
        y = cols[1]
        
        if dropna is None:
            dropna=True
        if dropna==False:
            data = data.dropna(subset=[y])
            data[x] = data[x].where(~data[x].isna(), 'NaN')
        else:
            data = data.dropna()

        groups = [
            group.to_numpy()
            for _, group in data.groupby(x, observed=True)[y]
        ]
        if len(groups) < 2 or data[y].nunique() < 2:
            return np.nan
        return scipy.stats.kruskal(*groups).pvalue
    
    def test_all_num_cat_kruskal_wallis(self,
                                        data:pd.DataFrame, 
                                        numeric_columns:str|list|None=None,
                                        categoric_columns:str|list|None=None,
                                        categoric_target:str|list|None=None,
                                        numeric_target:str|list|None=None,
                                        cols_to_exclude_from_targets:str|list|None=None,
                                        check_assumptions:bool|None=None,
                                        assumption_check_params:dict|None=None):
        """
        where categoric_columns and numeric_columns default to auto_detect if not None or column(s) entered as str or list
        categoric_target and numeric_target can both be entered as str or list(s) of strings
        in case wehn neither catigoric_target or numeric_target is None:
            P-Value is eveluated when one OR the other is present in any combination, not one AND the the other
        cols_to_exclude_from_targets do override columns passed in as targets, 
            [IN THE CASE OF AnalyzeDataset MODULE, THAT MEANS RESULTS FOR EXCLUDED COLUMNS SHOULD ALREADY BE STORED AS CLASS ATRIBUTES]
                
        """

        if assumption_check_params is None:
            assumption_check_params = {}
        if check_assumptions is None:
            check_assumptions=True
        if check_assumptions==True:
            if assumption_check_params:
                levene_alpha = assumption_check_params.get('levene_alpha', 0.05)
                ks_alpha = assumption_check_params.get('ks_alpha', 0.05)
                return_pseudo = assumption_check_params.get('return_pseudo', False)
                pseudo_test_max_global_ties_ratio = assumption_check_params.get('pseudo_test_max_global_ties_ratio', 0.5)
                full_pseudo = assumption_check_params.get('full_pseudo', False)
                guesstimate = assumption_check_params.get('guesstimate',False)
                n_jobs      = assumption_check_params.get('n_jobs',3)
            else: 
                levene_alpha =  0.05
                ks_alpha = 0.05
                return_pseudo = False
                pseudo_test_max_global_ties_ratio =  0.5
                full_pseudo = False
                guesstimate = False
                n_jobs      = 3
        if assumption_check_params:
                dropna=assumption_check_params.get('dropna', True)
        else:
            dropna= True

        combinations = self.determine_column_combinations(data, 
                                                          numeric_columns=numeric_columns,
                                                          categoric_columns=categoric_columns,
                                                          categoric_target=categoric_target,
                                                          numeric_target=numeric_target,
                                                          cols_to_exclude_from_targets=cols_to_exclude_from_targets)

        if len(combinations)<1:
            return pd.DataFrame(columns=['category','numeric','P-value'] if check_assumptions==False else ['category','numeric','P-value','assumptions_met'])
        res_dict={'category':[],'numeric':[],'P-value':[]}
        if check_assumptions==True:
            res_dict['assumptions_met']=[]
        for combo in combinations:
            p=self.one_way_kruskal_wallis(data[[*combo]],
                                          dropna=dropna)
            res_dict['category'].append(combo[0])
            res_dict['numeric'].append(combo[1])
            res_dict['P-value'].append(p)
            if check_assumptions==True:
                if not np.isfinite(p):
                    kruskal_assumptions_met = False
                else:
                    kruskal_assumptions_met = self.kruskal_wallis_assumptions(data[combo[0]],
                                                                        data[combo[1]],
                                                                        levene_alpha=levene_alpha,
                                                                        ks_alpha=ks_alpha,
                                                                        dropna=dropna,
                                                                        retrieve_meta=False,
                                                                        return_pseudo=return_pseudo,
                                                                        pseudo_test_max_global_ties_ratio=pseudo_test_max_global_ties_ratio,
                                                                        full_pseudo=full_pseudo,
                                                                        jupyter_output=False,
                                                                        guesstimate=guesstimate,
                                                                        n_jobs=n_jobs)
                res_dict['assumptions_met'].append(kruskal_assumptions_met)
        res = pd.DataFrame(res_dict)
        res = res[[i for i in ['category','numeric','P-value','assumptions_met'] if i in res.columns]]
        return res
    #=========================================================================================================================================================
    # a comparison function that supports either 'kruskal' or 'anova'
    #=========================================================================================================================================================
    
    def num_cat_column_comparison(self,
                                  data:pd.DataFrame, 
                                  alpha:float|None=None,
                                  keep_above_p:bool|None=None, 
                                  numeric_columns:list|None=None,
                                  categoric_columns:list|None=None,
                                  numeric_target:str|list|None=None,
                                  categoric_target:str|list|None=None, 
                                  test_method:None|str=None,
                                  cols_to_exclude_from_targets:str|list|None=None,
                                  check_assumptions:bool|None=None,
                                  anova_assumption_check_params:dict|None=None,
                                  kruskal_assumption_check_params:dict|None=None):
        """
        test_method can be of ('kruskal','anova')
        takes alpha as a parameter 
        if keep_above_p==False, observations with p_values<alpha are returned
        if keep_above_p==True p_values>=alpha are returned
        else all p_values are returned
        """

        test_method = 'kruskal' if test_method is None else test_method
        alpha = 0.05 if alpha is None else alpha
        if test_method=='kruskal':
            p_table=self.test_all_num_cat_kruskal_wallis(data=data, 
                                                        numeric_columns=numeric_columns,
                                                        categoric_columns=categoric_columns,
                                                        categoric_target=categoric_target,
                                                        numeric_target=numeric_target,
                                                        cols_to_exclude_from_targets=cols_to_exclude_from_targets,
                                                        check_assumptions=check_assumptions,
                                                        assumption_check_params=kruskal_assumption_check_params,
                                                        )
            
        elif test_method=='anova':
            p_table=self.test_all_num_cat_ANOVA(data=data, 
                                                        numeric_columns=numeric_columns,
                                                        categoric_columns=categoric_columns,
                                                        categoric_target=categoric_target,
                                                        numeric_target=numeric_target,
                                                        cols_to_exclude_from_targets=cols_to_exclude_from_targets,
                                                        check_assumptions=check_assumptions,
                                                        assumption_check_params=anova_assumption_check_params)     
    
        else:
            raise ValueError("Unknown test_method. test_method should be one of ('kruskal','anova')")
        if keep_above_p==False:
            return p_table.loc[p_table['P-value']<alpha].reset_index(drop=True)
        elif keep_above_p==True:
            return p_table.loc[p_table['P-value']>=alpha].reset_index(drop=True)
        else: 
            return p_table

    # ========================================================================================================================================================================= 
    # ========================================================================================================================================================================= 
    # ========================================================================================================================================================================= 
    # two sample t_tests
    def two_sample_t_tests(self,
                           data:pd.DataFrame,
                           include_invalid:bool=False):

        """
        where data columns[ 0 , 1 ] should be [ x_feature=category , y_target=numeric ]
        Default is "welch's" t_test because it is robust to unequal varaince and unequal sample sizes

        By default, comparisons involving a group with fewer than two non-null
        observations or zero/undefined variance are omitted. Set
        include_invalid=True to retain those comparisons with a NaN P-value.
        """

        cols=data.columns
        cat=cols[0]
        num=cols[1]
        df1=data.groupby(cat,as_index=False,observed=True)[num].agg(['mean','std','count'])
        if df1.shape[0] < 2:
            count_dtype = 'Int64' if include_invalid else int
            return pd.DataFrame(
                {
                    'subcat_1': pd.Series(dtype=df1[cat].dtype),
                    'subcat_2': pd.Series(dtype=df1[cat].dtype),
                    'P-value': pd.Series(dtype=float),
                    'n_samples_1': pd.Series(dtype=count_dtype),
                    'n_samples_2': pd.Series(dtype=count_dtype),
                }
            )

        combos = list(itertools.combinations(df1[cat].unique(), 2))
        merged=pd.DataFrame(combos,columns=['subcat_1','subcat_2'])
        merged=merged.merge(df1,how='left',right_on=cat,left_on='subcat_1')
        merged=merged.merge(df1,how='left',right_on=cat,left_on='subcat_2',suffixes=('1','2'))
        merged=merged.rename(columns={'count1':'n_samples_1','count2':'n_samples_2'})

        valid = (
            (merged['n_samples_1'] > 1)
            & (merged['n_samples_2'] > 1)
            & np.isfinite(merged['std1'])
            & np.isfinite(merged['std2'])
            & (merged['std1'] > 0)
            & (merged['std2'] > 0)
        )
        merged['P-value'] = np.nan
        if valid.any():
            merged.loc[valid,'P-value'] = scipy.stats.ttest_ind_from_stats(
                mean1=merged.loc[valid,'mean1'],
                std1=merged.loc[valid,'std1'],
                nobs1=merged.loc[valid,'n_samples_1'],
                mean2=merged.loc[valid,'mean2'],
                std2=merged.loc[valid,'std2'],
                nobs2=merged.loc[valid,'n_samples_2'],
                equal_var=False,
            ).pvalue

        if not include_invalid:
            merged=merged.loc[valid]

        merged=merged[['subcat_1','subcat_2','P-value','n_samples_1','n_samples_2']].reset_index(drop=True)
        count_dtype = 'Int64' if include_invalid else int
        merged['n_samples_1']=merged['n_samples_1'].astype(count_dtype)
        merged['n_samples_2']=merged['n_samples_2'].astype(count_dtype)
        return merged

    def subcategory_similarities(self,
                                 catx_numy_df:pd.DataFrame,
                                 alpha:None|float=None,
                                 return_similar:None|bool=None,
                                 min_observations:None|int=None,
                                 include_invalid:bool=False):
        """
        where catx_numy_df is a pd.DataFrame with categoric x feature and numeric y target
        calls two_sample_t_tests and can filter based on p_value and/or sample size according to parameters
        """
        alpha = 0.05 if alpha is None else alpha
        return_similar = False if return_similar is None else return_similar
        data=self.two_sample_t_tests(catx_numy_df,include_invalid=include_invalid)
        if return_similar==False:
            data=data.loc[data['P-value']<alpha]
        if min_observations is not None:
            data=data.loc[(data['n_samples_1']>min_observations)&(data['n_samples_2']>min_observations)]
        data=data.reset_index(drop=True)
        return data
