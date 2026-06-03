

import pandas as pd
import numpy as np
import inspect



from .ANOVA import ANOVA
from .Chi2 import Chi2
from .Coefficient import Coefficient
from .TTests import TTests


class CompareColumns(ANOVA, Chi2, Coefficient, TTests):

    def column_comparison(self,
                        df,
                        numnum_meth_alpha_above:tuple|list|None=('welch',0.05,False),
                        numcat_meth_alpha_above:tuple|list|None=('kruskal',0.05,False),
                        catcat_meth_alpha_above:tuple|list|None=('chi2',0.05,False),
                        numeric_columns:str|list|None=None,
                        categoric_columns:str|list|None=None,
                        numeric_target:str|list|None=None,
                        categoric_target:str|list|None=None):
        """
        DOES NOT CONSIDER ASSUMPTIONS
            for assumption support try CompareColumns().multi_test_column_comparison()
        parameters:
        df: a pandas dataframe
        numnum_meth_alpha_above, numcat_meth_alpha_above, and catcat_meth_alpha_above take input of:
            None or a tuple with (test method, alpha threshold, and whether >= or < in relation to threshold or both)
            if tuple, values should be (string, float, boolean|None).
            Examples: ('chi2',0.05,False), ('anova',0.025,None), ('welch',0.01,True).
            where: 
                numnum_meth_alpha_above for a numeric-to-numeric comparison. Accepts methods of ('welch','student','pearson','spearman',kendall').
                numcat_meth_alpha_above for a categoric-to-numeric comparison. Accepts methods of ('kruskal','anova').
                catcat_meth_alpha_above for a categoric-to-categoric comparison. Accepts method of ('chi2'). 
        numeric_columns and categoric_columns accept manual column input. Otherwise columns are autodetected,
        numeric_target and categoric_target accept target columns. If either or both, only combinations involving targets will be considered
        """
        # a list to store datframes in. it will be used in a concat function 
        result_frames_to_concat = []

        #create boolean variables to use for managing targets
        is_cat_target, is_num_target = (not (not categoric_target)), (not (not numeric_target))
        # if there are any targets at all
        if is_cat_target or is_num_target:  # there is cat or num targets or both
            if not is_num_target: # then there is a cat target and no num target
                include_numnum, include_numcat, include_catcat = False, True, True
            elif not is_cat_target:  # then there is a num target and no cat target
                include_numnum, include_numcat, include_catcat = True, True, False
            else:  # there are cat and num targets
                include_numnum, include_numcat, include_catcat = True, True, True
        else:  # there are no targets
            include_numnum, include_numcat, include_catcat = True, True, True

        # append relevant dataframes to the concat list
        if include_numcat and (numcat_meth_alpha_above is not None):
            #retrieve cat to num df
            if numcat_meth_alpha_above[0] not in ('anova','kruskal'):
                raise ValueError(f"Categoric to Numeric method not recognized. Expected one of ('anova','kruskal'). Recieved {numcat_meth_alpha_above[0]}",ValueError)
            numcat_df=self.num_cat_column_comparison(df,
                                                        alpha=numcat_meth_alpha_above[1],
                                                        keep_above_p=numcat_meth_alpha_above[2],
                                                        numeric_columns=numeric_columns,
                                                        categoric_columns=categoric_columns,
                                                        numeric_target=numeric_target,
                                                        categoric_target=categoric_target,
                                                        test_method=numcat_meth_alpha_above[0],
                                                        check_assumptions=False)  
            numcat_df=numcat_df.rename(columns={'category':'column_b','numeric':'column_a'})
            numcat_df['test']=numcat_meth_alpha_above[0]
            if numcat_df.shape[0]>0:
                result_frames_to_concat.append(numcat_df)
        if include_catcat and (catcat_meth_alpha_above is not None):
            # retrieve cat to cat df
            if catcat_meth_alpha_above[0] not in ('chi2'):
                raise ValueError(f"Categoric to Categoric method not recognized. Expected one of ('chi2'). Recieved {catcat_meth_alpha_above[0]}",ValueError)
            catcat_df=self.categorical_column_comparison(df,
                                                            alpha=catcat_meth_alpha_above[1],
                                                            keep_above_p=catcat_meth_alpha_above[2],
                                                            categoric_columns=categoric_columns,
                                                            categoric_target=categoric_target,
                                                             check_assumptions=False )
            catcat_df=catcat_df.rename(columns={'category_a':'column_a','category_b':'column_b'})
            catcat_df['test']=catcat_meth_alpha_above[0]
            if catcat_df.shape[0]>0:
                result_frames_to_concat.append(catcat_df)
        if include_numnum and (numnum_meth_alpha_above is not None):
            # retrieve num to num df
            if numnum_meth_alpha_above[0] in ('pearson','spearman','kendall'):
                numnum_df=self.num_num_column_coefficient_comparison(df,
                                                                    corr_threshold=numnum_meth_alpha_above[1],
                                                                    keep_above_corr=numnum_meth_alpha_above[2],
                                                                    numeric_columns=numeric_columns,
                                                                    target=numeric_target,
                                                                    corr_method=numnum_meth_alpha_above[0])
            elif numnum_meth_alpha_above[0] in ('welch','student'):
                numnum_df=self.num_num_column_t_test_comparison(df,
                                                                alpha=numnum_meth_alpha_above[1],
                                                                keep_above_p=numnum_meth_alpha_above[2],
                                                                numeric_columns=numeric_columns,
                                                                target=numeric_target,
                                                                t_test_method=numnum_meth_alpha_above[0])
                
            else:
                raise ValueError(f"Numeric to Numeric method not recognized. Expected one of ('pearson','spearman','kendall','welch','student'). Recieved {numnum_meth_alpha_above[0]}",ValueError)
            numnum_df=numnum_df.rename(columns={'numeric_1':'column_a','numeric_2':'column_b'})
            numnum_df['test']=numnum_meth_alpha_above[0]
            if numnum_df.shape[0]>0:                
                    result_frames_to_concat.append(numnum_df)
                    
        possible_columns=['column_a','column_b','test','P-value','Correlation']
        if (not result_frames_to_concat):
            return pd.DataFrame(columns=possible_columns)
        result=pd.concat(result_frames_to_concat)
        result=result[[col for col in possible_columns if col in result.columns]]
        return result




    def multi_test_column_comparison(self,
                        df,
                        numnum_meth_alpha_above:tuple|list|None=[('pearson',0.6,True),('spearman',0.6,True),('kendall',0.6,True)],
                        numcat_meth_alpha_above:tuple|list|None=[('kruskal',0.05,False),('anova',0.05,False)],
                        catcat_meth_alpha_above:tuple|list|None=[('chi2',0.05,False)],
                        numeric_columns:str|list|None=None,
                        categoric_columns:str|list|None=None,
                        numeric_target:str|list|None=None,
                        categoric_target:str|list|None=None,
                        cols_to_exclude_from_targets:str|list|None=None,
                        check_assumptions:bool|None=None,
                        anova_assumption_check_params:dict|None=None,
                        kruskal_assumption_check_params:dict|None=None,
                        chi2_assumption_check_params:dict|None=None):
        """
        parameters:
        df: a pandas dataframe
        numnum_meth_alpha_above, numcat_meth_alpha_above, and catcat_meth_alpha_above take input of:
            None or a LIST of tuple(s) with (test method, alpha threshold, and whether >= or < in relation to threshold or None for both)
            if tuple, values should be (string, float, boolean|None).
            Examples: ('chi2',0.05,False), ('anova',0.025,None), ('welch',0.01,True).
            where: 
                numnum_meth_alpha_above for a numeric-to-numeric comparison. Accepts methods of ('welch','student','pearson','spearman',kendall').
                numcat_meth_alpha_above for a categoric-to-numeric comparison. Accepts methods of ('kruskal','anova').
                catcat_meth_alpha_above for a categoric-to-categoric comparison. Accepts method of ('chi2'). 
        numeric_columns and categoric_columns accept manual column input. Otherwise columns are autodetected,
        numeric_target and categoric_target accept target columns. If either or both, only combinations involving targets will be considered
        
        Kruskal Wallis Assumption Defaults & Explanations for parameter kruskal_assumption_check_params
                levene_alpha =  0.05  # for leven test of variance
                ks_alpha = 0.05       # ks test of similar distributions
                dropna = True         # whether categoric variable NaNs should be treated as categories or dropped
                return_pseudo = False # when standard assumptions aren't met, revert to pseudo test
                    returns 'Pseudo' or false based on test for difference in distributions within groups: 
                    location shifts, variance differences, skewness, tail behavior
                    these/this assumption takes less computation
                pseudo_test_max_global_ties_ratio =  0.5   # max ratio in pseudo test: >0.5 is bad, >0.7 is dangerous
                full_pseudo = False                        # only use pseudo, not the full assumption check

        ANOVA Assumption Defaults & Explanations for parameter anova_assumption_check_params
                normality_alpha= 0.02   # for Kolmogorov-Smirnov test of normality
                homogeneity_alpha= 0.02 # leven test of variance
                min_n= 5                # min obs per group
                iqr_multiplier= 2       # outlier detection
                dropna= True            # determines whether to drop categoric NaN obs, or treat them as categories
        """
        # a list to store datframes in. it will be used in a concat function 
        result_frames_to_concat = []

        # manage targets
        #   create boolean variables to use for managing targets
        is_cat_target, is_num_target = (categoric_target is not None), (numeric_target is not None)
        # if there are any targets at all
        if is_cat_target or is_num_target:  # there is cat or num targets or both
            if not is_num_target: # then there is a cat target and no num target
                include_numnum, include_numcat, include_catcat = False, True, True
            elif not is_cat_target:  # then there is a num target and no cat target
                include_numnum, include_numcat, include_catcat = True, True, False
            else:  # there are cat and num targets
                include_numnum, include_numcat, include_catcat = True, True, True
        else:  # there are no targets
            include_numnum, include_numcat, include_catcat = True, True, True

        # append relevant dataframes to the concat list
        if include_numcat and (numcat_meth_alpha_above is not None):
            if (not isinstance(numcat_meth_alpha_above[0],list)) and (not isinstance(numcat_meth_alpha_above[0],tuple)):
                raise ValueError(f"multi_test_column_comparison() should have var-to-var test instructions nested such as [(),()]. Found {numcat_meth_alpha_above}")
            for instruction in numcat_meth_alpha_above:                
                #retrieve cat to num df
                if instruction[0] not in ('anova','kruskal'):
                    raise ValueError(f"Categoric to Numeric method not recognized. Expected one of ('anova','kruskal'). Recieved {numcat_meth_alpha_above[0]}",ValueError)
                numcat_df=self.num_cat_column_comparison(df,
                                                            alpha=instruction[1],
                                                            keep_above_p=instruction[2],
                                                            numeric_columns=numeric_columns,
                                                            categoric_columns=categoric_columns,
                                                            numeric_target=numeric_target,
                                                            categoric_target=categoric_target,
                                                            test_method=instruction[0],
                                                            cols_to_exclude_from_targets=cols_to_exclude_from_targets,                                                            
                                                            check_assumptions=check_assumptions,
                                                            anova_assumption_check_params=anova_assumption_check_params,
                                                            kruskal_assumption_check_params=kruskal_assumption_check_params)  
                numcat_df=numcat_df.rename(columns={'category':'column_b','numeric':'column_a'})
                numcat_df['test']=instruction[0]
                if numcat_df.shape[0]>0:
                    result_frames_to_concat.append(numcat_df)
        if include_catcat and (catcat_meth_alpha_above is not None):
            if (not isinstance(catcat_meth_alpha_above[0],list)) and (not isinstance(catcat_meth_alpha_above[0],tuple)):
                raise ValueError(f"multi_test_column_comparison() should have var-to_var test instructions nested such as [(),()]. Found {numcat_meth_alpha_above}")
            for instruction in catcat_meth_alpha_above: 
                # retrieve cat to cat df
                if instruction[0] not in ('chi2'):
                    raise ValueError(f"Categoric to Categoric method not recognized. Expected one of ('chi2'). Recieved {catcat_meth_alpha_above[0]}",ValueError)
                catcat_df=self.categorical_column_comparison(df,
                                                                alpha=instruction[1],
                                                                keep_above_p=instruction[2],
                                                                categoric_columns=categoric_columns,
                                                                categoric_target=categoric_target,                                                            
                                                                check_assumptions=check_assumptions,
                                                                assumption_check_params=chi2_assumption_check_params )
                catcat_df=catcat_df.rename(columns={'category_a':'column_a','category_b':'column_b'})
                catcat_df['test']=instruction[0]
                if catcat_df.shape[0]>0:
                    result_frames_to_concat.append(catcat_df)
        if include_numnum and (numnum_meth_alpha_above is not None):
            if (not isinstance(numnum_meth_alpha_above[0],list)) and (not isinstance(numnum_meth_alpha_above[0],tuple)):
                raise ValueError(f"multi_test_column_comparison() should have var-to_var test instructions nested such as [(),()]. Found {numcat_meth_alpha_above}")
            for instruction in numnum_meth_alpha_above: 
                # retrieve num to num df
                if instruction[0] in ('pearson','spearman','kendall'):
                    numnum_df=self.num_num_column_coefficient_comparison(df,
                                                                        corr_threshold=instruction[1],
                                                                        keep_above_corr=instruction[2],
                                                                        numeric_columns=numeric_columns,
                                                                        target=numeric_target,
                                                                        corr_method=instruction[0])
                elif instruction[0] in ('welch','student'):
                    numnum_df=self.num_num_column_t_test_comparison(df,
                                                                    alpha=instruction[1],
                                                                    keep_above_p=instruction[2],
                                                                    numeric_columns=numeric_columns,
                                                                    target=numeric_target,
                                                                    t_test_method=instruction[0])
                    
                else:
                    raise ValueError(f"Numeric to Numeric method not recognized. Expected one of ('pearson','spearman','kendall','welch','student'). Recieved {numnum_meth_alpha_above[0]}",ValueError)
                numnum_df=numnum_df.rename(columns={'numeric_1':'column_a','numeric_2':'column_b'})
                numnum_df['test']=instruction[0]
                if numnum_df.shape[0]>0:
                    result_frames_to_concat.append(numnum_df)
        possible_columns=['column_a','column_b','test','P-value','Correlation','assumptions_met']
        if (not result_frames_to_concat):
            return pd.DataFrame(columns=possible_columns)
        result=pd.concat(result_frames_to_concat)
        result=result[[col for col in possible_columns if col in result.columns]]
        return result