
import pandas as pd
import numpy as np
import scipy
import warnings





class UnivariateNormal:

    def __inti__(self):
        pass
    #test for uniformness  
    ## consider adding a flag if any cell has an expected value <
    def normality_test(self,
                        x:pd.Series):
        """
        accepts x as a panda series 
        if samp<300: scipy.stats.shapiro
        else: scipy.stats.normaltest
        returns a p value
        """
        x = x.dropna()
        if len(x) < 300:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")  
                stat, p = scipy.stats.shapiro(x)   
        else:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always") 
                stat, p = scipy.stats.normaltest(x)  
        """if w:
            print(w[-1].message) """ 
        return p


    def test_all_num_columns_normality_test(self,
                                             data,
                                             columns:str|list|None=None,
                                             cols_to_exclude_from_targets:str|list|None=None):
        """
        if columns == None, this defaults to detect ['number',np.number] dtypes and won't see 'int'
        returns a df of [col, p-value]
        """
        if not columns:
            columns=list(data.select_dtypes(['number',np.number]).columns)
        elif isinstance(columns,str):
            columns=[columns]
        if cols_to_exclude_from_targets is not None:
            if isinstance(cols_to_exclude_from_targets,str):
                cols_to_exclude_from_targets=[cols_to_exclude_from_targets]
            columns = [col for col in columns if col not in cols_to_exclude_from_targets]
        if not columns:
            return pd.DataFrame(columns=['numeric','P-value'])
        res_dict={'numeric':[],'P-value':[]}
        for col in columns:
            p=self.normality_test(data[col])
            res_dict['numeric'].append(col)
            res_dict['P-value'].append(p)
        res = pd.DataFrame(res_dict)
        return res[['numeric','P-value']]



    def filterable_all_column_normality_test(self,
                        df,
                        cat_alpha_above:tuple|list=(0.05,False),
                        numeric_columns:str|list|None=None,
                        cols_to_exclude_from_targets:str|list|None=None):
        """
        parameters:
        df: a pandas dataframe
        cat_alpha_above:tuple|list=(0.05,False) 
            where index 0 is alpha, 
            and index 1 should be True to include H1, 
                False to include H0
                None for unfiltered
        categoric_columns:str|list|None=None is a list of columns to test. If None, columns will be infered
            hence, if None, it is nescesary that column datatypes are accurate
        expected_probs can be a list of probabilites to test against. None defaults to uniform distribution
            no further documentation on expected_probs available at this time
        """
        normal_df = self.test_all_num_columns_normality_test(data=df,
                                                            columns=numeric_columns,
                                                            cols_to_exclude_from_targets=cols_to_exclude_from_targets)
        if cat_alpha_above[1]==False:
            return normal_df.loc[normal_df['P-value']<cat_alpha_above[0]].reset_index(drop=True)
        if cat_alpha_above[1]==True:
            return normal_df.loc[normal_df['P-value']>=cat_alpha_above[0]].reset_index(drop=True)
        else:
            return normal_df
