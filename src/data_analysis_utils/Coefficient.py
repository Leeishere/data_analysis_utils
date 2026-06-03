import pandas as pd
import numpy as np
import itertools
from itertools import combinations




class Coefficient:
    def __init__(self):
        self.one_way_pearson_coefficient_overview=None

    def target_features_corr_to_rest(self,data:pd.DataFrame,target:str|list,method:str='pearson'):
        """
        Where method can be of 'pearson', 'kendall', 'spearman'
        Iterates through all input dataframe columns and caculates the correlation with target column(s)
        returns an unsorted dataframe data[number_1, number_2, Correlation]
        """
        if type(target)==str:
            target=[target]
        cols=data.columns
        correlations=[]
        targs=[]
        y_vars=[]
        for col in cols:
            for targ in target:
                if col==targ:
                    continue
                corr=data[targ].corr(data[col],method=method)
                correlations.append(corr)
                targs.append(targ)
                y_vars.append(col)
        return pd.DataFrame({'numeric_1':targs,'numeric_2':y_vars,'Correlation':correlations}) 
    


    def test_all_num_num_coefficient(self,data,corr_method:str='spearman', numeric_columns:str|list|None=None, target:list|str|None=None):
        """
        Where numeric_columns==None will fall back to auto_detect
        """        
        if numeric_columns is None: 
            numeric_columns=list(data.select_dtypes('number').columns)
        if  isinstance(numeric_columns,str):
            numeric_columns=numeric_columns

        num_data=pd.DataFrame(data.copy())
        if target is not None:
            if isinstance(target,str):
                target=[target]
            return self.target_features_corr_to_rest(num_data[list(set(numeric_columns+target))],target,method='pearson')
        num_data=num_data[numeric_columns].corr(method=corr_method).unstack().reset_index(drop=False).rename(columns={'level_0':'numeric_1','level_1':'numeric_2',0:'Correlation'})
        mask=num_data['numeric_1']!=num_data['numeric_2']
        num_data = num_data.loc[mask].reset_index(drop=True)
        map_base=pd.DataFrame(list(itertools.combinations(numeric_columns, 2)),columns=['numeric_1','numeric_2'])
        num_data=map_base.merge(num_data,how='inner',on=['numeric_1','numeric_2'])
        return num_data


    def num_num_column_coefficient_comparison(self,data, corr_threshold=0.6,keep_above_corr:bool|None=True,numeric_columns:str|list|None=None,target:list|str|None=None,corr_method:str='spearman'):
        """
        corr_method can be one of ('pearson','spearman','kendall')
        keep_above_corr determines filtering with True, False, or None
        takes corr_threshold as a parameter 
        """
        num_data=self.test_all_num_num_coefficient(data,corr_method,numeric_columns,target)
        if keep_above_corr==True:
            num_data=num_data.loc[np.abs(num_data['Correlation'])>=corr_threshold].reset_index(drop=True)
        elif keep_above_corr==False: 
            num_data = num_data.loc[np.abs(num_data['Correlation'])<corr_threshold].reset_index(drop=True)
        return num_data



