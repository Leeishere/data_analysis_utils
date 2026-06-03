
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.chdtrc.html#scipy.special.chdtrc
# of note: scipy.special.chdtrc(dof,chi_stat) is optimized for jax, pytorch, and cupy-drop-in replacements  

import pandas as pd
import numpy as np
import itertools
from itertools import combinations
import scipy
from scipy import special
from scipy.special import chdtrc
import warnings
import gc

class Chi2:

    # a helper function that drops or turns pd.NA or np.nan into text 'NaN'
    def _dropna_else_cat(self,
                    observed_values:pd.Series|pd.DataFrame, 
                    dropna:bool|None=None):
        """
        if dropna==False, np.nan and pd.NA become 'NaN' as text and can easily be computed as a subcategory 
        such as in hyp tests
        else np.nan and pd.NA are dropped
        """
        if dropna is None:
            dropna=True
        if dropna == True:
            observed_values = observed_values.dropna()
        elif dropna == False:
            observed_values = observed_values.where(~observed_values.isna(),'NaN')
        return observed_values
    
    # check chi2 assumptions 
    def _chi2_assumptions_check(self,
                            data:pd.Series|pd.DataFrame,
                            _series:bool|None=None):
        """
        where series has to be input. if False, data should be pd.DataFrame. if True, data should be pd.Series
        ensure 80% of data(expected) is >=5
        and all are>=1
        """
        if _series is None: 
            raise RuntimeError('_series cannot be None. Input _series==True if input data is pd.Series input, or _series==False if pd.DataFrame')
        elif _series==True:
            if type(data)!=pd.Series:
                raise RuntimeError('_series==True, but found type: ',type(data))
            elif (data.min()<1) or  ( (data<5).sum().sum() > (data.shape[0]*.2) ):
                return False
        elif _series==False:
            if type(data)!=pd.DataFrame:
                raise RuntimeError('_series==False. Expected: pd.DataFrame but found type: ',type(data))
            elif (data.min().min()<1) or  ( ((data<5).sum().sum()) > ((data.shape[0]*data.shape[1])*.2) ):
                return False
        return True

    def column_makeup(self,
                      data:pd.DataFrame,
                      x1:str,
                      x2:str):
        """
        accepts impute as arg1: a dataframe 
        arg2: list of column(s) to calculate the grouped count(s) and percentages of
        returns a dataframd with columns [arg2,count,pct_makeup]
        """
        columns=[x1,x2]
        func_df= data[columns].groupby(columns,as_index=False,observed=False).size().rename(columns={'size':'count'})
        total=func_df['count'].sum()
        func_df['pct_makeup']=func_df['count']/total         
        return func_df.sort_values(by=[i for i in columns]).reset_index(drop=True)
        
    def contingency_table(self,
                          data:pd.DataFrame,
                          x_1:str,
                          x_2:str,
                          kind:str|None=None):
        """
        input a dataframe, column_name_1:str, column_name_2'str, and kind = 'joint_probability | 'frequency' (defaults to joint_probability)  
        returns a contingency table
        BOTTOM AND LEFT COLUMNS CONTAINING MARGINAL VALUES  
        """
        kind = 'joint_probability' if kind is None else kind
        base_quantity_table=self.column_makeup(data,
                                               x_1,
                                               x_2)
        value_col='pct_makeup' if kind=='joint_probability' else 'count'
        contingency_table=base_quantity_table.pivot(index=x_1,columns=x_2,values=value_col).reset_index(drop=False)
        del base_quantity_table
        contingency_table.columns.name=x_2
        contingency_table['right_margin'.upper()]=contingency_table.iloc[:,1:].sum(axis=1)
        contingency_table=pd.concat( [contingency_table, pd.Series(['bottom_margin'.upper()]+list(contingency_table.iloc[:,1:].sum(axis=0)),index=contingency_table.columns).to_frame().T   ]  )
        contingency_table=contingency_table.set_index(contingency_table.columns[0])
        contingency_table = contingency_table.fillna(0)
        contingency_table = contingency_table.apply(pd.to_numeric, errors='coerce')
        if kind=='joint_probability': contingency_table = contingency_table.astype(float)
        else:  contingency_table = contingency_table.astype(int)
        return contingency_table

    #expected values
    def frequencies_table(self,
                          data:pd.DataFrame,
                          x_1:str,
                          x_2:str,
                          kind:str|None=None):
        """
        kind='frequency' for a frequency table
        """
        kind = 'joint_probability' if kind is None else kind
        base_quantity_table=self.column_makeup(data,
                                               x_1,
                                               x_2)
        value_col='pct_makeup' if kind=='joint_probability' else 'count'
        frequencies_table=base_quantity_table.pivot(index=x_1,columns=x_2,values=value_col).reset_index(drop=False)
        del base_quantity_table
        frequencies_table=frequencies_table.set_index(frequencies_table.columns[0])
        return frequencies_table
        
    ## consider adding a flag if any cell has an expected value <
    def chi_squared_independence(self,
                                 data:pd.DataFrame,
                                 x_1:str,
                                 x_2:str,
                                check_assumptions:bool|None=None,
                                dropna:bool|None=None):     
        """
        test of independence
        """


            
        if dropna is None:
            dropna=True
        # drop or turn pd.NA and np.nan into text: 'NaN' based on dropna bool
        test_data   =self._dropna_else_cat(data[[x_1,x_2]], 
                                            dropna=dropna) 

        try:
            observed = self.frequencies_table(test_data, 
                                              x_1, 
                                              x_2, 
                                              kind='frequency')
            expected = ( observed.sum(axis=1).to_numpy().reshape(-1,1)@observed.sum(axis=0).to_numpy().reshape(1,-1) ) / observed.sum().sum()
            chi_stat = (( (observed-expected)**2)/expected).sum().sum()
            dof      = (observed.shape[0]-1)*(observed.shape[1]-1)
            p_value  = scipy.special.chdtrc(dof,chi_stat)
            if check_assumptions==True:
                assumption_met = self._chi2_assumptions_check(pd.DataFrame(expected),
                                                                _series=False)
                return p_value, assumption_met    
            return p_value
        except:
            obs     =test_data.groupby([x_1,x_2],as_index=False,observed=False).size().rename(columns={'size':'observed'})
            sums_x_1=test_data.groupby(x_1,as_index=False,observed=False).size().rename(columns={'size':'x_1_totals'})
            sums_x_2=test_data.groupby(x_2,as_index=False,observed=False).size().rename(columns={'size':'x_2_totals'})
            dof     = (sums_x_1.shape[0]-1)*(sums_x_2.shape[0]-1)
            obs     =pd.merge(obs,sums_x_1,how='left',on=x_1)
            obs     =pd.merge(obs,sums_x_2,how='left',on=x_2)
            grand_total=obs['observed'].sum().sum()
            obs['expected']=(obs['x_1_totals']*obs['x_2_totals'])/grand_total
            chi_stat  =  (((obs['observed'] - obs['expected'])**2)  /  obs['expected']).sum().sum()
            p_value  = scipy.special.chdtrc(dof,chi_stat)
            if check_assumptions==True:
                assumption_met = self._chi2_assumptions_check(obs['expected'],
                                                                _series=True)
                return p_value, assumption_met  
            return p_value


    def test_all_cat_columns_chi_independence(self,
                                              data:pd.DataFrame,
                                              columns:str|list|None=None,
                                              target:str|list|None=None,
                                              cols_to_exclude_from_targets:str|list|None=None,
                                              check_assumptions:bool|None=None,
                                              assumption_check_params:dict|None=None):
        """
        if columns == None, this defaults to detect 'object' and 'category' dtypes and won't see 'int'
        additional can be None or numeric column(s) to analyze
        if target is not None it should be a list. Only combinations that include target will be returned
        """

        if check_assumptions is None:
            check_assumptions=True
       
        if assumption_check_params:
            dropna=assumption_check_params.get('dropna', True)
        else:
            dropna = True

        if columns is None:
            columns = list(set(list(data.select_dtypes('object').columns)+list(data.select_dtypes('category').columns)))
        elif isinstance(columns,str):
            columns = [columns]
        if target == None:
            combinations = list(itertools.combinations(columns,2))
        else: 
            if isinstance(target,str):
                target=[target]
            if cols_to_exclude_from_targets is not None:
                if isinstance(cols_to_exclude_from_targets,str):
                    cols_to_exclude_from_targets=[cols_to_exclude_from_targets]
                target = [targ for targ in target if targ not in cols_to_exclude_from_targets]
            columns = list( set( columns+target ) )
            combinations = [(targ, col) for targ in target for col in columns if targ != col]
        if not combinations:            
            return pd.DataFrame(columns=['category_a','category_b','P-value'] if check_assumptions==False else ['category_a','category_b','P-value','assumptions_met'])
        if check_assumptions==True:
            res_dict={'category_a':[],'category_b':[],'P-value':[],'assumptions_met':[]}
            for combo in combinations:
                combo_data = data[[combo[0],combo[1]]]
                p, assumption_met =self.chi_squared_independence(combo_data,
                                                combo[0],
                                                combo[1],
                                                check_assumptions=check_assumptions,
                                                dropna=dropna)
                res_dict['category_a'].append(combo[0])
                res_dict['category_b'].append(combo[1])
                res_dict['P-value'].append(p)
                res_dict['assumptions_met'].append(assumption_met)
        else:
            res_dict={'category_a':[],'category_b':[],'P-value':[]}
            for combo in combinations:
                combo_data = data[[combo[0],combo[1]]]
                p =self.chi_squared_independence(combo_data,
                                                 combo[0],
                                                 combo[1],
                                                 check_assumptions=check_assumptions,
                                                 dropna=dropna)
                res_dict['category_a'].append(combo[0])
                res_dict['category_b'].append(combo[1])
                res_dict['P-value'].append(p)
        if len(res_dict)<1:
            return pd.DataFrame(columns=['category_a','category_b','P-value'] if check_assumptions==False else ['category_a','category_b','P-value','assumptions_met'])
        res = pd.DataFrame(res_dict)
        res = res[[i for i in ['category_a','category_b','P-value','assumptions_met'] if i in res.columns]]
        return res

    # a function that calls chi2 independence to examine and compare dataset columns
    def categorical_column_comparison(self,
                                      data, 
                                      alpha:float|None=None, 
                                      keep_above_p:None|bool=None, 
                                      categoric_columns:list|None=None, 
                                      categoric_target:list|None=None,
                                      cols_to_exclude_from_targets:str|list|None=None,
                                      check_assumptions:bool|None=None,
                                      assumption_check_params:dict|None=None):
        """
        if columns == None, this defaults to detect 'object' and 'category' dtypes and won't see 'int'
        """
        alpha = 0.05 if alpha is None else alpha

        p_table=self.test_all_cat_columns_chi_independence(data,
                                                           categoric_columns,
                                                           categoric_target,
                                                           cols_to_exclude_from_targets=cols_to_exclude_from_targets,
                                                           check_assumptions=check_assumptions,
                                                           assumption_check_params=assumption_check_params)
        if keep_above_p==True:
            return p_table.loc[p_table['P-value']>=alpha].reset_index(drop=True)
        elif keep_above_p==False:
            return p_table.loc[p_table['P-value']<alpha].reset_index(drop=True)
        else:
            return p_table


    #test for uniformness  
    ## consider adding a flag if any cell has an expected value <
    def chi_squared_goodness_of_fit(self,
                                    x:pd.Series,
                                    check_assumptions:bool|None=None,
                                    dropna:bool|None=None):
        """
        accepts x as a panda series 
        uniform distribution is calculated and used to test agains within the function
        input of expected or expected probabilities is not supported
        """
 
            
        if dropna is None:
            dropna=True
        # drop or turn pd.NA and np.nan into text: 'NaN' based on dropna bool
        x=self._dropna_else_cat(pd.Series(x), 
                                dropna=dropna)        

        val_counts = x.value_counts()
        observed = pd.Series(val_counts.values,index=val_counts.index)
        
        expected =[x.shape[0]/x.nunique()]*observed.shape[0]
        expected = np.array(expected)
      
        chi_stat = (( (observed-expected)**2)/expected).sum()
        dof      = x.nunique()-1
        p        = scipy.special.chdtrc(dof, chi_stat)
        if check_assumptions == True:
            return p, self._chi2_assumptions_check(pd.Series(expected),_series=True)
        return p

    def test_all_cat_columns_chi_good_of_fit(self,
                                             data,
                                             columns:str|list|None=None,
                                             cols_to_exclude_from_targets:str|list|None=None,
                                             dropna:bool|None=None,
                                             check_assumptions:bool|None=None):
        """
        if columns == None, this defaults to detect 'object' and 'category' dtypes and won't see 'int'
        returns a df of [col, p-value]
        does not support input of expected or expected probabilities
        """
        if check_assumptions is None:
            check_assumptions=True

        if not columns:
            columns=list(data.select_dtypes(['object','category']).columns)
        elif isinstance(columns,str):
            columns=[columns]
        if cols_to_exclude_from_targets is not None:
            if isintance(cols_to_exclude_from_targets,str):
                cols_to_exclude_from_targets=[cols_to_exclude_from_targets]
            columns = [col for col in columns if col not in cols_to_exclude_from_targets]
        if not columns:
            return pd.DataFrame(columns=['category','P-value'] if check_assumptions==False else ['category','P-value','assumptions_met'])
        if check_assumptions==True:
            res_dict={'category':[],'P-value':[],'assumptions_met':[]}
            for col in columns:
                p, assumption_met =self.chi_squared_goodness_of_fit(data[col],
                                                check_assumptions=check_assumptions,
                                                dropna=dropna)
                res_dict['category'].append(col)
                res_dict['P-value'].append(p)
                res_dict['assumptions_met'].append(assumption_met)
        else:
            res_dict={'category':[],'P-value':[]}
            for col in columns:
                p=self.chi_squared_goodness_of_fit(data[col],
                                                check_assumptions=check_assumptions,
                                                dropna=dropna)
                res_dict['category'].append(col)
                res_dict['P-value'].append(p)
        res = pd.DataFrame(res_dict)
        res = res[[i for i in ['category','P-value','assumptions_met'] if i in res.columns]]
        return res



    def filterable_all_column_goodness_of_fit(self,
                        df,
                        cat_alpha_above:tuple|list=(0.05,False),
                        categoric_columns:str|list|None=None,
                        cols_to_exclude_from_targets:str|list|None=None,
                        dropna:bool|None=None,
                        check_assumptions:bool|None=None):
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
        good_of_fit_df = self.test_all_cat_columns_chi_good_of_fit(df,
                                                                   columns=categoric_columns,
                                                                    cols_to_exclude_from_targets=cols_to_exclude_from_targets,
                                                                    dropna=dropna,
                                                                    check_assumptions=check_assumptions)
        if cat_alpha_above[1]==False:
            return good_of_fit_df.loc[good_of_fit_df['P-value']<cat_alpha_above[0]].reset_index(drop=True)
        if cat_alpha_above[1]==True:
            return good_of_fit_df.loc[good_of_fit_df['P-value']>=cat_alpha_above[0]].reset_index(drop=True)
        else:
            return good_of_fit_df

    ### DEPRECIATED
    def categorical_column_good_fit(self,data,alpha=0,is_uniform=True):
        """
        is_uniform specifies to return only results that meet is or isn't booleon criterion
        returns all cat columns by default. So dtype 'category', or 'object' must be assigned to the data.
        """
        p_table=self.test_all_cat_columns_chi_good_of_fit(data,additional=True)
        if is_uniform==False:
            return p_table.loc[p_table['P-value']<alpha].reset_index(drop=True)
        return p_table.loc[p_table['P-value']>=alpha].reset_index(drop=True)



    def chi2_homogeneity(self,x1,x2):
        """
        Chi-square test of homogeneity for two samples.
        to answer: “Do these two samples come from the same categorical distribution?”
        """
        x1 = pd.Series(x1).value_counts().to_frame('x1')
        x2 = pd.Series(x2).value_counts().to_frame('x2')
        observed=x1.merge(x2,how='outer',left_index=True,right_index=True)
        observed = observed.fillna(0)

        row_totals=observed.sum(axis=1)
        col_totals=observed.sum(axis=0)
        grand_total=row_totals.sum()
        expected = np.outer(row_totals, col_totals) / grand_total
        #expected = pd.DataFrame(expected, index=observed.index, columns=observed.columns)
        low = expected < 5
        if low.sum() / expected.size > 0.2:
            warnings.warn("More than 20% of expected counts < 5")
            return np.nan
        """
        if (expected < 5).any().any():
            warnings.warn("Chi-square assumption violated: expected count < 5\nDefault: return NaN")
            return np.nan
        """
        chi_stat=(((observed.to_numpy()-expected)**2)/expected).sum().sum()
        dof      = (observed.shape[0]-1)*(observed.shape[1]-1)
        p_value  = scipy.special.chdtrc(dof,chi_stat)
        return p_value




    def chi_subcat_analysis(self,colx1_colx2_df):
        """
        takes two colomns and breaks down how subcategories within each compare
        it looks in both directions: 
        every combination of 2 subcats in colx1 are compared based on how they relate to colx2.
        and every subcat combination of 2 in colx2 are compared as well.
        """
        header_1, header_2 =colx1_colx2_df.columns[0],colx1_colx2_df.columns[1]
        colx1,colx2=colx1_colx2_df[header_1],colx1_colx2_df[header_2]
        #get all combinations for both columns apart from each other
        subcat_combos_in_1, subcat_combos_in_2 = list(itertools.combinations(list(set(colx1)),2)), list(itertools.combinations(list(set(colx2)),2))
        #built the base dataframe that will store P-values
        subcat_1_df=pd.DataFrame(subcat_combos_in_1,columns=['subcat_a','subcat_b'])
        subcat_1_df['source_column']=header_1
        subcat_2_df=pd.DataFrame(subcat_combos_in_2,columns=['subcat_a','subcat_b'])
        subcat_2_df['source_column']=header_2
        result_df=pd.concat([subcat_1_df,subcat_2_df])
        result_df['P-value']=np.nan
        #iterate through a loop that will check combination in both directions
        for i in range(max(len(subcat_combos_in_1),len(subcat_combos_in_2))):

            if i < len(subcat_combos_in_1):
                subcat_1a=colx1_colx2_df.loc[colx1_colx2_df[header_1]==subcat_combos_in_1[i][0]][header_2]
                subcat_1b=colx1_colx2_df.loc[colx1_colx2_df[header_1]==subcat_combos_in_1[i][1]][header_2]
                p_1=self.chi2_homogeneity(subcat_1a,subcat_1b)
                result_df.loc[(result_df['subcat_a']==subcat_combos_in_1[i][0])&(result_df['subcat_b']==subcat_combos_in_1[i][1])&(result_df['source_column']==header_1),'P-value']=p_1

            if i < len(subcat_combos_in_2):
                subcat_2a=colx1_colx2_df.loc[colx1_colx2_df[header_2]==subcat_combos_in_2[i][0]][header_1]
                subcat_2b=colx1_colx2_df.loc[colx1_colx2_df[header_2]==subcat_combos_in_2[i][1]][header_1]
                p_2=self.chi2_homogeneity(subcat_2a,subcat_2b)
                result_df.loc[(result_df['subcat_a']==subcat_combos_in_2[i][0])&(result_df['subcat_b']==subcat_combos_in_2[i][1])&(result_df['source_column']==header_2),'P-value']=p_2
        return result_df



    def subcategory_similarities(self,cat_cat_df,alpha=0.05,return_similar=False,drop_nans=True):
        """
        implements a chi**2 test of homogeneity between all subcat combinations in two columns both directions: colA->colB and colB->colA 
        returns a dataframe filtered by parameters: alpha=0.05,return_similar=False
        """
        data=self.chi_subcat_analysis(cat_cat_df)
        if drop_nans==True:
            data=data.dropna(subset=['P-value'])
        if return_similar==False:
            data=data.loc[data['P-value']<alpha]
        data=data.reset_index(drop=True)
        return data
    

    
        """
        Potential adds to homogeneity and independence tests:
        add Cramér’s V
        auto-collapse sparse categories
        generalize to k samples
        """

    #============================================================================================================================================================
    #============================================================================================================================================================
    #functions that investigate fs feature relationships are due to on being a supercategory of the other, where if it were partitioned, it would segment the subcat with minimal overlap of shared variable values

    def proportion_are_deterministic_subcats(self,df, super_col, sub_col):
        """
        return num_deterministic_subcats/num_non_deterministic_subcats
        takes a dataframe, a potential supercat, and a potential subcat
        returns number of subcats that are restricted to a partition defined by exactly one supercat over total times any subcat is paired with a test_supercat
        """
        data = df[[super_col, sub_col]].copy()
        
        # Count how many unique supercategories each subcategory maps to
        mapping_counts = data.groupby(sub_col)[super_col].nunique()    
        # Deterministic check
        deterministic = (mapping_counts == 1).all()    
        # Proportion of subcategories that map to only one supercategory
        proportion_deterministic = (mapping_counts == 1).mean()
        return proportion_deterministic

    # this should be done with .unstack(). .unstack() is often faster than crosstab in pandas, and oftern faster than marges or joins in cudf
    def evidence_is_supercat_given_subcat(self,df,supercat,subcat,is_cudf:bool=False):
        """
        Measures uncertainty in `super_col` given `sub_col`
        where:
        0 bits -> perfect supercategory
        Higher values -> more ambiguity
        """
        # Conditional entropy H(super_col | sub_col)-> note that this is H not P
        if is_cudf==True:
            # use groupby and merge because cudf crosstab and pivot aren't straighforward at this time 
            # where pivot would require aggrigation in a groupby anyways
            # .div()
            # https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/api/cudf.dataframe.div/#cudf.DataFrame.div
            # axis ->Only 0 is supported for series, 1 or columns supported for dataframe
            # .crosstab()
            # https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/api/cudf.crosstab/#cudf.crosstab
            # normalize -> Not supported
            makeup_df=df[[subcat,supercat]].copy().groupby([subcat,supercat],as_index=False,observed=False).size().set_index(subcat).drop(columns=supercat)
            #sums of interactions within subcat partitons
            totals_map=makeup_df.groupby(makeup_df.index,as_index=True)['size'].sum().to_frame().rename(columns={'size':'partition_total'})
            #map total subcat interactions with partitioned subcat interactions
            makeup_df=pd.merge(makeup_df,totals_map,left_index=True,right_index=True)
            del totals_map     
            makeup_df['normalized']=makeup_df['size']/makeup_df['partition_total']
            makeup_df=makeup_df['normalized'].to_frame().fillna(0)
            # use log() /log(2) in place of np.log2() to allow drop in replacements for numpy 
            # where 0.6931471805599453 is log(2)
            makeup_df['shannon_prep']=np.log(np.asarray(makeup_df['normalized']+1e-12))/0.6931471805599453*np.asarray(makeup_df['normalized'])
            # return bits
            return -makeup_df.groupby(makeup_df.index)['shannon_prep'].sum().mean()
        else:
            try:
                # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.crosstab.html
                # so each row sums to 1: pd.crosstab(index,columns,rows/sum_rows)
                crosstab = pd.crosstab(df[subcat], df[supercat], normalize='index')
                # Shannon Entropy
                # log 2 because we are expressing in bits for interpretability 
                # # use log() /log(2) in place of np.log2() to allow drop in replacements for numpy 
                # where 0.6931471805599453 is log(2)    
                bits = -(crosstab * np.log(crosstab + 1e-12)/0.6931471805599453).sum(axis=1).mean()
                return bits
            except:
                #fallback to groupby and merge in case of crosstab support issues
                makeup_df=df[[subcat,supercat]].copy().groupby([subcat,supercat],as_index=False,observed=False).size().set_index(subcat).drop(columns=supercat)
                #sums of interactions within subcat partitons
                totals_map=makeup_df.groupby(makeup_df.index,as_index=True)['size'].sum().to_frame().rename(columns={'size':'partition_total'})
                #map total subcat interactions with partitioned subcat interactions
                makeup_df=pd.merge(makeup_df,totals_map,left_index=True,right_index=True)
                del totals_map     
                makeup_df['normalized']=makeup_df['size']/makeup_df['partition_total']
                makeup_df=makeup_df['normalized'].to_frame().fillna(0)
                # use log() /log(2) in place of np.log2() to allow drop in replacements for numpy 
                # where 0.6931471805599453 is log(2)
                makeup_df['shannon_prep']=np.log(np.asarray(makeup_df['normalized']+1e-12))/0.6931471805599453*np.asarray(makeup_df['normalized'])
                # return bits
                return -makeup_df.groupby(makeup_df.index)['shannon_prep'].sum().mean()
                



    def map_subcat_to_supercat(self,df,supercat,subcat):
        """
        Returns a dataframe that maps to the most common supercategory for each subcategory.
        """
        data = df[[supercat, subcat]].copy()
        # Most common mapping for each subcategory
        mapping = data.groupby(subcat)[supercat].agg(lambda x: x.value_counts().idxmax()).sort_values(by=[supercat,subcat],ascending=[True,True]).reset_index(drop=True)
        return mapping
        


    def test_many_cats_for_deterministic_super_cats(self,df,columns:list):
        """
        takes a data frame and list of columns
        dataframe with columns 'supercategory', 'subcategory', 'proportion_true_subcategories'
        where:
        Cardinality determines candidate direction
        Entropy determines evidence strength
        """       
        columns=list(set(columns))
        data=df[columns].copy() 
        if len(columns)<1:
            return None
        res=[]
        l,r=0,1
        while l<len(columns)-1:
            if data[columns[l]].nunique()<=data[columns[r]].nunique():
                arrangement=(columns[l],columns[r],self.proportion_are_deterministic_subcats(data,columns[l],columns[r]))
            else: arrangement=(columns[r],columns[l],self.proportion_are_deterministic_subcats(data,columns[r],columns[l]))
            res.append(arrangement)
            if r>=len(columns)-1:
                l+=1
                r=l+1
            else:
                r+=1
        return pd.DataFrame(res,columns=['supercategory', 'subcategory', 'proportion_are_true_subcategories'])
    
    def get_deterministic_super_subcat_pairs(self,df,columns:list,min_proportion=1.0):
        """
        calls test_many_cats_for_deterministic_super_cat and filters results to return super&subcat pairs
        filtered by min_proportion
        where min_proporion is the proporion of subcategories that don't overlap with supercategories at all
        """
        frame=self.test_many_cats_for_deterministic_super_cats(df,columns)
        frame=frame.loc[frame['proportion_are_true_subcategories']>=min_proportion].sort_values(by=['supercategory','subcategory']).reset_index(drop=True)
        return frame


    def test_many_cats_for_evidence_of_super_cats(self,df,columns:list,is_cudf:bool=False):
        """
        takes a data frame and list of columns
        evidence is not probabilistic
        dataframe with columns 'supercategory', 'subcategory', 'ShannonEntropyBits'
        where:
        Cardinality determines candidate direction
        Entropy determines evidence strength
        """
        columns=list(set(columns))
        data=df[columns].copy()
        if len(columns)<1:
            return None
        res=[]
        l,r=0,1
        while l<len(columns)-1:
            if data[columns[l]].nunique()<=data[columns[r]].nunique():
                arrangement=(columns[l],columns[r],self.evidence_is_supercat_given_subcat(data,columns[l],columns[r],is_cudf))
            else: arrangement=(columns[r],columns[l],self.evidence_is_supercat_given_subcat(data,columns[r],columns[l],is_cudf))
            res.append(arrangement)
            if r>=len(columns)-1:
                l+=1
                r=l+1
            else:
                r+=1
        return pd.DataFrame(res,columns=['supercategory', 'subcategory', 'ShannonEntropyBits'])

    def imperfect_super_subcat_pairs(self,df,columns:list,max_evidence=0.2,is_cudf:bool=False):
        """
        calls test_many_cats_for_evidence_of_super_cats_cat and filters results to return super&subcat pairs
        filtered by max_evidence
        where max_evidence is the log2 binary evidence that of proof against a perfect cat&subcat relationship
        """
        frame=self.test_many_cats_for_evidence_of_super_cats(df,columns,is_cudf)
        frame=frame.loc[frame['ShannonEntropyBits']<=max_evidence].sort_values(by=['supercategory','subcategory']).reset_index(drop=True)
        return frame
        

