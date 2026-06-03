


from .CompareColumns import CompareColumns
from .Chi2 import Chi2
from .Combinators import calculate_num_combinations
from .PlotClass import PlotClass
from .UnivariateNormal import UnivariateNormal
from .BinnerClass import Bin




import pandas as pd
import numpy as np
import warnings
from warnings import warn
from itertools import combinations

class AnalyzeDataset(Bin, CompareColumns, Chi2, PlotClass, UnivariateNormal):

    def __init__(self,
                 numnum_meth_alpha_above_instructions:list|tuple|None|bool=True, 
                 numcat_meth_alpha_above_instructions:list|tuple|None|bool=True, 
                 catcat_meth_alpha_above_instructions:list|tuple|None|bool=True,
                 good_of_fit_uniform_test_instructions:list|tuple|None|bool=True,
                 normal_test_instructions:list|tuple|None|bool            =True,
                 multivariate_concatenation_delimiter:str|None            =None,
                 #continuous_ordinalized_suffix:str='-ADcont-Ordinalized',  #AD for AnalyzeDataset class, cont for continuous
                 #continuous_binned_suffix:str='-ADcont-Binned',
                 #categorical_ordinalized_suffix:str='-ADcat-Ordinalized',
                 auto_bin:bool|None                                       =None, 
                 multivariate_params:dict|None                            =None,
                 supercat_subcat_params:dict|None                         =None,
                 dropna_cats:bool|None                                    =None,
                 check_assumptions:bool|None                              =None,
                 anova_assumption_check_params:dict|None                  =None,
                 kruskal_assumption_check_params:dict|None                =None,
                 chi2_assumption_check_params:dict|None                   =None
                 ):                             
        """
        where for:
            numnum_meth_alpha_above_instructions:list|tuple|None|bool=True, 
            numcat_meth_alpha_above_instructions:list|tuple|None|bool=True, 
            catcat_meth_alpha_above_instructions:list|tuple|None|bool=True,
            good_of_fit_uniform_test_instructions:list|tuple|None|bool=True,
                True defaults to default params, None and False indicate it is not tested (such as for debugging)

        """   
        # 6 parameters that are for plotting
        # these are defaults that can be modified in the function parameters as a dict of parameters
        # they could be relocated into the function, but put here for future access, such as to modify or put into class __init__()
        # they are relevant in produce_all_plots() and 
        self.cat_univar_params ={
                                'proportions':False,
                                'n_wide':(6,40,4),
                                'super_title':"Univariate Categorical Variables - Reject Good-Of-Fit for Uniform"
                                }    
        self.num_univar_params={
                                'kde':None,
                                'proportions':False,
                                'n_wide':(6,40,4),
                                'super_title':"Univariate Numerical Variables - Reject Normal Distribution",
                                'force_significant_bin_edges':None,
                                'minimize_significant_bins':None,
                                'include_multivariate':True
                                }    
        self.catcat_bivar_params={
                                'n_wide':(6,40,5),
                                'stacked_bars_when_max_bars_is_exceeded':True,
                                'sorted':False,
                                'super_title':"Categoric-To-Categoric Bivariates - Reject Null"
                                }
        self.numnum_bivar_params={
                                'plot_type':'joint',
                                'linreg':True,
                                'super_title':'Numeric Bivariates With Significant Correlation(s)',
                                'plot_type_kwargs':None,
                                'linreg_kwargs':None
                                }
        self.numcat_bivar_params={
                                'plot_type':'boxen', 
                                'n_wide':(6,40,8),
                                'super_title':'Numeric-to-Categoric Bivariates  - Reject Null'
                                }
        self.super_subcat_pairs_params={                      
                                'row_height':2,
                                'cols_per_row':3,
                                'y_tick_fontsize':12,
                                'super_title':"Supercategory-Subcategory - One Categoriec Variable Partitions Another"
                                } 
        #
        #self.continuous_ordinalized_suffix              = continuous_ordinalized_suffix
        #self.continuous_binned_suffix                   = continuous_binned_suffix
        #self.categorical_ordinalized_suffix             = categorical_ordinalized_suffix
        self.multivariate_concatenation_delimiter       = "_|&|_" if multivariate_concatenation_delimiter is None else multivariate_concatenation_delimiter
             
        # test instructions
        self.numnum_meth_alpha_above              = [('pearson',0.6,None),('spearman',0.6,None),('kendall',0.6,None)] if numnum_meth_alpha_above_instructions == True else None if numnum_meth_alpha_above_instructions == False else numnum_meth_alpha_above_instructions # where t tests cannot share the parameter with correlation tests
        self.numcat_meth_alpha_above              = [('kruskal',0.05,None),('anova',0.05,None)] if numcat_meth_alpha_above_instructions == True else None if numcat_meth_alpha_above_instructions == False else numcat_meth_alpha_above_instructions # where variable is not like stat dataframe. dataframe has numric in column 0 and categoric in column 1
        self.catcat_meth_alpha_above              = [('chi2',0.05,None)] if catcat_meth_alpha_above_instructions == True else None if catcat_meth_alpha_above_instructions == False else catcat_meth_alpha_above_instructions
        self.good_of_fit_uniform_test_instructions = (0.05,None) if good_of_fit_uniform_test_instructions == True else None if good_of_fit_uniform_test_instructions == False else good_of_fit_uniform_test_instructions
        self.normal_test_instrucitons             = (0.05,None) if normal_test_instructions == True else None if normal_test_instructions == False else normal_test_instructions

        default_multivariate_params = {'max_n_combination_size':3, 'max_n_combinations':50_000,  'min_combo_size':2}
        if multivariate_params is not None:
            default_multivariate_params.update(multivariate_params)
        self.multivariate_params                  = default_multivariate_params
        default_supercat_subcat_params            =   {'max_evidence':0.2,  'isolate_super_subs':False }
        if supercat_subcat_params is not None:
            default_supercat_subcat_params.update(supercat_subcat_params)
        self.supercat_subcat_params               = default_supercat_subcat_params
        # HOW TO PROCESS DATA, 
        # presently check assumptions is only supported when categoric is involved, 
        # and dropna is automatic in numeric, but optional in categoric (pd.NA and np.nan become text: "NaN")  
        # dropna_cats is a parameter used in gooness of fit. others include dropna instructions in the parameter dict, but default to this     
        self.dropna_cats                          = True if dropna_cats is None else dropna_cats
        self.check_assumptions                    = True if check_assumptions is None else check_assumptions
        # anove assumption checs
        default_anova_assumption_check_params     = {'normality_alpha':0.02, 'homogeneity_alpha':0.02, 'min_n':5, 'iqr_multiplier':3, 'dropna':self.dropna_cats} 
        if anova_assumption_check_params is not None:
            default_anova_assumption_check_params.update(anova_assumption_check_params)
        self.anova_assumption_check_params        = default_anova_assumption_check_params
        # kruskal assumption checks
        default_kruskal_assumption_check_params   = {'levene_alpha':0.01,'ks_alpha':0.01,'return_pseudo':True,'pseudo_test_max_global_ties_ratio': 0.7,'full_pseudo':False,'dropna':self.dropna_cats, 'n_jobs':-2, 'guesstimate':{'rej_max_pct_in_group':0.2,'max_num_outlier_all_reject':3, 'max_pct_reject_total':0.2}}
        if kruskal_assumption_check_params is not None:
            default_kruskal_assumption_check_params.update(kruskal_assumption_check_params)
        self.kruskal_assumption_check_params      = default_kruskal_assumption_check_params
        # chi2 assumption checks
        default_chi2_assumption_check_params      = {'dropna':self.dropna_cats}
        if chi2_assumption_check_params is not None:
            default_chi2_assumption_check_params.update(chi2_assumption_check_params)
        self.chi2_assumption_check_params         = default_chi2_assumption_check_params

        # keep track targets that have been fit   ---> can be used to filter targets in tests for future suport of piecemeal fitting-- such as when revising assumption handling in tests
        self.has_called_fit_column_relationships                   = set() 
        self.has_called_fit_goodness_of_fit_uniform                = set()
        self.has_called_fit_normal                                 = set()
        self.has_called_fit_supercat_subcat_pairs                  = []  # this stores nested lists that have been sorted: [ sorted(list1), sorted(list2), ..., sorted(listn)]
        #                                                                  it doesn't store in order of supercategory subcategory

        # UNIVARIATE
        #numeric univariate columns
        self.reject_null_normal             = set()
        self.fail_to_reject_null_normal     = set()
        # PASS
        #categoric univariate columns
        self.reject_null_good_of_fit        = set()
        self.fail_to_reject_null_good_of_fit= set()
        # BIVARIATE  (used for plotting and for identifying relationships to each column as an individual target

        #numeric to numeric bivariate pairs
        self.above_threshold_corr_numnum    = []  # nested lists are sorted [ [cola,colb], [cola,colb], ..., [cola,colb] ]
        self.below_threshold_corr_numnum    = [] # nested lists are sorted 
        #numeric to categoric bivariate pairs       WHERE (NUMERIC, CATEGORIC) IS ARRANGEMENT
        self.reject_null_numcat             = [] #  NOT sorted 
        self.fail_to_reject_null_numcat     = [] #  NOT sorted 
        #categoric to categoric bivariate pairs
        self.reject_null_catcat             = []
        self.fail_to_reject_null_catcat     = []
        self.supercategory_subcategory_pairs= []  # these are not sorted but the membership list of pairs that have been tested are sorted alphabetically 
        #                                           these are strict: Super, Sub in that order
        # MULTIVARIATE
        self.significant_multivariate_combinations = [] # where data form is: [ [[target,target_dtype],[list_combo_of_n_columns],test_type(s)], [], ...]
        #         .         .         .           .     # Its the same form derived in _prepare_target_plot_data using data stored in self.target_key_feature_meta_vals for plotting

        # ASSUMPTIONS NOT MET FOR HYPOTHESIS TESTING, NO VALID TEST RECORDED
        self.assumptions_not_met             = {'catcat':[],'numnum':[],'numcat':[],'num':set(),'cat':set()} #,'nummulticat':[],'catmulticat':[]}

        # TARGET COLUMNS KEYS AND REJECT OR CORR ABOVE VALUE DICTIONARY
        # where each column is a key, and values are as follows
        # the term 'significant_relationship' is used to describe rejected null or correlation >= threshold
        # not_significant is used to describe results where test(s) failed to reject the null, or correlation was below the threshold
        # THIS DICT CONTAINS MULTI-VARIATE
        # numeric to multi-categoric multivariate groups
        # categoric - categoric multiariate groups
        # for each column
        # a dict to store column by column info
        self.target_key_feature_meta_vals = {}
    # a dict template to set new columns with 
    def _blank_target_dict(self):
        """
        a template dict
        used to initiate columns in self.target_key_feature_meta_vals = {}
        """
        return {
                'significant_numeric_relationships':[],
                'significant_numeric_tests':[],            # indexes align tests to 'significant_numeric_relationships' pairs
                'significant_categoric_relationships':[],  # 
                'significant_categoric_tests':[],            # indexes align tests to 'significant_categoric_relationships' pairs
                'significant_categoric_combination_group_relationship':[],  #[[col1,col2,col3], [], ...]]
                'significant_categoric_combination_group_relationship_test_type':[],  #[test(s)_group_1, test(2)_group_2]
                'paired_to_a_supercategory':[],  # a list of columns it is partitioned by: its supercategory partitioner column
                'paired_to_a_supercategory_tests':[],
                'paired_to_a_subcategory':[],   # a list of columns that are partitioned by it: its subcategory groups column
                'paired_to_a_subcategory_tests':[],
                'target_dtype': [],  # can be one of 'numeric','categoric' to represent the data type of column_1 (in this case) ,
                'max_n_variates_paired_with':[0],   #  int(biggest lenght of combo checked, default: [0]
                'not_significant_numerics':[],
                'not_significant_categorics':[],  # where columns that are grouped together to form significant relationships are ( intended at some point to be )removed and added to 'significant_categoric_relationship' as group(s)
                'is_normal_or_uniform':[],   # for uniform tests: 'reject_uniform' or 'fail_to_reject_uniform'. For normal: 'reject_normal' or 'fail_to_reject_normal'
                #                              relevant in self._prepare_target_plot_data
                'assumptions_not_met':{'catcat':[],'numnum':[],'numcat':[],'num':[],'cat':[]},   # ,'nummulticat':[],'catmulticat':[]},
                'min_bins':[], #
                'min_bins_by_feature':{}
                }
    
    #---------------------------------------------------------------------------------------------------------------------------------
    # HELPER FUNCTIONS 
    #---------------------------------------------------------------------------------------------------------------------------------
    #######################################################################################
    # A HELPER FUNCTION THAT RETURNS FAIL TO REJECT NULL, REJECT NULL, ABOVE CORRELATION THRESHOLD, AND/OR BELOW CORRELATION THRESHOLD
    # it processes one bivariate group at a time: numeric-to-numeric, numeric-to-categoric, or categoric-to-categoric
    #######################################################################################

    def _categorize_bivariate_tests_as_rej_or_failrej(self,
                                                     test_df:pd.DataFrame,
                                                     test_instructions:list|tuple,
                                                     check_assumptions:bool|None=None):
        """
        where test_df is output from CompareColumns().multi_test_column_comparison(); 
            such as :
            test_df = CompareColumns().multi_test_column_comparison(
                                df,
                                numnum_meth_alpha_above=[('pearson',0.6,None),('spearman',0.6,None),('kendall',0.6,None)],
                                numcat_meth_alpha_above=[('kruskal',0.05,None),('anova',0.05,None)],
                                catcat_meth_alpha_above=[('chi2',0.05,None)])
        ------
        returns
        for rejected or correlation above threshold: 
            [[col_a,col_b,test(s)],[...],[...],...] 
            such as [['Purchase Amount (USD)', 'Purchase Amount (USD)_Ordinalized', 'kendall-pearson-spearman'], ...]
        else: w/o test
        """
        if check_assumptions is None:
            check_assumptions = self.check_assumptions 
        # create a default  
        assumptions_not_met_list = []
        
        # check vor valid test direction input. shold be list or list of nested lists/tuples
        if (not isinstance(test_instructions,list) ) and (not isinstance(test_instructions,tuple)):
            raise ValueError(f"expected a list or tuple such as ('test_type',float(threshold),bool|None). Recieved: {test_instructions}")
        if (not isinstance(test_instructions[0],list) ) and (not isinstance(test_instructions[0],tuple)):
            if isinstance(test_instructions[0],str) and isinstance(test_instructions[1],float) and (isinstance(test_instructions[2],bool) or test_instructions[2]==None):
                test_instructions=[test_instructions]

        # get rej null or above threshold coefficients
        list_of_tests = [i[0] for i in test_instructions]
        rej_or_corr_df = test_df.loc[test_df['test'].isin(list_of_tests)]
        failrej_or_below_corr_df = test_df.loc[test_df['test'].isin(list_of_tests)]
        # capture all assumptions_met==False
        if (check_assumptions) and ('assumptions_met' in test_df.columns):
            assumptions_not_met_list = test_df.loc[(test_df['test'].isin(list_of_tests))&(test_df['assumptions_met']==False)][['column_a','column_b']].drop_duplicates(keep='first').to_numpy().tolist()

        for instructions in test_instructions:
            # determine which side of threshold is of interest in this tests 
            if (instructions[0] in ('pearson','spearman','kendall')) and ('Correlation' in test_df.columns):  # Correlation might not be included if targets are cat
                rej_or_corr_df = rej_or_corr_df.loc[~((rej_or_corr_df['test']==instructions[0])&(rej_or_corr_df['Correlation']<instructions[1]))]
                failrej_or_below_corr_df = failrej_or_below_corr_df.loc[~((failrej_or_below_corr_df['test']==instructions[0])&(failrej_or_below_corr_df['Correlation']>=instructions[1]))]
            elif ('P-value' in test_df.columns):   # P-value might not be included if all targets and columns are numeric
                rej_or_corr_df = rej_or_corr_df.loc[~((rej_or_corr_df['test']==instructions[0])&(rej_or_corr_df['P-value']>=instructions[1]))]
                failrej_or_below_corr_df = failrej_or_below_corr_df.loc[~((failrej_or_below_corr_df['test']==instructions[0])&(failrej_or_below_corr_df['P-value']<instructions[1]))]
        
        # filter out all assumptions_met==False & and concat assumption met result to test type
        if (check_assumptions) and ('assumptions_met' in rej_or_corr_df.columns):
            rej_or_corr_df                     =  rej_or_corr_df.loc[rej_or_corr_df['assumptions_met']!=False]
            rej_or_corr_df['test']             =  rej_or_corr_df['test'].astype(str)+':'+rej_or_corr_df['assumptions_met'].astype(str) 
        if (check_assumptions) and ('assumptions_met' in failrej_or_below_corr_df.columns):
            failrej_or_below_corr_df           = failrej_or_below_corr_df.loc[failrej_or_below_corr_df['assumptions_met']!=False]
            failrej_or_below_corr_df['test']   = failrej_or_below_corr_df['test'].astype(str)+':'+failrej_or_below_corr_df['assumptions_met'].astype(str)
        # POTENTIAL ADD P VALUE TO TEST STRING HERE 
        # determine how many tests for each pair
        rej_or_corr_df['num_tests_per_pair'] = rej_or_corr_df.groupby(['column_a','column_b'])['test'].transform('count')
        mx=rej_or_corr_df['num_tests_per_pair'].max()
        if mx>1:
            def concat_test_types(num_tests,test_x,test_y):
                if num_tests>1:
                    return str(test_x)+'|'+str(test_y)
                else:
                    return str(test_x)

            rej_or_corr_df=rej_or_corr_df[['column_a','column_b','num_tests_per_pair','test']]
            new_rej_or_corr_df = rej_or_corr_df
            for match in range(2,mx+1): 
                new_rej_or_corr_df = new_rej_or_corr_df.merge(
                                        rej_or_corr_df,
                                        how='inner',
                                        on=['column_a','column_b','num_tests_per_pair'],
                                        suffixes=('_x', '_y')
                                        )
                new_rej_or_corr_df['test']=new_rej_or_corr_df.apply(lambda x: concat_test_types(x['num_tests_per_pair'],x['test_x'],x['test_y']),axis = 1)
                new_rej_or_corr_df=new_rej_or_corr_df[['column_a','column_b','num_tests_per_pair','test']]
            def split_tests(test):
                return sorted(list(set(test.split('|'))))     # where set ensured unique so that num_tests_per_pair determined in groupby is represented
            new_rej_or_corr_df['test'] = new_rej_or_corr_df['test'].map(split_tests)  
            new_rej_or_corr_df['len_test'] = new_rej_or_corr_df['test'].map(len)
            new_rej_or_corr_df = new_rej_or_corr_df.loc[new_rej_or_corr_df['num_tests_per_pair']==new_rej_or_corr_df['len_test']][['column_a','column_b','test']]
            def clean_test(test):
                return '-'.join(sorted(test))
            new_rej_or_corr_df['test']=new_rej_or_corr_df['test'].map(clean_test)
            new_rej_or_corr_df=new_rej_or_corr_df.drop_duplicates(keep='first')
            rej_or_corr_df=new_rej_or_corr_df
            del new_rej_or_corr_df
        else:
            rej_or_corr_df = rej_or_corr_df[['column_a','column_b','test']].drop_duplicates(subset=['column_a','column_b'],keep='first')

        # get list such as [(column_a,column_b,test),(), ...] to return for rejected or corr above
        # and get temporary list such as [(column_a,column_b),(), ...] to filter non-rej and corr below
        rej_or_corr_cols_tests = rej_or_corr_df.to_numpy().tolist()
        filter_columns = rej_or_corr_df[['column_a','column_b']].to_numpy().tolist()

        # get fail to reject or below corr columns, but NOT TEST TYPE
        failrej_or_below_corr_cols_tests = failrej_or_below_corr_df[['column_a','column_b']].drop_duplicates(keep='first').to_numpy().tolist()
        failrej_or_below_corr_cols_tests = [i for i in failrej_or_below_corr_cols_tests if i not in filter_columns]

        # need to filter to cover cases where one test met assumptions for vars another test didn't
        assumptions_not_met_list = [ vars for vars in assumptions_not_met_list if ((vars not in filter_columns) and (vars not in failrej_or_below_corr_cols_tests))]
        return rej_or_corr_cols_tests, failrej_or_below_corr_cols_tests, assumptions_not_met_list




    #######################################################################################
    # A FUNCTION TO PROCESS USE self._categorize_bivariate_tests_as_rej_or_failrej ITERATIVELY 
    # TO UPDATE         self.above_threshold_corr_numnum, self.below_threshold_corr_numnum, self.reject_null_numcat
    #       AND  self.fail_to_reject_null_numcat, self.reject_null_catcat, self.fail_to_reject_null_catcat 
    # AND TO CREATE A TARGET PROFILE FOR EACH VARIABLE: self.target_key_feature_meta_vals
    #######################################################################################



        
    def _update_model_with_test_df_to_col_pairs_and_cols_as_targets(self,
                                                test_df:pd.DataFrame,
                                                targets:list|None=None, 
                                                check_assumptions:bool|None=None):
        """
        a function that calls self._categorize_bivariate_tests_as_rej_or_failrej iteratively
        and updates class variables
        """

        test_instructions_list = [instruction for instruction in [self.numnum_meth_alpha_above,self.numcat_meth_alpha_above,self.catcat_meth_alpha_above] if instruction != None]
        
        # a helper function that updates class instances instead of overwriting in cases where fit is called multiple times, such as to prevent data loss
        def class_instance_updater(new_vals, 
                                   existing_instance, 
                                   sort_them:bool=False):
            """
            return a list that can replace the existing object: parameter: existing_instace
            does not modify existing in place
            """
            # only need to check if it's not the first fit
            # in that case existing_instance will be [] on first fit
            if existing_instance:
                vals_to_add = []
                for pair in new_vals:
                    # case of [numeric, categoric]
                    if sort_them==False:
                        is_new_pair=True
                        for existing in existing_instance:
                            if ((pair[0]==existing[0]) and (pair[1]==existing[1])) or ((pair[0]==existing[1]) and (pair[1]==existing[0])):
                                is_new_pair=False
                                break
                        if is_new_pair==True:
                            vals_to_add.append(pair)
                    # cases of [numeric, numeric] or [categoric, categoric]
                    else:
                        if len(pair)>=3:
                            npair = sorted(pair[:2].copy())+pair[2:].copy()
                        else:
                            npair = sorted(pair.copy())
                        if npair not in existing_instance:
                            vals_to_add.append(npair)
                result = existing_instance + vals_to_add 
            else:
                if sort_them==True:
                    if len(new_vals[0])>2:
                        new_vals = [sorted(vl[:2].copy())+vl[2:] for vl in new_vals]
                    else:
                        new_vals = [sorted(vl.copy()) for vl in new_vals]
                result = new_vals
            return result
        
        def assumptions_not_met_target_updater(pair,_targets,key):
            """
            keys: 'catcat', 'numnum','numcat'
            """
            if ( (not _targets) or (pair[0] in _targets) ) :
                if self.target_key_feature_meta_vals.get(pair[0],None) is None:
                    self.target_key_feature_meta_vals[pair[0]] = self._blank_target_dict()                                                                                  
                    self.target_key_feature_meta_vals[pair[0]]['assumptions_not_met'][key].append(pair[1]) 
                elif pair[1] not in self.target_key_feature_meta_vals[pair[0]]['assumptions_not_met'][key]:
                    self.target_key_feature_meta_vals[pair[0]]['assumptions_not_met'][key].append(pair[1])                
            if ( (not _targets) or (pair[1] in _targets) ) :
                if self.target_key_feature_meta_vals.get(pair[1],None) is None:
                    self.target_key_feature_meta_vals[pair[1]] = self._blank_target_dict()
                    self.target_key_feature_meta_vals[pair[1]]['assumptions_not_met'][key].append(pair[0]) 
                elif pair[0] not in self.target_key_feature_meta_vals[pair[1]]['assumptions_not_met'][key]:
                    self.target_key_feature_meta_vals[pair[1]]['assumptions_not_met'][key].append(pair[0])  
            return  
          
          
        for instructions in test_instructions_list:
            # returns lists: [[num, cat, test],[],...], [[num, cat],[],...] Note there is no test in fail to reject: not_significant
            # where assumptions not met is [] if check_assumptions==False, or 'assumptions_met' not in test_df columns, or if none exist
            significant , not_significant, assumptions_not_met_list = self._categorize_bivariate_tests_as_rej_or_failrej(test_df=test_df,
                                                                                            test_instructions=instructions,
                                                                                            check_assumptions=check_assumptions)
                
            # use the test instrucitons to determine where to put each column: categorical or numeric
            samp_test=instructions[0][0]

            if assumptions_not_met_list:
                if samp_test in ('pearson','spearman','kendall','welch','student'):
                    _key = 'numnum'
                elif samp_test in ('kruskal','anova'):
                    _key = 'numcat'
                elif samp_test in ('chi2'):
                    _key = 'catcat'
                if _key!='numcat':
                    for pair in assumptions_not_met_list:
                        assumptions_not_met_target_updater(pair, targets, _key)
                    assumptions_not_met_update_list = [sorted(pair.copy()) for pair in assumptions_not_met_list if sorted(pair.copy()) not in self.assumptions_not_met[_key]]
                else:
                    for pair in assumptions_not_met_list:
                        assumptions_not_met_target_updater(pair, targets ,_key)
                    assumptions_not_met_update_list = [pair for pair in assumptions_not_met_list if pair not in self.assumptions_not_met[_key]]
                self.assumptions_not_met[_key]+=assumptions_not_met_update_list
    
            # store relationships and test for each column
            # map col_a and col_b to target col's value_dict in self.target_key_feature_meta_vals
            # begin with significant relationships                                                                            
            if significant:
                if samp_test in ('pearson','spearman','kendall','welch','student'):
                    left_destination='significant_numeric_relationships'
                    left_test_destination = 'significant_numeric_tests'
                    right_destination='significant_numeric_relationships'
                    right_test_destination = 'significant_numeric_tests'
                    # update model only if not already included in model
                    self.above_threshold_corr_numnum = class_instance_updater(significant, self.above_threshold_corr_numnum, sort_them=True)              
                      
                elif samp_test in ('kruskal','anova'):
                    left_destination='significant_numeric_relationships'
                    left_test_destination = 'significant_numeric_tests'
                    right_destination='significant_categoric_relationships'
                    right_test_destination = 'significant_categoric_tests'
                    # update model only if not already included in model
                    self.reject_null_numcat = class_instance_updater(significant, self.reject_null_numcat, sort_them=False)        

                elif samp_test in ('chi2'):
                    left_destination='significant_categoric_relationships'
                    left_test_destination = 'significant_categoric_tests'
                    right_destination='significant_categoric_relationships'
                    right_test_destination = 'significant_categoric_tests'
                    # update model only if not already included in model
                    self.reject_null_catcat = class_instance_updater(significant, self.reject_null_catcat, sort_them=True)                     

                            
                else:
                    raise ValueError(f'{samp_test} not recognized as a test type. Recognized are: pearson,spearman,kendall,welch,student,kendall,anova,chi2')
                #############################################################################################
                # there is some chance of recalculating pairs such as if one is target adn the other isn't, then 
                # a secong call is made targeting the previous non-target
                # but it is simple here because the target is tested agains all posible, the non-target is only test agains the target
                # math could be relplaced with member searches that locate previous targets that paired w curr target but were not recorded for curr target
                #############################################################################################
                for pair_test in significant:
                    # initialize targets in the dict if the aren't already initialized
                    left_col, right_col, test_dash_tests = pair_test[0], pair_test[1], pair_test[2]
                    # left col
                    if not targets or (left_col in targets):
                        if left_col not in self.target_key_feature_meta_vals.keys():
                            self.target_key_feature_meta_vals[left_col]=self._blank_target_dict()
                        if not self.target_key_feature_meta_vals[left_col]['target_dtype']:
                            if left_destination=='significant_numeric_relationships':
                                self.target_key_feature_meta_vals[left_col]['target_dtype']=['numeric']
                            elif left_destination=='significant_categoric_relationships':
                                self.target_key_feature_meta_vals[left_col]['target_dtype']=['categoric']
                        if right_col not in self.target_key_feature_meta_vals[left_col][right_destination]:
                            self.target_key_feature_meta_vals[left_col][right_destination].append(right_col)
                            self.target_key_feature_meta_vals[left_col][right_test_destination].append(test_dash_tests)
                    # right col
                    if not targets or (right_col in targets):
                        if right_col not in self.target_key_feature_meta_vals.keys():
                            self.target_key_feature_meta_vals[right_col]=self._blank_target_dict()
                        if not self.target_key_feature_meta_vals[right_col]['target_dtype']:
                            if right_destination=='significant_numeric_relationships':
                                self.target_key_feature_meta_vals[right_col]['target_dtype']=['numeric']
                            elif right_destination=='significant_categoric_relationships':
                                self.target_key_feature_meta_vals[right_col]['target_dtype']=['categoric']
                        if left_col not in self.target_key_feature_meta_vals[right_col][left_destination]:
                            self.target_key_feature_meta_vals[right_col][left_destination].append(left_col)
                            self.target_key_feature_meta_vals[right_col][left_test_destination].append(test_dash_tests)
          
            
            # repeat for insignificant relationships
            if not_significant:  
                if samp_test in ('pearson','spearman','kendall','welch','student'):
                    left_destination='not_significant_numerics'
                    right_destination='not_significant_numerics'
                    # update model only if not already included in model
                    self.below_threshold_corr_numnum = class_instance_updater(not_significant, self.below_threshold_corr_numnum, sort_them=True)  

                elif samp_test in ('kruskal','anova'):
                    left_destination='not_significant_numerics'
                    right_destination='not_significant_categorics'
                    # update model only if not already included in model
                    self.fail_to_reject_null_numcat = class_instance_updater(not_significant, self.fail_to_reject_null_numcat, sort_them=False)  
                    
                elif samp_test in ('chi2'):
                    left_destination='not_significant_categorics'
                    right_destination='not_significant_categorics'  
                    # update model only if not already included in model
                    self.fail_to_reject_null_catcat = class_instance_updater(not_significant, self.fail_to_reject_null_catcat, sort_them=True) 
                else:
                    raise ValueError(f'{samp_test} not recognized as a test type. Recognized are: pearson,spearman,kendall,welch,student,kendall,anova,chi2')      
                
                # initialize targets in the dict if the aren't already initialized
                for pair_test in not_significant:                    
                    left_col, right_col = pair_test[0], pair_test[1]
                    # left col
                    if (not targets) or (left_col in targets):
                        if left_col not in self.target_key_feature_meta_vals.keys():
                            self.target_key_feature_meta_vals[left_col]=self._blank_target_dict()
                        if not self.target_key_feature_meta_vals[left_col]['target_dtype']:
                            if left_destination=='not_significant_numerics':
                                self.target_key_feature_meta_vals[left_col]['target_dtype']=['numeric']
                            elif left_destination=='not_significant_categorics':
                                self.target_key_feature_meta_vals[left_col]['target_dtype']=['categoric']
                        if right_col not in self.target_key_feature_meta_vals[left_col][right_destination]:
                            self.target_key_feature_meta_vals[left_col][right_destination].append(right_col)

                    # right col
                    if (not targets) or (right_col in targets):
                        if right_col not in self.target_key_feature_meta_vals.keys():
                            self.target_key_feature_meta_vals[right_col]=self._blank_target_dict()
                        if not self.target_key_feature_meta_vals[right_col]['target_dtype']:
                            if right_destination=='not_significant_numerics':
                                self.target_key_feature_meta_vals[right_col]['target_dtype']=['numeric']
                            elif right_destination=='not_significant_categorics':
                                self.target_key_feature_meta_vals[right_col]['target_dtype']=['categoric']
                        if left_col not in self.target_key_feature_meta_vals[right_col][left_destination]:
                            self.target_key_feature_meta_vals[right_col][left_destination].append(left_col)
        return self

    #######################################################################################
    # A FUNCTION TO PROCESS CATEGORIC AND NUMERIC TARGETS
    #
    #######################################################################################
    def _combine_targets(self,numeric_target:list|tuple|str|None, categoric_target:list|tuple|str|None):
        """
        a helper function to process targets
        """            
        targets=[]
        if isinstance(numeric_target,str):
            numeric_target = [numeric_target]
        if numeric_target!=None:
            targets = targets+numeric_target
        if isinstance(categoric_target,str):
            categoric_target = [categoric_target]
        if categoric_target!=None:
            targets = targets+categoric_target
        return targets

    #######################################################################################
    # A FUNCTION TO CONCATINATE COLUMNS INTO ONE OBJECT TYPE COLUMN THAT CAN BE USED FOR TESTS
    #
    #######################################################################################


    def _concatenate_columns_axis_1(self,
                                    dataframe:pd.DataFrame, 
                                    column_combo:list|tuple,
                                    header_divider:str="_&_"):
        """
        takes a dataframe 
        list of columns in the dataframe that should be concatinated into one Series with dtype object
        a header divider as string that divides variable headers and variabel values
        returns pd.Series with name as concatinated headers
        """

        new_col_header = column_combo[0]
        result = dataframe[column_combo[0]]
        other_columns=column_combo[1:]
        for index in range(len(other_columns)):
            new_col_header = new_col_header + header_divider + other_columns[index]
            result = result.astype(str)+header_divider+dataframe[other_columns[index]].astype(str)
        result.name=new_col_header
        return result
    
    #######################################################################################
    # A FUNCTION THAT CONSIDERS COLUMN PAIRS ONE BY ONE TO DETECT SUPERCATEGORY-SUBCATEGORY RELATIONSHIPS
    # used in fit_organize_supercat_subcat_pairs() to fit/reorganize the model
    #######################################################################################

    def _are_supercat_subcats(self,
                                           data:pd.DataFrame,
                                           max_evidence:float=0.2,  
                                           pairs_list:list|tuple|None=None ):
                              
        """
        determines relationships
        where max_evidence is shannon_entropy, and less evidence is stronger support that there is a supercategory-subcategory relationship
        by default, this analyzed column pairs in self.fail_to_reject_null_catcat and updates the model, such as to remove super-subcat pairs
        within the targets dict and put them in super or sub value lists
        to override the default, input pairs_list, and the function will compute only input pairs
        
        Variables are tested in both directions  -->Subcat -->supercat

        returns a list of [[supercat, subcat], [], ...], where each pair is a supercat-subcat relationship
        and a list of [True, False, ...] that coresponds to (input, or self.fail_to_reject_null_catcat)
        """

        # a function to retrieve the index if exists
        def index_or_None(member:str,lst:list):
            """
            """
            res = None
            for ind,val in enumerate(lst):
                if val == memeber:
                    res=ind
                    break
                return res

        # determine what pairs to test
        if pairs_list is None:
            pairs = self.reject_null_catcat
        else: 
            pairs = pairs_list
        
        # stores [True, False, True, ...] for pairs in input pairs_list
        is_supsub = []
        # reorder list to indicate relationship: [[supercat, subcat], [], ...]
        # where new oder always has supercat first
        list_reordered = []
        for pair in pairs:
            first_way=self.evidence_is_supercat_given_subcat(data, pair[0], pair[1])
            first_way=(first_way<=max_evidence)
            second_way=False
            if not first_way:
                second_way=self.evidence_is_supercat_given_subcat(data, pair[1], pair[0] )
                second_way=(second_way<=max_evidence)
            is_partitioned = ( first_way or second_way)
            is_supsub.append(is_partitioned)
            if first_way:
                order=[pair[0], pair[1]]
            else:
                order=[ pair[1], pair[0]]
            list_reordered.append(order)
        return list_reordered, is_supsub  
    
    #---------------------------------------------------------------------------------------------------------------------------------
    # FIT FUNCTIONS: 
    # 1) fit_column_relationships
    # 2) fit_multivariate_column_relationships
    # 3) fit_supercat_subcat_pairs
    #---------------------------------------------------------------------------------------------------------------------------------    

    #######################################################################################
    # A FUNCTION TO FIT COLUMN RELATIONSHIPS BASED ON NULL HYPOTHESIS THRESHOLDS AND CORRELATIONS
    # numeric-to-numeric, numeric-to-categoric, or categoric-to-categoric, categoric_univaraite
    # NO multivariate beyond bivariate, but this must be run before the multivariate can be processed
    # target columns w/ significant and not-significant column relationships: self.target_key_feature_meta_vals
    #######################################################################################

    def fit_column_relationships(self, 
                                 df: pd.DataFrame,
                                 numeric_columns=None,
                                 categoric_columns=None,
                                 numeric_target:list|tuple|str|None=None,
                                 categoric_target:list|tuple|str|None=None,
                                cols_to_exclude_from_targets:str|list|None=None,
                                check_assumptions:bool|None=None,
                                anova_assumption_check_params:dict|None=None,
                                kruskal_assumption_check_params:dict|None=None,
                                chi2_assumption_check_params:dict|None=None
                                 ):
        """
        parameters:
            df: pd.DataFrame: the dataframe to process. It will be statistically analyzed
            numeric_columns, categoric_columns, numeric_target, categoric_target
                are None by default but can be specified with lists (list or string for targets)
        this function  updates: 
                self.above_threshold_corr_numnum   
                self.below_threshold_corr_numnum   
                self.reject_null_numcat             
                self.fail_to_reject_null_numcat     
                self.reject_null_catcat             
                self.fail_to_reject_null_catcat 
                    which can be all be used for plotting 
        it also updates   
                self.target_key_feature_meta_vals
                    which provides per column relations that aid in ML
                        and can be used in to process multivariate relations beyond bivariate, 
                            but only bivariate is processid within this function
        """
        if anova_assumption_check_params is None:
            anova_assumption_check_params=self.anova_assumption_check_params
        if kruskal_assumption_check_params is None:
            kruskal_assumption_check_params=self.kruskal_assumption_check_params
        if chi2_assumption_check_params is None:
            chi2_assumption_check_params=self.chi2_assumption_check_params

        if check_assumptions is None:
            check_assumptions = self.check_assumptions 

        # compute the statistic df to get p-values and correlations
        test_df = self.multi_test_column_comparison(
                                df,
                                numnum_meth_alpha_above=self.numnum_meth_alpha_above,
                                numcat_meth_alpha_above=self.numcat_meth_alpha_above,
                                catcat_meth_alpha_above=self.catcat_meth_alpha_above,
                                numeric_columns=numeric_columns,
                                categoric_columns=categoric_columns,
                                numeric_target=numeric_target,
                                categoric_target=categoric_target,
                                cols_to_exclude_from_targets=cols_to_exclude_from_targets,
                                check_assumptions=check_assumptions,
                                anova_assumption_check_params=anova_assumption_check_params,
                                kruskal_assumption_check_params=kruskal_assumption_check_params,
                                chi2_assumption_check_params=chi2_assumption_check_params 
                                )
        # identify target(s) if present
        targets = self._combine_targets(numeric_target=numeric_target, 
                                        categoric_target=categoric_target)

        # compute significant and not-significant bivariate pairs -- w/ test for significant pairs
        self._update_model_with_test_df_to_col_pairs_and_cols_as_targets(test_df=test_df,
                                                                        targets=targets,
                                                                        check_assumptions=check_assumptions)
        # update class object that tracks variables that have been fit
        if targets:
            t_update = set(targets)
        else: 
            t_update = set()
            if (self.numcat_meth_alpha_above is not None) or ((self.numnum_meth_alpha_above is not None) and (self.catcat_meth_alpha_above is not None)):
                headers=set(df.columns)
                t_update.update(headers)
            else:
                if numnum_meth_alpha_above is not None:
                    nums= set(df.select_dtypes([np.number,'number']))
                    t_update.update(nums)
                if catcat_meth_alpha_above is not None:
                    cats= set(df.select_dtypes(['object','category']))
                    t_update.update(cats)                              
        self.has_called_fit_column_relationships.update(t_update)

        return self
    
    #######################################################################################
    # A FUNCTION TO TEST GOODNESS OF FIT FOR UNIFORM DISTRIBUTION
    # UPDATES   self.reject_null_good_of_fit AND self.fail_to_reject_null_good_of_fit
    #######################################################################################
        
    def fit_goodness_of_fit_uniform(self,
                                    df:pd.DataFrame,
                                    categoric_columns:str|list|None=None,
                                    dropna:bool|None=None,
                                    check_assumptions:bool|None=None):
        """
        """
        if dropna is None:
            dropna=self.dropna_cats
        if check_assumptions is None:
            check_assumptions = self.check_assumptions
        
        # avoid refitting
        if not categoric_columns:
            categoric_columns = [ col for col in df.select_dtypes(['object','category']).columns if col not in self.has_called_fit_goodness_of_fit_uniform ]
        
        # test categorical univariate agianst a uniform distribution
        good_of_fit_uniform_df = self.filterable_all_column_goodness_of_fit(
                                                                            df,
                                                                            cat_alpha_above=self.good_of_fit_uniform_test_instructions,
                                                                            categoric_columns=categoric_columns,
                                                                            dropna=dropna,
                                                                            check_assumptions=check_assumptions)
        good_of_fit_threshold=self.good_of_fit_uniform_test_instructions[0]
        # reject and fail to reject
        if check_assumptions==False:
            # reject
            reject_null_gof                  = set(list(good_of_fit_uniform_df.loc[good_of_fit_uniform_df['P-value'] <good_of_fit_threshold]['category'].values))
            # fail to reject
            fail_to_reject_null_gof          = set(list(good_of_fit_uniform_df.loc[good_of_fit_uniform_df['P-value']>=good_of_fit_threshold]['category'].values))
        else:
            # reject
            reject_null_gof                  = set(list(good_of_fit_uniform_df.loc[(good_of_fit_uniform_df['P-value'] <good_of_fit_threshold) & (good_of_fit_uniform_df['assumptions_met']!=False)]['category'].values))
            # fail to reject
            fail_to_reject_null_gof          = set(list(good_of_fit_uniform_df.loc[(good_of_fit_uniform_df['P-value']>=good_of_fit_threshold) & (good_of_fit_uniform_df['assumptions_met']!=False)]['category'].values))
            assumptions_not_met              = set(list(good_of_fit_uniform_df.loc[good_of_fit_uniform_df['assumptions_met']==False]['category'].values))
            # assumptions_not_met              = assumptions_not_met.difference(reject_null_gof)
            # assumptions_not_met              = assumptions_not_met.difference(fail_to_reject_null_gof)
            self.assumptions_not_met['cat'].update(assumptions_not_met) 
            # update targets dict
            for col in assumptions_not_met:
                if self.target_key_feature_meta_vals.get(col,None)==None:
                    self.target_key_feature_meta_vals[col]=self._blank_target_dict()
                self.target_key_feature_meta_vals[col]['assumptions_not_met']['cat']=[col]
                if (col in self.has_called_fit_column_relationships) and (not self.target_key_feature_meta_vals[col]['is_normal_or_uniform']):
                    self.target_key_feature_meta_vals[col]['is_normal_or_uniform'] = ['assumptions_not_met_uniform']

        # track reject
        self.reject_null_good_of_fit.update(reject_null_gof)
        # track that what's been called
        self.has_called_fit_goodness_of_fit_uniform.update(reject_null_gof)
        # track fail
        self.fail_to_reject_null_good_of_fit.update(fail_to_reject_null_gof)
        # track that what's been called
        self.has_called_fit_goodness_of_fit_uniform.update(fail_to_reject_null_gof)

        # update target dict: self.target_key_feature_meta_vals
        for col in list(reject_null_gof):
            if self.target_key_feature_meta_vals.get(col,None)==None:
                self.target_key_feature_meta_vals[col]=self._blank_target_dict()
            if (col in self.has_called_fit_column_relationships) and (not self.target_key_feature_meta_vals[col]['is_normal_or_uniform']):
                self.target_key_feature_meta_vals[col]['is_normal_or_uniform']=['reject_uniform']
        for col in list(fail_to_reject_null_gof):
            if self.target_key_feature_meta_vals.get(col,None)==None:
                self.target_key_feature_meta_vals[col]=self._blank_target_dict()
            if (col in self.has_called_fit_column_relationships) and (not self.target_key_feature_meta_vals[col]['is_normal_or_uniform']):
                self.target_key_feature_meta_vals[col]['is_normal_or_uniform']=['fail_to_reject_uniform']
        return  self
    
    #######################################################################################
    # A FUNCTION TO TEST FOR NORMAL DISTRIBUTION
    # UPDATES   self.reject_null_normal AND self.fail_to_reject_null_normal
    #######################################################################################
        
    def fit_normal(self,
                    df:pd.DataFrame,
                    numeric_columns:str|list|None=None):
        """
        
        """

        
        # avoid refitting
        if not numeric_columns:
            numeric_columns = [ col for col in df.select_dtypes([np.number,'number']).columns if col not in self.has_called_fit_normal]
        # test numeric vars against a normal distripution
        normal_df = self.filterable_all_column_normality_test(df,
                                                                cat_alpha_above=self.normal_test_instrucitons,
                                                                numeric_columns=numeric_columns,
                                                                cols_to_exclude_from_targets=None,)        

        normal_threshold=self.normal_test_instrucitons[0]
        # reject and fail to reject
        #if check_assumptions==False:  # NOT SUPPORTED AT THIS TIME
        # reject
        reject_null_normal                  = set(list(normal_df.loc[normal_df['P-value'] <normal_threshold]['numeric'].values))
        # fail to reject
        fail_to_reject_null_normal          = set(list(normal_df.loc[normal_df['P-value']>=normal_threshold]['numeric'].values))
        """
        # THIS ASSUMPTIONS SNIPPET IS COPPIED FROM GOODNESS OF FIT, SO WOULN'T WORK AS IS
        else:
            # reject
            reject_null_gof                  = set(list(normal_df.loc[(normal_df['P-value'] <normal_threshold) & (normal_df['assumptions_met']!=False)]['category'].values))
            # fail to reject
            fail_to_reject_null_gof          = set(list(normal_df.loc[(normal_df['P-value']>=normal_threshold) & (normal_df['assumptions_met']!=False)]['category'].values))
            assumptions_not_met              = set(list(normal_df.loc[normal_df['assumptions_met']==False]['category'].values))
            # assumptions_not_met              = assumptions_not_met.difference(reject_null_gof)
            # assumptions_not_met              = assumptions_not_met.difference(fail_to_reject_null_gof)
            self.assumptions_not_met['num'].update(assumptions_not_met) 
            # update targets dict
            for col in assumptions_not_met:
                if self.target_key_feature_meta_vals.get(col,None)==None:
                    self.target_key_feature_meta_vals[col]=self._blank_target_dict()
                self.target_key_feature_meta_vals[col]['assumptions_not_met']['num']=[col]"""
                
        # track reject
        self.reject_null_normal.update(reject_null_normal)
        # track that what's been called
        self.has_called_fit_normal.update(reject_null_normal)
        # track fail
        self.fail_to_reject_null_normal.update(fail_to_reject_null_normal)
        # track that what's been called
        self.has_called_fit_normal.update(fail_to_reject_null_normal)

        # update target dict: self.target_key_feature_meta_vals
        for col in list(reject_null_normal):
            if self.target_key_feature_meta_vals.get(col,None)==None:
                self.target_key_feature_meta_vals[col]=self._blank_target_dict()
            if (not self.target_key_feature_meta_vals[col]['is_normal_or_uniform']) and (col in self.has_called_fit_column_relationships):
                self.target_key_feature_meta_vals[col]['is_normal_or_uniform']=['reject_normal']
        for col in list(fail_to_reject_null_normal):
            if self.target_key_feature_meta_vals.get(col,None)==None:
                self.target_key_feature_meta_vals[col]=self._blank_target_dict()
            if (not self.target_key_feature_meta_vals[col]['is_normal_or_uniform']) and (col in self.has_called_fit_column_relationships):
                self.target_key_feature_meta_vals[col]['is_normal_or_uniform']=['fail_to_reject_normal']
        return  self
    
    
    #######################################################################################
    # A FUNCTION TO FIT MULTIVARIATE COLUMN RELATIONSHIPS FOR COLUMNS THAT WERE NOT INCLUDED IN BIVARIATE RELATIONSHIPS IN fit_column_relationships()
    # This concatinates categoricat variables and performs Chi2, ANOVA, and Kruskal-Wallis tests
    #######################################################################################

    def fit_multivariate_column_relationships(self,
                                              df:pd.DataFrame,
                                              targets:list|tuple|str|None,
                                              numeric_targets:bool=True,  # if True, targets are from self.target_key_feature_meta_vals
                                              catigorci_targets:bool=True,  # if True, targets are from self.target_key_feature_meta_vals
                                              max_n_combination_size:int|None=3,
                                              max_n_combinations:int|None=50_000,
                                              min_combo_size:int=2,
                                            cols_to_exclude_from_targets:str|list|None=None,
                                            check_assumptions:bool|None=None,
                                            anova_assumption_check_params:dict|None=None,
                                            kruskal_assumption_check_params:dict|None=None,
                                            chi2_assumption_check_params:dict|None=None
                                              ):
        """
        this iterates through 
            self.target_key_feature_meta_vals = { 
                    column_1:{'not_significant_categorics':[]  
                             'target_dtype': []  # can be one of ['numeric'],['categoric'] to represent the data type of column_1 (in this case)
                             },
        and removes each columns from 'not_significant_categorics' when:
            the colums are in categorical combination(s) that rejects the null hypothosis, [columns are removed from consideration after all combos in combo_size have been considered, hence, it may be that many are in several combos of the same size]
                and creates keys and puts those columns into lists that get nested in: 
                self.target_key_feature_meta_vals = {
                        column_1:{'significant_categoric_combination_group_relationship':[[combo,group,1],[col1,col2,col3]]  
                                'significant_categoric_combination_group_relationship_test_type':[test(s)_group_1, test(2)_group_2]  # the tests are stored in the same index location as the group combo
                                },
        parameters:
                    targets:list|tuple|str|None --> target columns. Each target will be tested/compared in relation to combinations: 
                            Combinations are a range of [min_combo_size, max_n_combination_size] concatinated categorical variable
                            NOTE: if targets is not None, targets override parameters: numeric_targets and categoric_targets
                            if not None, only targets will be considered
                    The following are not used when targets is not None: 
                            numeric_targets:bool=True, --> give autodetect instructions. If False, numeric targets won't be included
                            categoric_targets:bool=True, -->  give autodetect instructions. If False, categoric targets won't be included
                            by default both are True, hence all columns are considered as targets
                    The following limit the number of combinations and combination sizes
                            max_n_combination_size:int|None, --> Set the max lenght of individual combinations
                            max_n_combinations:int|None=50_000 --> The max number of combinations, per variable for each combination size in the range [2, max_n_combination_size]
                                    the range is computed from smalled to largest combination size. 
                                    If the combination size is large enough to excede max_n_combinations, iteration stops and ONLY lower combination sizes are returned
                    min_combo_size defaults to minimum lenght of combos == 2, but it can take values other than 2
                    mulitvariate_test_significant_in_many_groups False removes any column from potential columns as soon as it is tested in a group that has a significant relationship 
        """
        default_anova = self.anova_assumption_check_params.copy()
        if anova_assumption_check_params is not None:
            default_anova.update(anova_assumption_check_params)
        anova_assumption_check_params = default_anova
        default_kruskal =self.kruskal_assumption_check_params.copy()
        if kruskal_assumption_check_params is not None:
            default_kruskal.update(kruskal_assumption_check_params)
        kruskal_assumption_check_params = default_kruskal
        default_chi = self.chi2_assumption_check_params.copy()
        if chi2_assumption_check_params is not None:
            default_chi.update(chi2_assumption_check_params)
        chi2_assumption_check_params = default_chi

        if check_assumptions is None:
            check_assumptions = self.check_assumptions 
        
        # fit_column_relationships() has to have been called first:
        if not self.has_called_fit_column_relationships:
            raise ValueError("fit_column_relationships() needs to run before this function.")
        
        # make a list of target variables if is/are string(s)
        if isinstance(targets,str):
            targets=[targets]

        # use stored columns, or update stored columns
        # if targets is an empty list or None
        if (not targets) or (targets==None):
            targets = list(self.has_called_fit_column_relationships)
        else:
            # update class object that tracks targets that have been fit 
            t_update=set( i for i in targets)  
            self.has_called_fit_column_relationships.update(t_update)     


        # iterate through targets list
        for target in targets:    
            # compute pairs, then remove those columns and compute 3's with left over, etcetera for 4's, 5's sorforth if/when parameters consider byond 2's
            #iterate while max num possible combos is not exceded: max_n_combinations
            biggest_combo_size_checked=0
            population_size = len(set(self.target_key_feature_meta_vals[target]['not_significant_categorics']))
            curr_combo_size = min_combo_size
            curr_max_len_combo = min(population_size,max_n_combination_size) # used to update curr_n_possible_combinations and elsewhere
            # a statement to handle cases when there is no population
            if curr_combo_size<=population_size:
                curr_n_possible_combinations = calculate_num_combinations(population_size,curr_combo_size) 
            else: 
                curr_n_possible_combinations = 0
            while (curr_combo_size<=curr_max_len_combo) and (curr_n_possible_combinations<=max_n_combinations) and (population_size>=curr_combo_size):
                # BEGIN INNER LOOP TO ITERATE THROUGH THESE COMBOS
                # iterate through categoric variables that have not yet been included in a reject null test in previous combo sizes
                combo_generator = combinations(list(set(self.target_key_feature_meta_vals[target]['not_significant_categorics'])),curr_combo_size)
                seen_and_significant={}
                for nth_combo in range(curr_n_possible_combinations):
                    try:
                        curr_combo = next(combo_generator)
                    except: #in case the user inputs a float for min combo size.
                        break
                    curr_combo = list(curr_combo)

                    # concatinate the curr_combo columns into one series, pair it in a dataframe with the target,c
                    header_divider                                                 = self.multivariate_concatenation_delimiter  # how concatinated multivariate tests' headers will seperate columns
                    df_to_use_in_concat_func                                       = df[curr_combo].copy()
                    concated_feature_col                                           = self._concatenate_columns_axis_1(dataframe=df_to_use_in_concat_func,
                                                                                                                    column_combo=curr_combo,
                                                                                                                    header_divider=header_divider)
                    target_and_concated_second_col_df                              = df[[target]].copy()
                    concated_feature_col_header                                    = concated_feature_col.name
                    target_and_concated_second_col_df[concated_feature_col_header] = concated_feature_col.astype('object')
                    # Hypothosis test target_and_concated_second_col_df 
                    if self.target_key_feature_meta_vals[target]['target_dtype'] == ['numeric']:
                        test_instructions = self.numcat_meth_alpha_above
                        test_df = self.multi_test_column_comparison(
                                        target_and_concated_second_col_df,
                                        numnum_meth_alpha_above=None,
                                        numcat_meth_alpha_above=self.numcat_meth_alpha_above,
                                        catcat_meth_alpha_above=None,
                                        numeric_columns=None,
                                        categoric_columns=None,
                                        numeric_target=target,
                                        categoric_target=None ,
                                        cols_to_exclude_from_targets=cols_to_exclude_from_targets,
                                        check_assumptions=check_assumptions,
                                        anova_assumption_check_params=anova_assumption_check_params,
                                        kruskal_assumption_check_params=kruskal_assumption_check_params,
                                        chi2_assumption_check_params=chi2_assumption_check_params
                                        )
                    elif self.target_key_feature_meta_vals[target]['target_dtype'] == ['categoric']: 
                        test_instructions = self.catcat_meth_alpha_above
                        test_df = self.multi_test_column_comparison(
                                        target_and_concated_second_col_df,
                                        numnum_meth_alpha_above=None,
                                        numcat_meth_alpha_above=None,
                                        catcat_meth_alpha_above=self.catcat_meth_alpha_above,
                                        numeric_columns=None,
                                        categoric_columns=None,
                                        numeric_target=None,
                                        categoric_target=target ,
                                        cols_to_exclude_from_targets=cols_to_exclude_from_targets,
                                        check_assumptions=check_assumptions,
                                        anova_assumption_check_params=anova_assumption_check_params,
                                        kruskal_assumption_check_params=kruskal_assumption_check_params,
                                        chi2_assumption_check_params=chi2_assumption_check_params
                                        )
                    else:
                        raise ValueError(f"No data type detected for {target}: self.target_key_feature_meta_vals[target]['target_dtype'] should be one of ['categoric'] or ['numeric']")
                    if test_df.shape[0]>0: 
                        significant , not_significant, assumptions_not_met_list = self._categorize_bivariate_tests_as_rej_or_failrej(test_df=test_df,
                                                                                                                                    test_instructions=test_instructions,
                                                                                                                                    check_assumptions=check_assumptions)
                        if significant:  #significant looks like: [[col_a,col_b,test(s)],[...],[...],...] But in this case shorter
                            # the columns are already known: curr_combo: so just extract the test
                            curr_test=''
                            for info in significant:
                                curr_test+=info[2]
                            self.target_key_feature_meta_vals[target]['significant_categoric_combination_group_relationship'].append(sorted(curr_combo.copy()))
                            self.target_key_feature_meta_vals[target]['significant_categoric_combination_group_relationship_test_type'].append(curr_test)
                            # same as in self._prepare_target_plot_data(): [[target_column,'categoric'],col,test]-> [[targ, dtype],sorted(combo),test]
                            item_to_append_to_overall_model_reject_combo_matches = [[target,self.target_key_feature_meta_vals[target]['target_dtype'][0]],sorted(curr_combo.copy()),curr_test]  
                            self.significant_multivariate_combinations.append(item_to_append_to_overall_model_reject_combo_matches)
                            # store columns that won't be passed to the next combo size
                            if seen_and_significant:
                                curr_combo = set(curr_combo)
                                seen_and_significant.update(curr_combo)
                            else:
                                seen_and_significant=set(curr_combo)
                        """if assumptions_not_met_list:
                            # the columns are already known: curr_combo: so just extract the test
                            curr_test=''
                            for info in significant:
                                curr_test+=info[2]
                            _key = self.target_key_feature_meta_vals[target]['target_dtype'][0]
                            loc_key = _key[:3]+'multicat'
                            item_to_append_to_model_assumptions_not_met = [[target,_key],sorted(curr_combo.copy()),curr_test]                        
                            if item_to_append_to_model_assumptions_not_met not in self.assumptions_not_met[loc_key]:
                                self.assumptions_not_met[loc_key].append(item_to_append_to_model_assumptions_not_met)
                            if item_to_append_to_model_assumptions_not_met not in self.target_key_feature_meta_vals[target]['assumptions_not_met'][loc_key]:
                                self.target_key_feature_meta_vals[target]['assumptions_not_met'][loc_key].append(item_to_append_to_model_assumptions_not_met)
                                """
                # after each size, remove combos that have already been in a significant group from 'not_significant_categorics' 
                # the idea is that, otherwise, these groups would re-apear with additional columns that may not really contribute 
                # risk of loss of information is already minimized for cases of replacement because col_a can already be paired as both (col_a,col_b) and (col_a,col_c) 
                # furthermore, because they are removed here, 
                #           some edge cases where fit calls are made peacemeal are avoided, 
                #           other edge cases should be handled elsewhere

                try: 
                    for var in list(seen_and_significant):
                        var_index=self.target_key_feature_meta_vals[target]['not_significant_categorics'].index(var)
                        discard = self.target_key_feature_meta_vals[target]['not_significant_categorics'].pop(var_index)
                except:
                    for var in list(seen_and_significant):
                        var_index=self.target_key_feature_meta_vals[target]['not_significant_categorics'].index(var)
                        if var_index < len( self.target_key_feature_meta_vals[target]['not_significant_categorics'] )-1:
                            self.target_key_feature_meta_vals[target]['not_significant_categorics'] = self.target_key_feature_meta_vals[target]['not_significant_categorics'][:var_index] + self.target_key_feature_meta_vals[target]['not_significant_categorics'][var_index+1:]
                        else:
                            self.target_key_feature_meta_vals[target]['not_significant_categorics'] = self.target_key_feature_meta_vals[target]['not_significant_categorics'][:var_index]

                #update biggest_combo_size_checked, curr_combo_size, curr_n_possible_combinations, and population_size
                biggest_combo_size_checked=curr_combo_size
                curr_combo_size+=1
                population_size=len(list(set(self.target_key_feature_meta_vals[target]['not_significant_categorics'])))
                curr_max_len_combo = min(population_size,max_n_combination_size)
                curr_n_possible_combinations=calculate_num_combinations(population_size,min(curr_combo_size,curr_max_len_combo))  # where curr_max_len_combo is <=population_size by design
            # if the loop was never entered 
            if curr_combo_size==min_combo_size:
                if curr_n_possible_combinations > max_n_combinations:
                    # consider raising max_n_combinations
                    warn(f"\nNo combinations considered for {target.title}.\nConsider increasing max_n_combinations.\n max_n_combinations == {max_n_combinations}, but found {curr_n_possible_combinations} combinations @ min combo size == {min_combo_size}.")
                elif population_size<curr_combo_size:
                    warn(f"\nNo combinations considered for {target.title()}. \nPopulation_size == {population_size}, but minimum combo size is {min_combo_size}")
                # indicate that this target never entered the loop, and biggest size of combos that were checked: 0
                self.target_key_feature_meta_vals[target]['max_n_variates_paired_with']=[0]
            else:
                #add/update a key,value pair to indicate that this target did enter the loop, and biggest size of combos tested
                self.target_key_feature_meta_vals[target]['max_n_variates_paired_with']=[biggest_combo_size_checked]

        return self       
    
    #######################################################################################
    # A FUNCTION TO REMOVE SUBCAT PAIRS FROM REGECT NULL CAT-TO-CAT RESULTS
    # removes from rej null catcat pairs and adds to self.supercategory_subcategory_pairs
    # removes rej null categoric from individual categorical targets and adds to paired_to_supercat (if target is subcat), or to paired_to_subcat(if target is supercat)
    #######################################################################################
        
    def fit_supercat_subcat_pairs(self,
                                    data:pd.DataFrame,
                                    max_evidence:float=0.2,  
                                     pairs_list:list|tuple|None=None,
                                      isolate_super_subs:bool|None=None ):
        """
        calls:  _are_supercat_subcats()
        to detect spercategory-subcategory relationships
        then updates class objects:
            adds to self.supercategory_subcategory_pairs
            removes rej null categoric from individual categorical targets and adds to paired_to_supercat (if target is subcat), or to paired_to_subcat(if target is supercat)
        Variables are tested in both directions  -->Subcat -->supercat     
        if isolate_super_subs == False, super and sub will be added to appropriate storages, but not removed from cat bivariate storages  
        """
        # fit_column_relationships() has to have been called first:
        if not self.has_called_fit_column_relationships:
            raise ValueError("fit_column_relationships() needs to run before this function.")
        
        if isolate_super_subs is None:
            isolate_super_subs=False
        
        sup_subs, true_false_list =  self._are_supercat_subcats(data,
                                           max_evidence=max_evidence,  
                                           pairs_list=pairs_list) 

        def find_member(search_val:str|list, 
                        list_to_search:list, 
                        search_val_type:str, 
                        search_val_len:int|None=None):
            """
            reurns an index location of the search_val in the list_to_search, or returns None
            where search_val_type can be 'string' or 'list'
            it is assumed the vals in list_to_search are the same
            search_val_len is used in cas of list such as search_val==list_to_search[index][:search_val_len] such as to exclude test type when searching self.reject_null_catcat
            if search_val_len is None, only exact matches will be considered
            """
            location = None
            if (search_val_type=='string') or (search_val_len is None):
                for possible_location, possible_value in enumerate(list_to_search):
                        if search_val==possible_value:
                            location=possible_location
                            break
            elif search_val_type=='list':
                for possible_location, possible_value in enumerate(list_to_search):
                        if search_val==possible_value[:search_val_len]:
                            location=possible_location
                            break
            return location  
       


        for tf_bool, cols in zip(true_false_list,sup_subs):
            if tf_bool==True:
                if cols not in self.supercategory_subcategory_pairs:
                    self.supercategory_subcategory_pairs.append(cols)
            # store record of fit columns in class.The pairs should already have been filtered in self._are_supercat_subcats()
            if (sorted(cols.copy()) not in self.has_called_fit_supercat_subcat_pairs):
                self.has_called_fit_supercat_subcat_pairs.append(sorted(cols.copy()))

            
            #identify each
            super, sub = cols[0], cols[1]
            super_test, sub_test =  None, None
            super_index=find_member(super, 
                        self.target_key_feature_meta_vals[sub]['significant_categoric_relationships'],
                        search_val_type='string', 
                        search_val_len=None)
            if super_index is not None:
                # capture test
                super_test = self.target_key_feature_meta_vals[sub]['significant_categoric_tests'][super_index]

            sub_index=find_member(sub, 
                                self.target_key_feature_meta_vals[super]['significant_categoric_relationships'], 
                                    search_val_type='string', 
                                    search_val_len=None)
            if sub_index is not None:
                # capture test
                sub_test = self.target_key_feature_meta_vals[super]['significant_categoric_tests'][sub_index]

            # >remove supercat subcat pairs from self.reject_null_catcat  the next loop stores them based on targets
            if isolate_super_subs==True:
                index_loc = find_member(sorted(cols.copy()), 
                                                self.reject_null_catcat, 
                                                search_val_type='list', 
                                                search_val_len=2)
                if index_loc is not None:
                    try:
                        discard = self.reject_null_catcat.pop(index_loc)
                    except:
                        if index_loc<len(self.reject_null_catcat)-1:
                            self.reject_null_catcat = self.reject_null_catcat[:index_loc] + self.reject_null_catcat[index_loc+1:]
                        else:
                            self.reject_null_catcat = self.reject_null_catcat[:index_loc]      

                
                # update targets by removing super/sub and capturing tests. both will be used to update super/sub values in dict
                if super in self.has_called_fit_column_relationships:

                        try:
                            discard_col=self.target_key_feature_meta_vals[super]['significant_categoric_relationships'].pop(sub_index)
                            discard_test=self.target_key_feature_meta_vals[super]['significant_categoric_tests'].pop(sub_index)
                        except:
                            if sub_index<(len(self.target_key_feature_meta_vals[super]['significant_categoric_relationships'])-1):
                                self.target_key_feature_meta_vals[super]['significant_categoric_relationships']=self.target_key_feature_meta_vals[super]['significant_categoric_relationships'][:sub_index]+self.target_key_feature_meta_vals[super]['significant_categoric_relationships'][sub_index+1:]
                                self.target_key_feature_meta_vals[super]['significant_categoric_tests']        =        self.target_key_feature_meta_vals[super]['significant_categoric_tests'][:sub_index]+self.target_key_feature_meta_vals[super]['significant_categoric_tests'][sub_index+1:]
                            else:
                                self.target_key_feature_meta_vals[super]['significant_categoric_relationships']=self.target_key_feature_meta_vals[super]['significant_categoric_relationships'][:sub_index]
                                self.target_key_feature_meta_vals[super]['significant_categoric_tests']        =        self.target_key_feature_meta_vals[super]['significant_categoric_tests'][:sub_index]
                    
            
                # update targets by removing super/sub and capturing tests. both will be used to update super/sub values in dict
                if sub in self.has_called_fit_column_relationships:

                        try:
                            discard_col=self.target_key_feature_meta_vals[sub]['significant_categoric_relationships'].pop(super_index)
                            discard_test=self.target_key_feature_meta_vals[sub]['significant_categoric_tests'].pop(super_index)
                        except:
                            if super_index<(len(self.target_key_feature_meta_vals[sub]['significant_categoric_relationships'])-1):
                                self.target_key_feature_meta_vals[sub]['significant_categoric_relationships']=self.target_key_feature_meta_vals[sub]['significant_categoric_relationships'][:super_index]+self.target_key_feature_meta_vals[sub]['significant_categoric_relationships'][super_index+1:]
                                self.target_key_feature_meta_vals[sub]['significant_categoric_tests']        =        self.target_key_feature_meta_vals[sub]['significant_categoric_tests'][:super_index]+self.target_key_feature_meta_vals[sub]['significant_categoric_tests'][super_index+1:]
                            else:
                                self.target_key_feature_meta_vals[sub]['significant_categoric_relationships']=self.target_key_feature_meta_vals[sub]['significant_categoric_relationships'][:super_index]
                                self.target_key_feature_meta_vals[sub]['significant_categoric_tests']        =        self.target_key_feature_meta_vals[sub]['significant_categoric_tests'][:super_index]
                    
            # UPDATE TARGETS by adding super/sub and tests
            if (sub in self.has_called_fit_column_relationships) and (super not in self.target_key_feature_meta_vals[sub]['paired_to_a_supercategory']):
                self.target_key_feature_meta_vals[sub]['paired_to_a_supercategory_tests'].append(super_test)
                self.target_key_feature_meta_vals[sub]['paired_to_a_supercategory'].append(super)
            if (super in self.has_called_fit_column_relationships) and (sub not in self.target_key_feature_meta_vals[super]['paired_to_a_subcategory']):
                self.target_key_feature_meta_vals[super]['paired_to_a_subcategory_tests'].append(sub_test)
                self.target_key_feature_meta_vals[super]['paired_to_a_subcategory'].append(sub)
        return self
    

    """
    #######################################################################################
    # A FUNCTION THAT WRAPS FIT FUNCTIONS
    # FUNC 1) fit_column_relationships(self, 
                                 df: pd.DataFrame,
                                 numeric_columns=None,
                                 categoric_columns=None,
                                 numeric_target=None,
                                 categoric_target=None,
                                 )
    # FUNC 2) fit_multivariate_column_relationships(self,
                                              df:pd.DataFrame,
                                              targets:list|tuple|str|None,
                                              numeric_targets:bool=True,
                                              catigorci_targets:bool=True,
                                              max_n_combination_size:int|None=3,
                                              max_n_combinations:int|None=50_000,
                                              min_combo_size:int=2
                                              )
    # FUNC 3) fit_supercat_subcat_pairs(self,
                                           data:pd.DataFrame,
                                           max_evidence:float=0.2,  
                                           pairs_list:list|tuple|None=None)
    # #######################################################################################
    """

    def fit_full_dataset_analysis(self,
                              data:pd.DataFrame,
                                numeric_columns=None,
                                categoric_columns=None,
                                numeric_target=None,
                                categoric_target=None,
                                fit_good_of_fit:bool=True,
                                fit_normal:bool=True,
                                fit_multivariates:bool=False,
                                fit_supercat_subcats:bool=False,
                                check_assumptions:bool|None=None,
                                anova_assumption_check_params:dict|None=None,
                                kruskal_assumption_check_params:dict|None=None,
                                chi2_assumption_check_params:dict|None=None,
                                dropna_gof:bool|None=None
                                ): 
        """
        parameters for manual entry of columns, and for targets
            numeric_columns=None,
            categoric_columns=None,
        """
        default_anova = self.anova_assumption_check_params.copy()
        if anova_assumption_check_params is not None:
            default_anova.update(anova_assumption_check_params)
        anova_assumption_check_params = default_anova
        default_kruskal =self.kruskal_assumption_check_params.copy()
        if kruskal_assumption_check_params is not None:
            default_kruskal.update(kruskal_assumption_check_params)
        kruskal_assumption_check_params = default_kruskal
        default_chi = self.chi2_assumption_check_params.copy()
        if chi2_assumption_check_params is not None:
            default_chi.update(chi2_assumption_check_params)
        chi2_assumption_check_params = default_chi

        self.fit_column_relationships( df=data,
                                numeric_columns=numeric_columns,
                                categoric_columns=categoric_columns,
                                numeric_target=numeric_target,
                                categoric_target=categoric_target,
                                check_assumptions=check_assumptions,
                                anova_assumption_check_params=anova_assumption_check_params,
                                kruskal_assumption_check_params=kruskal_assumption_check_params,
                                chi2_assumption_check_params=chi2_assumption_check_params
                                )
        if fit_good_of_fit==True and (not ((numeric_target is not None) and (categoric_target is None))):
            if (categoric_target is None) and ( categoric_columns is None):
                cat_cols = None
            elif (categoric_target is None):
                cat_cols = categoric_columns
            else:
                cat_cols = categoric_target
            if isinstance(cat_cols,str):
                    cat_cols=[cat_cols]                    
            self.fit_goodness_of_fit_uniform(
                                        df=data,
                                        categoric_columns=cat_cols,
                                        dropna=dropna_gof,
                                        check_assumptions=check_assumptions)   
                 
        if fit_normal==True and (not ((categoric_target is not None) and (numeric_target is None))):
            if (numeric_target is None) and ( numeric_columns is None):
                num_cols = None
            elif (numeric_target is None):
                num_cols = numeric_columns
            else:
                num_cols = numeric_target
            if isinstance(num_cols,str):
                    num_cols=[num_cols]  
            self.fit_normal(df=data,
                            numeric_columns=num_cols)

        if fit_multivariates==True:
            # identify target(s) if present
            targets = self._combine_targets(numeric_target=numeric_target, 
                                            categoric_target=categoric_target)
            fit_multivariate_args=self.multivariate_params
            self.fit_multivariate_column_relationships(df=data,
                                                targets=targets,
                                                check_assumptions=check_assumptions,
                                                anova_assumption_check_params=anova_assumption_check_params,
                                                kruskal_assumption_check_params=kruskal_assumption_check_params,
                                                chi2_assumption_check_params=chi2_assumption_check_params,
                                                **fit_multivariate_args
                                                )
        if (fit_supercat_subcats==True) and (not ((numeric_target is not None) and (categoric_target is None))):
            fit_super_subcat_args=self.supercat_subcat_params
            self.fit_supercat_subcat_pairs(data=data,
                                            **fit_super_subcat_args) 
        




    ##############################################################################################################################
    ##############################################################################################################################
    # VISUALIZING THE OUPUT
    ##############################################################################################################################
    ##############################################################################################################################


    #######################################################################################
    # A FUNCTION TO PLOT UNIVARIATE WHERE REJECT NULL GOODNESS OF FIT FOR UNIFORM DISTRIBUTION
    # PLOTS VALUES STORED IN self.reject_null_good_of_fit
    #######################################################################################

    def plot_non_uniform_categorical(self,
                                        data:pd.DataFrame,
                                        categorical:list|tuple|None=None,
                                        proportions:bool=False,
                                        n_wide:int|tuple|list=(6,40,4),
                                        super_title:str|None="Univariate Categoric - Reject Good-Of-Fit for Uniform" ,
                   streamlit_:bool|None = None):
        """
        where if categorical is None, columns from self.reject_null_good_of_fit will be ploted. Otherwise columns in cateigorical will be ploted.
        n_wide indicates (columns wide, max sum of bars on row, row height in inches)
        """
        if streamlit_ is None:
            streamlit_ = False    
        if categorical is None:
            categorical = list(self.reject_null_good_of_fit)
        elif isinstance(categorical,str):
            categorical = [categorical]
        if (not categorical) and (not streamlit_):
            warnings.warn("There are not any non-uniform categorical variables stored in the model.\nEither none exist, or they haven't been fit.")

        self.univariate_categorical_snapshot(
                                        data=data,
                                        categorical=categorical,
                                        proportions=proportions,
                                        n_wide=n_wide,
                                        super_title=super_title,
                                            streamlit_=streamlit_)
        return self


    #######################################################################################
    # A FUNCTION TO PLOT UNIVARIATE WHERE REJECT NULL NORMAL DISTRIBUTION
    # PLOTS VALUES STORED IN self.reject_null_normal
    #######################################################################################

    def plot_non_normal_numeric(self,
                                        data:pd.DataFrame,
                                        numerical:list|tuple|None=None,
                                        kde:bool|None=None,
                                        proportions:bool=False,
                                        n_wide:int|tuple|list=(6,40,4),
                                        super_title:str|None="Univariate Numeric - Reject Normal Distribution",
                                        force_significant_bin_edges:bool|None=None,
                                        minimize_significant_bins:bool|None=None,
                                        include_multivariate:bool|None=None ,
                   streamlit_:bool|None = None):
        """
        where if categorical is None, columns from self.reject_null_good_of_fit will be ploted. Otherwise columns in cateigorical will be ploted.
        n_wide indicates (columns wide, max sum of bars on row, row height in inches)
        force_significant_bin_edges==True calculates min bins that retain statistical significance
        minimize_significant_bins is ignored if force_significant_bin_edges != True, else if True, it minimizes num bins
        """
        if streamlit_ is None:
            streamlit_ = False    
        if include_multivariate is None:
            include_multivariate = True
        if numerical is None:
            numerical = list(self.reject_null_normal)
        if (not numerical) and (not streamlit_):
            warnings.warn("There are not any non-normal numerical variables stored in the model.\nEither none exist, or they haven't been fit.")
        if isinstance(numerical,str):
            numerical = [numerical]
        keep_bins_significant = {}
        if force_significant_bin_edges==True:
            for col in numerical:
                min_bins, _ = self.get_a_varaibles_binning_metrics(data=data,
                                                                    target=col,
                                                                    check_multivar=include_multivariate,
                                                                    original_value_count_threashold=5)
                if (min_bins):
                    if minimize_significant_bins!=True:
                        try:
                            bin_edges = np.histogram_bin_edges(data[col], bins='auto')
                        except:
                            bin_edges = np.histogram_bin_edges(data[col], range = (data[col].min(),data[col].max()))
                        if (min_bins) and (len(bin_edges)-1)<min_bins:
                            try:
                                bin_edges = np.histogram_bin_edges(data[col],bins=min_bins)
                            except:
                                bin_edges = np.histogram_bin_edges(data[col],bins=min_bins, range = (data[col].min(),data[col].max()))
                    else:
                        try:
                            bin_edges = np.histogram_bin_edges(data[col],bins=min_bins)
                        except:
                            try:
                                bin_edges = np.histogram_bin_edges(data[col],bins=min_bins, range = (data[col].min(),data[col].max()))
                            except:
                                bin_edges = np.histogram_bin_edges(data[col], range = (data[col].min(),data[col].max()))
                    u_d = {col:pd.Series(bin_edges)}
                    keep_bins_significant.update(u_d)
        
        self.univariate_numerical_snapshot(data=data,
                                        numerical=numerical,
                                        n_wide=n_wide,
                                        kde=kde,
                                        super_title=super_title,
                                        proportions=proportions,
                                        keep_bins_significant=keep_bins_significant,
                                            streamlit_=streamlit_)

        return self
        
    #######################################################################################
    # A FUNCTION TO PLOT NUMERIC-TO-NUMERIC RELATIONSHIPS
    # PLOTS VALUES STORED IN self.above_threshold_corr_numnum
    #######################################################################################   

    def plot_bivariate_categoric_categoric(self,
                                           data:pd.DataFrame,  
                                           column_combinations:list|tuple|None=None,                      
                                        n_wide:int|tuple|list=(6,40,5),
                                        stacked_bars_when_max_bars_is_exceeded:bool=True,
                                        sorted:bool=False,
                                        super_title:str|None="Categoric-Categoric Bivariates - Reject Null" ,
                   streamlit_:bool|None = None):
        """
        where if column_combinations is None, combinations from self.reject_null_catcat will be ploted. Otherwise combinations in column_combinations will be ploted.
        n_wide indicates (columns wide, max sum of bars on row, row height in inches)
        """
        if streamlit_ is None:
            streamlit_ = False    
        if column_combinations is None:
            column_combinations = self.reject_null_catcat
        if (not column_combinations) and (not streamlit_):
            warnings.warn("The model does not contain any categoric-to-categoric column pairs with significant relationships.\nEither none exist, or they haven't been fit.")
        self.bivariate_categorical_snapshot(
                            data=data,
                            column_combinations=column_combinations,                        
                            n_wide=n_wide,
                            stacked_bars_when_max_bars_is_exceeded=stacked_bars_when_max_bars_is_exceeded,
                            sorted=sorted,
                            super_title=super_title,
                                            streamlit_=streamlit_)
        return self
    
    #######################################################################################
    # A FUNCTION TO PLOT NUMERIC-TO-NUMERIC RELATIONSHIPS
    # PLOTS VALUES STORED IN self.above_threshold_corr_numnum
    #######################################################################################

    def plot_bivariate_numeric_numeric(self,
                                       data:pd.DataFrame,
                                       column_combos:list|tuple|None=None,
                                       plot_type:str='joint',
                                       linreg:bool=True,
                                       super_title:str='Numeric-Numeric Bivariates - With Correlation',
                                       plot_type_kwargs:dict|None=None,
                                       linreg_kwargs:dict|None=None ,
                   streamlit_:bool|None = None):
        """
        where if column_combos is None, combinations from self.above_threshold_corr_numnum will be ploted. Otherwise combos in column_combos will be ploted.
        n_wide indicates (columns wide, max sum of bars on row, row height in inches)
        """
        if streamlit_ is None:
            streamlit_ = False    
        if column_combos is None:
            column_combos = self.above_threshold_corr_numnum
        if (not column_combos) and (not streamlit_):
            warnings.warn("The model does not contain any numeric-to-numeric column pairs with significant relationships.\nEither none exist, or they haven't been fit.")
        self.bivariate_numeric_numeric_snapshot(
                                           data=data,
                                            column_combos=column_combos,
                                            plot_type=plot_type,
                                            linreg=linreg,                        
                                            super_title=super_title,
                                            plot_type_kwargs=plot_type_kwargs,
                                            linreg_kwargs=linreg_kwargs,
                                            streamlit_=streamlit_)
        return self

    #######################################################################################
    # A FUNCTION TO PLOT NUMERIC-TO-CATEGORIC RELATIONSHIPS
    # PLOTS VALUES STORED IN self.reject_null_numcat  
    # where (num,cat) is the nested arrangement
    #######################################################################################

    def plot_numeric_to_categoric_relationships(self,
                                            data:pd.DataFrame,
                                            column_combos:list|tuple|None=None,
                                            plot_type:str='boxen', #box, boxen, or violin
                                            n_wide:int|tuple|list=(6,40,8),
                                            super_title:str|None='Numeric-Categoric Bivariates - Reject Null' ,
                   streamlit_:bool|None = None):
        """
        where if column_combos is None, combinations from self.reject_null_numcat will be ploted. Otherwise combos in column_combos will be ploted.
        n_wide indicates (columns wide, max sum of bars on row, row height in inches)
        """
        if streamlit_ is None:
            streamlit_ = False    
        if column_combos is None:
            column_combos = self.reject_null_numcat 
        if (not column_combos)  and (not streamlit_):
            warnings.warn("There are not any numeric-to-categoric column pairs with significant relationships.\nEither none exist, or they haven't been fit.")
        self.numeric_to_categorical_snapshot(data=data,
                                            column_combos=column_combos,
                                            plot_type=plot_type,
                                            n_wide=n_wide,
                                            super_title=super_title,
                                            streamlit_=streamlit_)
        return self
    
    #######################################################################################
    # A FUNCTION TO PLOT SUPERCATEGORIES TO SUBCATEGORIES
    # PLOTS VALUES STORED IN self.supercategory_subcategory_pairs
    #######################################################################################

    def plot_super_subcats(self,
                        data:pd.DataFrame,
                        supercat_subcat_pairs:list|tuple|None=None,
                        row_height:int=2,
                        cols_per_row:int=3,
                        y_tick_fontsize:int=12,
                        super_title:str|None=None ,
                   streamlit_:bool|None = None):
        """
        where if supercat_subcat_pairs is None, combinations from self.supercategory_subcategory_pairs will be ploted. Otherwise combos in supercat_subcat_pairs will be ploted.
        n_wide indicates (columns wide, max sum of bars on row, row height in inches)
        """
        if streamlit_ is None:
            streamlit_ = False    
        if row_height is None:
            row_height=2
        if cols_per_row is None:
            cols_per_row=3
        if y_tick_fontsize is None:
            y_tick_fontsize=12
        if super_title is None:
            super_title = "Supercategory-Subcategory - One Categoriec Variable Partitions Another"
        if supercat_subcat_pairs is None:
            supercat_subcat_pairs = self.supercategory_subcategory_pairs
        if (not supercat_subcat_pairs) and (not streamlit_):
            warnings.warn("There are not any Supercategory-Subcategory relationships to plot.\nEither none exist, or they haven't been fit.")
        figure_map, figure_plot_params = self._prep_super_subcat_figure_maps(data, 
                                                supercat_subcat_pairs = supercat_subcat_pairs, 
                                                row_height=row_height, 
                                                cols_per_row=cols_per_row, 
                                                y_tick_fontsize=y_tick_fontsize)
        self.plot_supercats_subcats(data, 
                                figure_map, 
                                super_title=super_title,
                                *figure_plot_params,
                                            streamlit_=streamlit_
                                )
        return self


    ############################################################################################
    # A WRAPPER FUNCTION THAT PLOTS ALL THE SUPPORTED PLOTS AVAILABLE
    # CALLS: self.plot_non_uniform_categorical(), self.plot_bivariate_categoric_categoric(), 
    #   self.plot_bivariate_numeric_numeric(), self.plot_numeric_to_categoric_relationships(), self.plot_super_subcats() 
    #   INTERNALLY
    ############################################################################################

    def produce_all_plots(self,
                  data:pd.DataFrame,
                  cat_univar:list|tuple|None|bool        =None,
                  num_univar:list|tuple|None|bool        =None,
                  catcat_bivar:list|tuple|None|bool      =None,
                  numnum_bivar:list|tuple|None|bool      =None,
                  numcat_bivar:list|tuple|None|bool      =None,  # where in practice num is placed before cat
                  super_subcat_pairs:list|tuple|None|bool=None,  
                  #multivar not yet supported,
                  cat_univar_params:dict|None        =None,
                  catcat_bivar_params:dict|None      =None,
                  numnum_bivar_params:dict|None      =None,
                  numcat_bivar_params:dict|None      =None,
                  super_subcat_pairs_params:dict|None=None,
                  num_univar_params:dict|None = None ,
                   streamlit_:bool|None = None): 
        """
        where data is the dataframe values are taken from
        cat_univar, catcat_bivar, numnum_bivar, numcat_bivar, and super_subcat_pairs
            can be of list|tuple|None|bool. 
                default is None
                if None, then values are taken from class objects stored in fit calls
                if False, then values are not ploted
        cat_univar_params, catcat_bivar_params, numnum_bivar_params, numcat_bivar_params, super_subcat_pairs_params
            accept custom plot parameters
        """
        if streamlit_ is None:
            streamlit_ = False
        if cat_univar!=False:
            if cat_univar==True:
                cat_univar=None
            default_cat_univar_params=self.cat_univar_params.copy() 
            if cat_univar_params is not None:
                default_cat_univar_params.update(cat_univar_params)
            cat_univar_params = default_cat_univar_params      
            self.plot_non_uniform_categorical(
                                            data=data,
                                            categorical=cat_univar,
                                            **cat_univar_params,
                   streamlit_ = streamlit_ )
        else:
            print('Plot Categoric Univariate is set to False')

        if num_univar!=False:
            if num_univar==True:
                num_univar=None
            default_num_univar_params=self.num_univar_params.copy()
            if num_univar_params is not None:
                default_num_univar_params.update(num_univar_params)
            num_univar_params=default_num_univar_params
            self.plot_non_normal_numeric(data=data,
                                        numerical=num_univar,
                                        **num_univar_params,
                   streamlit_ = streamlit_ )
        else:
            print('Plot Numeric Univariate is set to False')

        if catcat_bivar!=False:  
            if catcat_bivar==True:
                catcat_bivar=None
            default_catcat_bivar_params=self.catcat_bivar_params.copy()
            if catcat_bivar_params is not None:
                default_catcat_bivar_params.update(catcat_bivar_params)
            catcat_bivar_params=default_catcat_bivar_params
            self.plot_bivariate_categoric_categoric(
                                            data=data,  
                                            column_combinations=catcat_bivar,
                                            **catcat_bivar_params,
                   streamlit_ = streamlit_ ) 
        else:
            print('Plot Bivariate Categoric-Categoric is set to False')

        if numnum_bivar!=False:
            if numnum_bivar==True:
                numnum_bivar=None
            default_numnum_bivar_params=self.numnum_bivar_params.copy()
            if numnum_bivar_params is not None:
                default_numnum_bivar_params.update(numnum_bivar_params)
            numnum_bivar_params=default_numnum_bivar_params
            self.plot_bivariate_numeric_numeric(
                                            data=data,
                                            column_combos=numnum_bivar,
                                            **numnum_bivar_params,
                   streamlit_ = streamlit_ )
        else:
            print('Plot Bivariate Numeric-Numeric is set to False')

        if numcat_bivar!=False:
            if numcat_bivar==True:
                numcat_bivar=None
            default_numcat_bivar_params=self.numcat_bivar_params.copy()
            if numcat_bivar_params is not None:
                default_numcat_bivar_params.update(numcat_bivar_params)
            numcat_bivar_params=default_numcat_bivar_params
            self.plot_numeric_to_categoric_relationships(
                                            data=data,
                                            column_combos=numcat_bivar,
                                            **numcat_bivar_params,
                   streamlit_ = streamlit_ )
        else:
            print('Plot Bivariate Numeric-Categoric is set to False')

        if super_subcat_pairs!=False:
            if super_subcat_pairs==True:
                super_subcat_pairs=None
            default_super_subcat_pairs_params=self.super_subcat_pairs_params.copy()
            if super_subcat_pairs_params is not None:
                default_super_subcat_pairs_params.update(super_subcat_pairs_params)
            super_subcat_pairs_params=default_super_subcat_pairs_params
            self.plot_super_subcats(
                                    data=data,
                                    supercat_subcat_pairs=super_subcat_pairs,
                                    **super_subcat_pairs_params,
                   streamlit_ = streamlit_ ) 
        else:
            print('Plot Supercategory-Subcategory Partitions is set to False')

        return self
    
    #######################################################################################
    # FUNCTIONS AND HELPER FUNCITONS TO PLOT COLUMN(S) AND RELATIONSHIPS INVOLVING A TARGET OR TARGETS
    # PLOTS VALUES STORED IN self.target_key_feature_meta_vals
    #######################################################################################


    def _prepare_target_plot_data(self,
                                 target_column:str,
                                 reject_numcat:bool=True,
                                 reject_numnum:bool=True,
                                 reject_catcat:bool=True,
                                 is_super_or_subcat:bool=True,
                                 not_uniform_or_reject_normal:bool=True,
                                 reject_multivariates:bool=False):    #FALSE BECAUSE MULITVARIATE VISULIZATIONS ARE NOT YET SUPPORTED
        """
        where target_column is a string and other parameters are T/F to indicate whether or not they should be included
        """
        
        meta_dict_ = self.target_key_feature_meta_vals.get(target_column,None)
        # create a copy to allow use modifications to meta_dict w/o affecting class the object
        if meta_dict_!=None:
            meta_dict=meta_dict_.copy()
        if meta_dict is None:
            if isinstance(target_column,str):
                raise ValueError(f"{target_column} has not been fit.")
            else:
                raise ValueError(f"Expected data type == string, found {type(target_column)}")
            
        numcat, numnum, catcat, supersubcat, univariate, multivariate = [],[],[],[],[],[]

        data_type = self.target_key_feature_meta_vals[target_column]['target_dtype'][0]

        # match target with numeric cols
        if meta_dict['significant_numeric_relationships']:            
            nums=  zip(meta_dict['significant_numeric_relationships'], meta_dict['significant_numeric_tests'])
            if data_type=='numeric' and reject_numnum==True:
                for col, test in nums:
                    result=[target_column,col,test]
                    numnum.append(result)
            elif data_type=='categoric' and reject_numcat==True:
                for col, test in nums:
                    result=[col,target_column,test]
                    numcat.append(result)  # where numcat is not the order stored anywhere else either. stored are [num, cat]
        # match target with categoric cols
        if meta_dict['significant_categoric_relationships']:
            cats=  zip(meta_dict['significant_categoric_relationships'], meta_dict['significant_categoric_tests'])
            if data_type=='numeric' and reject_numcat==True:
                for col, test in cats:
                    result=[target_column,col,test]
                    numcat.append(result)
            elif data_type=='categoric' and reject_catcat==True:
                for col, test in cats:
                    result=[col,target_column,test]
                    catcat.append(result) 
        # match with multivarieate combos
        if meta_dict['significant_categoric_combination_group_relationship'] and reject_multivariates==True:
            combs= zip(meta_dict['significant_categoric_combination_group_relationship'], meta_dict['significant_categoric_combination_group_relationship_test_type'])
            if data_type=='numeric':
                for col, test in combs:
                    result=[[target_column,'numeric'],col,test]
                    multivariate.append(result)
            elif data_type=='categoric':
                for col, test in combs:
                    result=[[target_column,'categoric'],col,test]
                    multivariate.append(result) 
        # match to super or subcat if exists
        if meta_dict['paired_to_a_supercategory'] and is_super_or_subcat==True:
            for match in meta_dict['paired_to_a_supercategory']:
                res = [match, target_column]
                supersubcat.append(res)
        if meta_dict['paired_to_a_subcategory'] and is_super_or_subcat==True:
            for match in meta_dict['paired_to_a_subcategory']:
                res = [target_column, match]
                supersubcat.append(res)
        # if it's a univariate to plot
        plottable_responses = ('reject_uniform','reject_normal')  # potential to have normal distribution support too, but not supported at this time
        if (not_uniform_or_reject_normal==True) and any( result in plottable_responses for result in meta_dict['is_normal_or_uniform'] ):
               univariate.append(target_column)
        return  numcat, numnum, catcat, supersubcat, univariate, multivariate
    

    def _fit_target_visualizations(self,
                    data:pd.DataFrame,
                    targets:list|tuple|str,
                    reject_numcat:bool=True,
                    reject_numnum:bool=True,
                    reject_catcat:bool=True,
                    is_super_or_subcat:bool=True,
                    not_uniform_or_reject_normal:bool=True,
                    reject_multivariates:bool=False,
                    auto_fit:bool=True,   # to call fit function(s) when needed
                    check_assumptions:bool|None=None,
                    dropna_gof:bool|None=None,
                    anova_assumption_check_params:dict|None=None,
                    kruskal_assumption_check_params:dict|None=None,
                    chi2_assumption_check_params:dict|None=None,                                                
                    ): 
        
        """
        where targets is list|tuple of targets to plot. string is accepted too in cases of one target
        auto_fit==True indicates the target should be fit if not already, otherwise a RuntimeError will raise
        all other parameters are bool T/F to indicate whether they should be included
        """
        # ensure listlike targets
        if isinstance(targets,str):
            targets=[targets]
        # create a template dict
        def target_dict_template():
            return {'reject_numcat':[],
                    'reject_numnum':[],
                    'reject_catcat':[],
                    'is_super_or_subcat':[],
                    'not_uniform_or_reject_normal':[],
                    'reject_multivariates':[]
                    }
        # this will be updated based on target_dict_template() for each target
        targets_and_results = {}
        # loop through targets
        for target in targets:
            # determine target datatype
            target_dtype=data[target].dtype

            # check if the target has been fit at all (such as hyp test w other columns), and if auto_fit==True, call fit if needed
            if target not in self.has_called_fit_column_relationships:
                if auto_fit==True:
                    if target_dtype not in ('object','category'):
                        numeric_target=target
                        categoric_target=None
                    else:
                        numeric_target=None
                        categoric_target=target
                    self.fit_column_relationships(df=data,
                                 numeric_target=numeric_target,
                                 categoric_target=categoric_target,
                                check_assumptions=check_assumptions,                                
                                anova_assumption_check_params=anova_assumption_check_params,
                                kruskal_assumption_check_params=kruskal_assumption_check_params,
                                chi2_assumption_check_params=chi2_assumption_check_params
                                 )
                else:
                    raise RuntimeError(f'{target} has not been fit. Set auto_fit==True, or fit_column_relationships() needs to be called.')                
            # check if the target has been fit with multivariates, and if auto_fit==True, call fit if needed
            if (reject_multivariates==True) and (self.target_key_feature_meta_vals[target]['max_n_variates_paired_with'][0]==0):
                if auto_fit==True:
                    multivariate_params= self.multivariate_params
                    self.fit_multivariate_column_relationships(df=data,
                                              targets=target,
                                                check_assumptions=check_assumptions,
                                                anova_assumption_check_params=anova_assumption_check_params,
                                                kruskal_assumption_check_params=kruskal_assumption_check_params,
                                                chi2_assumption_check_params=chi2_assumption_check_params,
                                                **multivariate_params,
                                              )
                else:
                    raise RuntimeError(f'{target} has not been fit. Set auto_fit==True, or fit_multivariate_column_relationships() needs to be called.')
            # check if the target has been tested/fit for to a univariate distribution, and if auto_fit==True, call fit if needed            
            if (not_uniform_or_reject_normal==True):
                if  (target_dtype in ('object','category'))  and (target not in self.has_called_fit_goodness_of_fit_uniform):
                    if (auto_fit==True):
                        self.fit_goodness_of_fit_uniform(data,
                                        categoric_columns=target,
                                        dropna=dropna_gof,
                                        check_assumptions=check_assumptions)
                    else:
                        raise RuntimeError(f'{target} has not been fit. Set auto_fit==True, or fit_goodness_of_fit_uniform() needs to be called.')
                elif  (target_dtype in (float,np.float64,'float64','float32','float16',np.number,'numeric','number'))  and (target not in self.has_called_fit_normal):
                    if (auto_fit==True):
                        self.fit_normal(data,
                                    categoric_columns=target)
                    else:
                        raise RuntimeError(f'{target} has not been fit. Set auto_fit==True, or fit_normal() needs to be called.')
            # check if the target has been tested/fit for super_subcat relationships, and if auto_fit==True, call fit if needed           
            if (is_super_or_subcat==True) and (self.target_key_feature_meta_vals[target]['target_dtype']==['categoric']):
                if auto_fit==True:
                    # unfiltered pairs list. Pairs that have already been fit will be filtered inside fit_supercat_subcat_pairs(). Not doing so here avoids redundancy.
                    pairs_list=[[i,target] for i in self.target_key_feature_meta_vals[target]['significant_categoric_relationships']]
                    if pairs_list:
                        supsub_params=self.supercat_subcat_params 
                        self.fit_supercat_subcat_pairs(data,
                                                    **supsub_params,
                                                        pairs_list=pairs_list )                
                else:
                    raise RuntimeError(f'{target} has not been fit. Set auto_fit==True, or fit_supercat_subcat_pairs() needs to be called.')
            one_targ_numcat, one_targ_numnum, one_targ_catcat, one_targ_supersubcat, one_targ_univariate, one_targ_multivariate = self._prepare_target_plot_data(
                                                                            target_column=target,
                                                                            reject_numcat=reject_numcat,
                                                                            reject_numnum=reject_numnum,
                                                                            reject_catcat=reject_catcat,
                                                                            is_super_or_subcat=is_super_or_subcat,
                                                                            not_uniform_or_reject_normal=not_uniform_or_reject_normal,
                                                                            reject_multivariates=reject_multivariates)
            
            targets_and_results[target]=target_dict_template()
            targets_and_results[target]['reject_numcat']=targets_and_results[target]['reject_numcat']+one_targ_numcat
            targets_and_results[target]['reject_numnum']=targets_and_results[target]['reject_numnum']+one_targ_numnum
            targets_and_results[target]['reject_catcat']=targets_and_results[target]['reject_catcat']+one_targ_catcat
            targets_and_results[target]['is_super_or_subcat']=targets_and_results[target]['is_super_or_subcat']+one_targ_supersubcat
            targets_and_results[target]['not_uniform_or_reject_normal']=targets_and_results[target]['not_uniform_or_reject_normal']+one_targ_univariate
            targets_and_results[target]['reject_multivariates']=targets_and_results[target]['reject_multivariates']+one_targ_multivariate
            
        return targets_and_results


    def visualize_by_targets(self,
                    data:pd.DataFrame,
                    targets:list|tuple|str,
                    reject_numcat:bool=True,
                    reject_numnum:bool=True,
                    reject_catcat:bool=True,
                    is_super_or_subcat:bool=True,
                    not_uniform_or_reject_normal:bool=True,
                    reject_multivariates:bool=False,
                    auto_fit:bool=True,   # to call fit function(s) when needed
                    targets_share_plots:bool=False,
                    check_assumptions:bool|None=None,
                    dropna_gof:bool|None=None,
                    anova_assumption_check_params:dict|None=None,
                    kruskal_assumption_check_params:dict|None=None,
                    chi2_assumption_check_params:dict|None=None,
                    cat_univar_params:dict|None        =None,
                    catcat_bivar_params:dict|None      =None,
                    numnum_bivar_params:dict|None      =None,
                    numcat_bivar_params:dict|None      =None,
                    super_subcat_pairs_params:dict|None=None,
                    num_univar_params:dict|None        =None ,
                    streamlit_: bool|None=None
                    ):
        """
        where data is the dataframe that holds values
        targets is a string target variable or list|tuple of target(s)
        auto_fit indicates whethere to fit variables that haven't been fit, or to raise a RuntimeError
        targets_share_plots indicates whether to put all targets on the same figures, or to create seperate sets of figures for each target
        other parameters are True/False bool to indicate whether to include in plots. 
            they are:
                reject_numcat: significant categorical-numerical relationship pairs
                reject_numnum: significant numerical-numerical relationship pairs
                reject_catcat: significant categorical-categorical relationship pairs
                is_super_or_subcat: supercategory-subcategory pairs
                not_uniform_or_reject_normal: categorical variables that don't follow a uniform distribution {rejected normal distribution not yet supported for numeric},
                reject_multivariates: where target columns are tested/compared to concatenated variables
        """
        if streamlit_ is None:
            streamlit_ = False
        if dropna_gof is None:
            dropna_gof=self.dropna_cats
        if check_assumptions is None:
            check_assumptions = self.check_assumptions


        default_anova = self.anova_assumption_check_params.copy()
        if anova_assumption_check_params is not None:
            default_anova.update(anova_assumption_check_params)
        anova_assumption_check_params = default_anova
        default_kruskal =self.kruskal_assumption_check_params.copy()
        if kruskal_assumption_check_params is not None:
            default_kruskal.update(kruskal_assumption_check_params)
        kruskal_assumption_check_params = default_kruskal
        default_chi = self.chi2_assumption_check_params.copy()
        if chi2_assumption_check_params is not None:
            default_chi.update(chi2_assumption_check_params)
        chi2_assumption_check_params = default_chi

        ## should be made into a class object and updated here such as [targ for targ in targets if targ not in self.target_dict.keys()) self.targets_dict.update(result)
        targets_dict =  self._fit_target_visualizations(
                    data=data,
                    targets=targets,
                    reject_numcat=reject_numcat,
                    reject_numnum=reject_numnum,
                    reject_catcat=reject_catcat,
                    is_super_or_subcat=is_super_or_subcat,
                    not_uniform_or_reject_normal=not_uniform_or_reject_normal,
                    reject_multivariates=reject_multivariates,
                    auto_fit=auto_fit,   # to call fit function(s) when needed
                    check_assumptions=check_assumptions,
                    dropna_gof=dropna_gof,
                    anova_assumption_check_params=anova_assumption_check_params,
                    kruskal_assumption_check_params=kruskal_assumption_check_params,
                    chi2_assumption_check_params=chi2_assumption_check_params,
                    )
        # plot targets on one plot per plot type


        if targets_share_plots==True:
            reject_numcat, reject_numnum, reject_catcat, is_super_or_subcat, not_uniform_or_reject_normal, reject_multivariates=[],[],[],[],[],[]
            #iterate through dict and make sure no combos are repeated
            for k,v in targets_dict.items():

                #no repeat reject numcat
                if reject_numcat:
                    vrcn=[]
                    for val in v['reject_numcat']:
                        # determine if val is already in reject_numcat or not
                        is_new=True
                        for first_val in reject_numcat:
                            if ((val[0]==first_val[0]) and (val[1]==first_val[1])) or ((val[0]==first_val[1]) and (val[1]==first_val[0])):
                                is_new=False
                                break
                        if is_new==True:
                            vrcn.append(val)
                    reject_numcat=reject_numcat+vrcn
                else:
                    reject_numcat = v['reject_numcat']

                # no repeat numnum
                if reject_numnum:
                    vrnn=[]
                    for val in v['reject_numnum']:
                        # determine if val is already in reject_numnum or not
                        is_new=True
                        for first_val in reject_numnum:
                            if ((val[0]==first_val[0]) and (val[1]==first_val[1])) or ((val[0]==first_val[1]) and (val[1]==first_val[0])):
                                is_new=False
                                break
                        if is_new==True:
                            vrnn.append(val)
                    reject_numnum=reject_numnum+vrnn
                else:
                    reject_numnum=v['reject_numnum']

                # no repeat catcat
                if reject_catcat:
                    vrcc=[]
                    for val in v['reject_catcat']:
                        is_new=True
                        for first_val in reject_catcat:
                            if ((val[0]==first_val[0]) and (val[1]==first_val[1])) or ((val[0]==first_val[1]) and (val[1]==first_val[0])):
                                is_new=False
                                break
                        if is_new==True:
                            vrcc.append(val)
                    reject_catcat=reject_catcat+vrcc
                else:
                    reject_catcat = v['reject_catcat']

                #no repeat super subcat combos
                if is_super_or_subcat:
                    visos=[]
                    for val in v['is_super_or_subcat']:
                        is_new=True
                        for first_val in is_super_or_subcat:
                            if ((val[0]==first_val[0]) and (val[1]==first_val[1])) or ((val[0]==first_val[1]) and (val[1]==first_val[0])):
                                is_new=False
                                break
                        if is_new==True:
                            visos.append(val)
                    is_super_or_subcat=is_super_or_subcat+visos 
                else:
                    is_super_or_subcat=v['is_super_or_subcat']

                # no need to filter duplicates for good of fit uniform
                not_uniform_or_reject_normal=not_uniform_or_reject_normal+v['not_uniform_or_reject_normal']
                
                # there may be duplicates in multivariate, but because of the 'target' approach they are not filtered out of plots ==>> (ie target on one axis, other vars on the other)
                reject_multivariates=reject_multivariates+v['reject_multivariates']
            
            # don't pass empty lists or tuples
            if not reject_numcat: reject_numcat=False
            if not reject_numnum: reject_numnum=False
            if not reject_catcat: reject_catcat=False
            if not is_super_or_subcat: is_super_or_subcat=False
            if not not_uniform_or_reject_normal: #not_uniform_or_reject_normal=False
                cat_univar, num_univar = False, False
            else:
                cat_univar=[col for col in not_uniform_or_reject_normal if col in self.reject_null_good_of_fit]
                if not cat_univar: cat_univar=False
                num_univar=[col for col in not_uniform_or_reject_normal if col in self.reject_null_normal]
                if not num_univar: num_univar=False
            if not reject_multivariates: reject_multivariates=False
            # plot vars
            print(f"SIGNIFICANT VISUALIZATIONS FOR VARAIBLES IN:\n          {targets}")
            print('Non-Value Plots will Automatically be set to False')
            self.produce_all_plots(
                                data=data,
                                cat_univar=cat_univar,
                                num_univar=num_univar,
                                catcat_bivar=reject_catcat,
                                numnum_bivar=reject_numnum,
                                numcat_bivar=reject_numcat,
                                super_subcat_pairs=is_super_or_subcat,
                                cat_univar_params = cat_univar_params,
                                catcat_bivar_params = catcat_bivar_params,
                                numnum_bivar_params = numnum_bivar_params,
                                numcat_bivar_params = numcat_bivar_params,
                                super_subcat_pairs_params = super_subcat_pairs_params,
                                num_univar_params = num_univar_params,
                                streamlit_ = streamlit_)
        # plot targets individually
        else:
            for k,v in targets_dict.items():
                reject_numcat=v['reject_numcat']
                reject_numnum=v['reject_numnum']
                reject_catcat=v['reject_catcat']
                is_super_or_subcat=v['is_super_or_subcat']
                not_uniform_or_reject_normal=v['not_uniform_or_reject_normal']
                reject_multivariates=v['reject_multivariates']
                # don't pass empty lists or tuples
                if not reject_numcat: reject_numcat=False
                if not reject_numnum: reject_numnum=False
                if not reject_catcat: reject_catcat=False
                if not is_super_or_subcat: is_super_or_subcat=False
                if not not_uniform_or_reject_normal: #not_uniform_or_reject_normal=False
                    cat_univar, num_univar = False, False
                else:
                    cat_univar=[col for col in not_uniform_or_reject_normal if col in self.reject_null_good_of_fit]
                    if not cat_univar: cat_univar=False
                    num_univar=[col for col in not_uniform_or_reject_normal if col in self.reject_null_normal]
                    if not num_univar: num_univar=False
                if not reject_multivariates: reject_multivariates=False
                # call the plot function
                print('= = = = = '*20)
                print(f"SIGNIFICANT VISUALIZATIONS FOR {k}")
                print('Non-Value Plots will Automatically be set to False')
                self.produce_all_plots(
                                    data=data,
                                    cat_univar=cat_univar,
                                    num_univar=num_univar,
                                    catcat_bivar=reject_catcat,
                                    numnum_bivar=reject_numnum,
                                    numcat_bivar=reject_numcat,
                                    super_subcat_pairs=is_super_or_subcat,
                                    cat_univar_params = cat_univar_params,
                                    catcat_bivar_params = catcat_bivar_params,
                                    numnum_bivar_params = numnum_bivar_params,
                                    numcat_bivar_params = numcat_bivar_params,
                                    super_subcat_pairs_params = super_subcat_pairs_params,
                                    num_univar_params = num_univar_params,
                                    streamlit_ = streamlit_)
                print('= = = = = '*20)
        return self





    #######################################################################################
    # A FUNCTION TO find min bins for a single target column
    # returns values based on columns stored in self.target_key_feature_meta_vals[target]
    # does not update the model
    #######################################################################################

    def get_a_varaibles_binning_metrics(self,
                                        data:pd.DataFrame,
                                        target:str,
                                        check_multivar:bool|None=None,
                                        original_value_count_threashold:int|None=None):

        """
        target is a string that is a column in the dataframe
        check_multivar is True by default and indicates that multivariat features should be considered
        original_value_count_threashold is the min number of unique values a variable can have to be considered. default is 5

        returns (targ_abs_min_bin,check_multivar) , targ_to_feature_bin_map 
            where targ_abs_min_bin is the abs min bin for the target
            check_multivar is the input parameter to record whether multivariates were considered
            where targ_to_feature_bin_map is a per feature dict of min bins
        """
        if check_multivar is None:
            check_multivar=True
        if original_value_count_threashold is None:
            original_value_count_threashold=5

        data = data.copy()

        # stage data based on test type because many varaibles can have multiple test types
        test_dict = {'pearson':set(),'spearman':set(),'kendall':set(),'anova':set(),'kruskal':set(),'welch':set(),'student':set()}

        # gather target specific data
        # significant numeric
        for index, test_meta in enumerate(self.target_key_feature_meta_vals[target]['significant_numeric_tests']):
            tests = test_meta.split('-')
            for test in tests:
                curr_tst = test.split(':')[0]
                curr_update = {self.target_key_feature_meta_vals[target]['significant_numeric_relationships'][index]}
                test_dict[curr_tst].update(curr_update)
        
        # significant categoric
        for index, test_meta in enumerate(self.target_key_feature_meta_vals[target]['significant_categoric_tests']):
            tests = test_meta.split('-')
            for test in tests:
                curr_tst = test.split(':')[0]
                curr_update = {self.target_key_feature_meta_vals[target]['significant_categoric_relationships'][index]}
                test_dict[curr_tst].update(curr_update)
        
        # check supercat subcat memberships in existing result
        if self.target_key_feature_meta_vals[target]['paired_to_a_supercategory']:
            # if any are in, assume all are in based on class param:  supercat_subcat_params = {'isolate_super_subs':False } True
            if self.target_key_feature_meta_vals[target]['paired_to_a_supercategory'][0] not in self.target_key_feature_meta_vals[target]['significant_categoric_relationships']:
                
                for index, test_meta in enumerate(self.target_key_feature_meta_vals[target]['paired_to_a_supercategory_tests']):
                    tests = test_meta.split('-')
                    for test in tests:
                        curr_tst = test.split(':')[0]
                        curr_update = {self.target_key_feature_meta_vals[target]['paired_to_a_supercategory'][index]}
                        test_dict[curr_tst].update(curr_update)

        # same as w supercat
        if self.target_key_feature_meta_vals[target]['paired_to_a_subcategory']:
            if self.target_key_feature_meta_vals[target]['paired_to_a_subcategory'][0] not in self.target_key_feature_meta_vals[target]['significant_categoric_relationships']:
                
                for index, test_meta in enumerate(self.target_key_feature_meta_vals[target]['paired_to_a_subcategory_tests']):
                    tests = test_meta.split('-')
                    for test in tests:
                        curr_tst = test.split(':')[0]
                        curr_update = {self.target_key_feature_meta_vals[target]['paired_to_a_subcategory'][index]}
                        test_dict[curr_tst].update(curr_update)
                        
        # add temp cols to df that will be dropped later
        concated_temp_cols = []
        if (check_multivar==True) and self.target_key_feature_meta_vals[target]['significant_categoric_combination_group_relationship']:
            # used to ensure temp columns have unique names
            original_columns = set(data.columns)
            for combo in self.target_key_feature_meta_vals[target]['significant_categoric_combination_group_relationship']:
                concated_temp = self.multivariate_concatenation_delimiter.join(combo)
                while concated_temp in original_columns:
                    concated_temp = concated_temp + '_i'
                concated_temp_cols.append(concated_temp)
                data[concated_temp] = data[combo[0]]
                if (len(combo)>1) and (not isinstance(combo,str)):
                    for col in combo[1:]:
                        data[concated_temp] = data[concated_temp].astype(str) + self.multivariate_concatenation_delimiter + data[col].astype(str)
                
            for index, test_meta in enumerate(self.target_key_feature_meta_vals[target]['significant_categoric_combination_group_relationship_test_type']):
                tests = test_meta.split('-')
                for test in tests:
                    curr_tst = test.split(':')[0]
                    curr_update = {concated_temp_cols[index]}
                    test_dict[curr_tst].update(curr_update)

        # iterate through feature columns to find 'true' min bins
        targ_abs_min_bin = None
        targ_to_feature_bin_map = {}
        
        for key, value in test_dict.items():
            if key in ('pearson','spearman','kendall'):
                threshold = None
                for meta in self.numnum_meth_alpha_above:
                    if meta[0] == key:
                        threshold = meta[1]
                        break
                if threshold is None:
                    continue
                    #raise ValueError('Test type does not match model parameters')
                numnum_meth_alpha_above=(key,threshold,True)
                numcat_meth_alpha_above=None
                num_num_vars=list(value)
                num_cat_vars=None   
            elif key in ('welch','student'):
                threshold = None
                for meta in self.numnum_meth_alpha_above:
                    if meta[0] == key:
                        threshold = meta[1]
                        break
                if threshold is None:
                    continue
                    #raise ValueError('Test type does not match model parameters')        
                self.numnum_meth_alpha_above
                numnum_meth_alpha_above=(key,threshold,False)
                numcat_meth_alpha_above=None
                num_num_vars=list(value)
                num_cat_vars=None        
            elif key in('anova','kruskal'):
                threshold = None
                for meta in self.numcat_meth_alpha_above:
                    if meta[0] == key:
                        threshold = meta[1]
                        break
                if threshold is None:
                    continue
                    #raise ValueError('Test type does not match model parameters')        
                numnum_meth_alpha_above=None
                numcat_meth_alpha_above=(key,threshold,False)
                num_num_vars=None
                num_cat_vars=list(value)
            else:
                raise ValueError('Encountered unexpected test type.')
            curr_min, curr_targ_features_mins = self.determine_min_number_of_bins(dataframe=data, 
                                                num_num_pairs=num_num_vars, 
                                                cat_num_pairs=num_cat_vars, 
                                                original_value_count_threashold=original_value_count_threashold, 
                                                numnum_meth_alpha_above=numnum_meth_alpha_above,
                                                numcat_meth_alpha_above=numcat_meth_alpha_above,
                                                categoric_target=None, 
                                                numeric_target=[target],
                                                non_pair_numnum_numcat=True )
            if curr_min:
                if (targ_abs_min_bin is None) or ((curr_min[target] is not None) and (curr_min[target]>targ_abs_min_bin)):
                    targ_abs_min_bin=curr_min[target]
            if curr_targ_features_mins:
                for k,v in curr_targ_features_mins[target].items():
                    if k not in targ_to_feature_bin_map.keys():
                        targ_to_feature_bin_map[k]=v['min_within_threshold']
                    else:
                        if (not (not v['min_within_threshold'])) and (not (not targ_to_feature_bin_map[k])) and (v['min_within_threshold']<targ_to_feature_bin_map[k]):
                            up_d = {k: v['min_within_threshold']}
                            targ_to_feature_bin_map.update(up_d)
            

        # drop temp columns
        data = data.drop(columns=concated_temp_cols)

        return targ_abs_min_bin , targ_to_feature_bin_map
    

    def fit_binning_thresholds(self,
                               data:pd.DataFrame,
                               targets:list|tuple|str|None=None,
                            check_multivar:bool|None=None,
                            original_value_count_threashold:int|None=None):
        """
        """
        if targets is None:
            targets = []
            for col,v in self.items():
                if v['target_dtype']==['numeric']:
                    targets.append(col)
        elif isinstance(targets,str):
            targets = [targets]
        if check_multivar is None:
            check_multivar=True
        if original_value_count_threshold is None:
            original_value_count_threshold = 5
        for target in targets:
            targ_abs_min_bin , targ_to_feature_bin_map = self.get_a_varaibles_binning_metrics(self,
                                                                                                data=data,
                                                                                                target=target,
                                                                                                check_multivar=check_multivar,
                                                                                                original_value_count_threashold=original_value_count_threashold)
            if self.target_key_feature_meta_vals[target]['min_bins']:
                targ_abs_min_bin = max( targ_abs_min_bin, self.target_key_feature_meta_vals[target]['min_bins'][0] )
            self.target_key_feature_meta_vals[target]['min_bins']=[targ_abs_min_bin]
            self.target_key_feature_meta_vals[target]['min_bins_by_feature'].update(targ_to_feature_bin_map)
           
                
                
                
    
    #################################################################################################
    #  column_relationships_df(),  TO RETURN A DATAFRAME FROM self.target_key_feature_meta_vals
    # the dataframe only includes data on significan comparisons
    ##################################################################################################   


    def column_relationships_df(self,
                                targets:str|list|tuple|None=None):
        """
        includes data for all targets in input parameter targets if targets is not None, else all columns that have been fit are treated as columns

        processes self.target_key_feature_meta_vals
        and returns a dataframe that includes:
                                                'Target',
                                                'Type',
                                                'Distribution',
                                                'MaxLenCombosComparedTo'
                                                'FeatureColumn(s)'
                                                'Test(s)'

            columns:
                'FeatureColumn(s)': single strings that are column headers, or lists of headers that make up combo(s)
                'Test(s)': indicates the tests used to determine relationship status            
        """
        

        print("Where 'MaxLenComboComparedTo' can vary depending on compute limits and number of possible combinations in combo sizes.\n'FeatureColumn(s)' is a combination or single column that shares a significant relationship according to the test(s).\n'Test(s)' lists the test(s) used. If the testname is followed by a colon, that signals whether assumptions were met for that specifit test.")

        if targets is None:
            targets = self.has_called_fit_column_relationships
        if isinstance(targets,str):
            targets=[targets]

        grand_feature_variables         = []
        grand_test_types                = []
        grand_target_col                = []
        grand_max_multivar_combo_size   = []
        grand_data_type                 = []
        grand_is_normal_or_uniform      = []
            
        for target in targets:
            
            feature_variables         = []
            test_types                = []

            # first  set of many list concatinations
            feature_variables         += self.target_key_feature_meta_vals[target]['significant_numeric_relationships']
            test_types                += self.target_key_feature_meta_vals[target]['significant_numeric_tests']
            feature_variables         += self.target_key_feature_meta_vals[target]['significant_categoric_relationships']
            test_types                += self.target_key_feature_meta_vals[target]['significant_categoric_tests']
            
            # check supercat subcat memberships in existing result
            if self.target_key_feature_meta_vals[target]['paired_to_a_supercategory']:
                # if any are in, assume all are in based on class param:  supercat_subcat_params = {'isolate_super_subs':False } True
                if self.target_key_feature_meta_vals[target]['paired_to_a_supercategory'][0] not in self.target_key_feature_meta_vals[target]['significant_categoric_relationships']:
                    
                    feature_variables += self.target_key_feature_meta_vals[target]['paired_to_a_supercategory']
                    test_types        += self.target_key_feature_meta_vals[target]['paired_to_a_supercategory_tests']
            
            # same as w supercat
            if self.target_key_feature_meta_vals[target]['paired_to_a_subcategory']:
                if self.target_key_feature_meta_vals[target]['paired_to_a_subcategory'][0] not in self.target_key_feature_meta_vals[target]['significant_categoric_relationships']:
                    
                    feature_variables += self.target_key_feature_meta_vals[target]['paired_to_a_subcategory']
                    
                    test_types        += self.target_key_feature_meta_vals[target]['paired_to_a_subcategory_tests']
            # add temp cols to df that will be dropped later
            feature_variables         += self.target_key_feature_meta_vals[target]['significant_categoric_combination_group_relationship']
            test_types                += self.target_key_feature_meta_vals[target]['significant_categoric_combination_group_relationship_test_type']

            # use the len of the concated values to future multi-index
            height                     = len(feature_variables)
            # concate target results to the grand dataframe columns
            if height>0:
                grand_target_col              += [target]*height
                grand_max_multivar_combo_size += self.target_key_feature_meta_vals[target]['max_n_variates_paired_with'].copy()*height
                grand_data_type               += self.target_key_feature_meta_vals[target]['target_dtype'].copy()*height
                if not self.target_key_feature_meta_vals[target]['is_normal_or_uniform']: 
                    grand_is_normal_or_uniform    +=[pd.NA]*height
                else:
                    grand_is_normal_or_uniform    += self.target_key_feature_meta_vals[target]['is_normal_or_uniform'].copy()*height

                grand_feature_variables       += feature_variables
                grand_test_types              += test_types

        result_dataframe = pd.DataFrame({'Target':grand_target_col,
                                        'Type':grand_data_type,
                                        'Distribution':grand_is_normal_or_uniform,
                                        'MaxLenCombosComparedTo':grand_max_multivar_combo_size,
                                        'FeatureColumn(s)':grand_feature_variables,
                                        'Test(s)':grand_test_types})
        result_dataframe = result_dataframe.set_index(['Target',
                                                        'Type',
                                                        'Distribution',
                                                        'MaxLenCombosComparedTo'])
        return result_dataframe
        




