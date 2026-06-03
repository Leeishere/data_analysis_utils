import pandas as pd
import numpy as np
import scipy.stats
import warnings
import itertools
from itertools import combinations

from .CompareColumns import CompareColumns

class Bin(CompareColumns):
    def __init__(self):
        self.numeric_target_column_minimums=None
        self.numeric_feature_col_thresholds=None
        self.numnum_original_stats=None
        self.numcat_original_stats=None

    # a helper function that bins columns
    def binner(self,x,num_bins, rescale:bool=False, return_bins:bool=False):
        """
        used in Bin().
        can be used to bin a vector, or to retrieve the bin edges.
        set return_bins==True and no binning will occur. only the bin edges will be returned
        bins are used with right==True, Hence,  left_edge < X <= right_edge for any X
        if rescale==False the bins are 1,2, ..., n
        if rescale==True the bins are scaled to the input vector min and max
        """
        original_min,original_max=x.min(),x.max()
        bins=np.linspace(original_min-1e-10,original_max+1e-10,num_bins+1,endpoint=True)
        result_vector=np.digitize(x.copy(), bins, right=True)
        if return_bins==True:
            return bins
        if (rescale==False) | (num_bins==1):  # in case that ordinal 1,2,n bins are desired | case when there is only one bin
            return result_vector
        digitized_min, digitized_max = result_vector.min(), result_vector.max()
        denominator = (digitized_max - digitized_min)
        if denominator == 0:
            return np.full_like(result_vector, original_min, dtype=float)
        return original_min + ((result_vector - digitized_min) * (original_max - original_min) / denominator )
    
    #examine  relationships prior to binning
    def pre_bin_relationships(self,
                            df,
                            numnum_meth_alpha_above:tuple|list|None=('welch',0.05,False),
                            numcat_meth_alpha_above:tuple|list|None=('kruskal',0.05,False),
                            numeric_columns:list|tuple=None,
                            categoric_columns:list|tuple=None,
                            numeric_target:str|list|tuple|None=None,
                            categoric_target:str|list|tuple|None=None ): 
        """
        prepares data for Bin().pair_column_headers()
        #it calls kruskal or anova for categoric to numeric; and spearman, pearson, kendal, welch, or student for numeric to numeric
        """        
        num_cat=self.column_comparison(df,
                        numnum_meth_alpha_above=None,
                        numcat_meth_alpha_above=numcat_meth_alpha_above,
                        catcat_meth_alpha_above=None,
                        numeric_columns=numeric_columns,
                        categoric_columns=categoric_columns,
                        numeric_target=numeric_target,
                        categoric_target=categoric_target )
        num_num_df=self.column_comparison(df,
                        numnum_meth_alpha_above=numnum_meth_alpha_above,
                        numcat_meth_alpha_above=None,
                        catcat_meth_alpha_above=None,
                        numeric_columns=numeric_columns,
                        categoric_columns=categoric_columns,
                        numeric_target=numeric_target,
                        categoric_target=categoric_target )
        #store the dataframes as class objects
        #self.numnum_original_stats, self.numcat_original_stats = num_num_df,num_cat
        
        return num_num_df,num_cat
    
    def pair_column_headers(self,
                            num_num_df:pd.DataFrame|None=None,
                            num_cat_df:pd.DataFrame|None=None):
        """
        processes output from Bin().pre_bin_relationships()
        prepares data for Bin().determine_min_number_of_bins()
        if either arg is entered as None, it returns None for that arg
        """
        
        num_num_pairs, cat_num_pairs = [], []

        if num_cat_df is not None:
            cat_num_pairs=tuple((i.column_b,i.column_a) for i in num_cat_df[['column_a','column_b']].itertuples())
        if num_num_df is not None:
            num_num_pairs=tuple([(i.column_a,i.column_b) for i in num_num_df[['column_a','column_b']].itertuples()]+[(i.column_b,i.column_a)  for i in num_num_df[['column_a','column_b']].itertuples()])
        return num_num_pairs, cat_num_pairs


    def get_abs_coefficient_stat(self,data,method:str='spearman'):
        """
        a function used in Bin().determine_min_number_of_bins()
        and passed to Bin().find_min_bins()
        """
        xcol,ycol=data.columns[0],data.columns[1]
        return np.abs(data[ycol].corr(data[xcol],method=method))
    


    def find_min_bins(self, data, y_col, x_columns, test_func, threshold, direction_of_relationship,method:str|None=None):
        """
        direction_of_relationship='lower'|'greater' indicates area where bins are still related to other columns
        returns: 
            cols_min_max_and_stat={
                            'column_i': 
                                    {'min_w_relationship':int or None,
                                    'max_w_no_relationship':int or None,
                                    'threshold_stat':np.float64() or None}, 
                            'column_i':...}
            global_min_for_these_columns:int
        accepts test_func as arg. it can be 
        self.one_way_kruskal_wallis(self,two_col_num_cat)
        or  self.one_way_ANOVA(self,two_col_df_x_y)
        or self.get_abs_coefficient_stat(self,data)
            # the self.binner() function will scale the bins to match the min/max of the data when a numnum comparison is made, 
            # so numnum_method needs to be passed as ('pearson','spearman','kendall','welch','student')
            # method (spearman, pearson, kendall) needs to be passed when using coefficient methods
            # and 'welch' or 'student' when using T-Tests
            # and all coefficient methods use 'greater', but p-value methods use 'lower', 
        or self.students_t_test(self, df: pd.DataFrame, x1: str, x2: str) 
        or self.welchs_t_test(self,df:pd.DataFrame,x1:str,x2:str)
        """
        mins_for_each_col=[]
        cols_min_max_and_stat={}
        for col in x_columns:
            lowest_possible_bins,highest_possible_bins=1,data.shape[0]
            low=lowest_possible_bins
            high=highest_possible_bins
            min_relation_max_no_relation_stat=[None,None,None]# this is what sets:  cols_min_max_and_stat[col]={'min_within_threshold':min_relation_max_no_relation_stat[0],'max_beyond_threshold':min_relation_max_no_relation_stat[1],'threshold_stat':min_relation_max_no_relation_stat[2]}
            while low<=high:
                mid=(low+high)//2
                if method in ('pearson','spearman','kendall','welch','student'):
                    # 1st edge case for spearman and pearson coefficients
                    # because 1 bin isn't enough
                    if mid==1:
                        # spearman and pearson are not applicable with only one value in the vector, to set max_beyond_threshold to 1 and continue to 2 bins
                        if method in ('pearson','spearman'):
                            min_relation_max_no_relation_stat[1]=mid
                            low=mid+1
                            continue
                    data['binned']=self.binner(data[y_col],mid,rescale=True)
                else:
                    data['binned']=self.binner(data[y_col],mid,rescale=True)  
                if method in ('kendall','spearman','pearson'):
                    stat = test_func(data[[col,'binned']],method=method)
                else:
                    stat = test_func(data[[col,'binned']])
                if not np.isfinite(stat):
                    min_relation_max_no_relation_stat[0] = None
                    min_relation_max_no_relation_stat[1] = mid
                    min_relation_max_no_relation_stat[2] = stat
                    break
                
                # 2nd edge case for 'pearson' and 'spearman' coefficient, 
                # case when linear relationship is meant to be avoided
                if mid==2:
                    if (method in ('pearson','spearman')) and (direction_of_relationship=='lower'):
                        if stat<threshold:
                            min_relation_max_no_relation_stat[0]=None
                            min_relation_max_no_relation_stat[1]=mid
                            min_relation_max_no_relation_stat[2]=stat
                            break

                #print(f"bin: {mid}      stat: {stat}      threshold: {threshold}")
                #if (y_col=='Purchase Amount (USD)' and col=='Review Rating')|(y_col=='Previous Purchases' and col=='Gender')|(y_col=='Review Rating' and col=='Size'):
                #print(f"Y: {y_col}, X: {col}, Stat: {stat}, Mid: {mid}")

                if direction_of_relationship=='lower':
                    if stat>=threshold:
                        min_relation_max_no_relation_stat[1]=mid
                        low=mid+1
                    else:
                        min_relation_max_no_relation_stat[0]=mid
                        min_relation_max_no_relation_stat[2]=stat
                        high=mid-1

                elif direction_of_relationship=='greater':
                    if stat<threshold:
                        min_relation_max_no_relation_stat[1]=mid
                        low=mid+1
                    else:
                        min_relation_max_no_relation_stat[0]=mid
                        min_relation_max_no_relation_stat[2]=stat
                        high=mid-1
            
            if min_relation_max_no_relation_stat[0]!=None:
                mins_for_each_col.append(min_relation_max_no_relation_stat[0])
            cols_min_max_and_stat[col]={'min_within_threshold':min_relation_max_no_relation_stat[0],'max_beyond_threshold':min_relation_max_no_relation_stat[1],'threshold_stat':min_relation_max_no_relation_stat[2]}
        global_min_for_these_columns=max(mins_for_each_col) if len(mins_for_each_col) > 0 else None  # find the max of all the mins for this y_col. That min that is consistent through all varables
        return global_min_for_these_columns , cols_min_max_and_stat




    def determine_min_number_of_bins(self,
                                     dataframe:pd.DataFrame, 
                                     num_num_pairs:list|tuple|None=None, 
                                     cat_num_pairs:list|tuple|None=None, 
                                     original_value_count_threashold:int=5, 
                                     numnum_meth_alpha_above:tuple|None=('welch',0.05,False),
                                     numcat_meth_alpha_above:tuple|None=('kruskal',0.05,False),
                                     categoric_target:str|list|None=None, 
                                     numeric_target:str|list|None=None,
                                      non_pair_numnum_numcat:bool|None=None ):
        """
        takes output of Bin().pair_column_headers() as input
        shares alphas with Bin().pre_bin_relationships()
        uses self.find_min_bins() and binner() internally

        if non_pair_numnum_numcat == True: num_num_pairs and cat_num_pairs should not be lists of nested pairs. 
                                           they should be un-nested univariate num and cat respectively
                                           they will be used as features
                                           by default, it is set to false and input from Bin().pair_column_headers() the original design input
                                IT WILL ALSO ONLY TAKE A NUMERIC TARGET AS BINNABLE
                                IT IS DESIGNED FOR USE WITH AnalyzeDataset.py to target variables one-by-one
                                           
        call coeff or p-value functions based on 
        numnum_method of ('spearman','kendall','pearson','welch','student')
        and numcat_method of ('kruskal','anova')
        tests methods should be entered as follows:
            numnum_meth_alpha_above:tuple|None=('welch',0.05,False),
            numcat_meth_alpha_above:tuple|None=('kruskal',0.05,False)
                where each tuple is (test_method, threshold, keep >= threshold): (str,float,bool)

        outputs: 
        result metrics = {'target column i': 
                                min_bins_that_maintain_global_relationships,
                        'target column n':.................}
        x_col_thresholds = {'feature column i': 
                                {'xi column_threshold': (min num bins with relationship,  
                                                        max num bins with no relationship, 
                                                        np.float64(coeff or p_value w/relationship))},
                            'feature column n':..........}
        considers categoric_target, numeric_target inputs when outputting
        """
        # make sure targets are list or None
        if isinstance(categoric_target,str):
            categoric_target=[categoric_target]
        if isinstance(numeric_target,str):
            numeric_target=[numeric_target]
        if num_num_pairs is None:
            num_num_pairs = []
        if cat_num_pairs is None:
            cat_num_pairs = []
        if numeric_target is None:
            numeric_target = []
        if categoric_target is None:
            categoric_target = []
        if non_pair_numnum_numcat is None:
            non_pair_numnum_numcat = False
        if non_pair_numnum_numcat==True:
            if isinstance(cat_num_pairs,str):
                cat_num_pairs=[cat_num_pairs]
            if isinstance(num_num_pairs,str):
                num_num_pairs=[num_num_pairs]

        # variables that keep the bins on the correct side of the thresholds
        
        if numcat_meth_alpha_above is not None:
            numcat_direction_of_relationship = 'lower' if numcat_meth_alpha_above[2] == False else 'greater' 
        if numnum_meth_alpha_above is not None:
            numnum_direction_of_relationship = 'lower' if numnum_meth_alpha_above[2] == False else 'greater'

        data=dataframe.copy()
        # extract the columns that will be y-target
        # if cat, then all num are binned based on cat targ (plus num targs to all cats) 
        if (not categoric_target) and (numeric_target) :
            cols_to_bin = numeric_target
        # disigned for AnalyzeDataset.py, where targets are input individually
        elif non_pair_numnum_numcat==True:
            cols_to_bin = numeric_target
        else:
            cols_to_bin=list(set([i[1] for i in cat_num_pairs]+[i[1] for i in num_num_pairs]))  #use index 1 for both 
        #track minumum bins and max, no-relationship bins
        minimums={}
        y_relation_to_x_col_thresholds={}
        #iterate through target-numeric columns
        for col in cols_to_bin:
            if data[col].nunique(dropna=True) <= original_value_count_threashold:
                continue
            #extract the columns that will be x-features
            if non_pair_numnum_numcat==True:
                x_cat_columns=cat_num_pairs
                x_num_columns=num_num_pairs
            else:
                x_cat_columns=tuple(set(i[0] for i in cat_num_pairs if i[1]==col))
                x_num_columns=tuple(set(i[0] for i in num_num_pairs if i[1]==col))


            # filter based on numeric_target and categoric_target
            if (numeric_target) or (categoric_target):

                if (not categoric_target):   # then numeric_target
                    if col not in numeric_target:   
                        continue
                elif (not numeric_target):   # then categoric_target
                    # only need cat targets
                    x_cat_columns=[catcolumn for catcolumn in x_cat_columns if catcolumn in categoric_target]
                    if ( not x_cat_columns) or (numcat_meth_alpha_above is None) :
                        continue
                    x_num_columns=()   # no need to test col against num targets                        
                else:   # they are both 
                    # filter if col not target, else test all against col
                    if col not in numeric_target: 
                        x_cat_columns=[catcolumn for catcolumn in x_cat_columns if catcolumn in categoric_target]
                        if  (not x_cat_columns) or (numcat_meth_alpha_above is None) :
                            continue
                        x_num_columns=()   # no need to test col against num  


            # make calls to func min bins and metrics
            min_number_of_bins_numerical, min_number_of_bins_categorical = None, None
            # boolean varaibles: 
            is_cat_columns= not (not x_cat_columns)
            not_cat_columns= (not x_cat_columns)
            is_numcat_metrics=(numcat_meth_alpha_above is not None)
            not_numcat_metrics=(numcat_meth_alpha_above is None)
            is_num_columns= not (not x_num_columns)
            not_num_columns=(not x_num_columns)
            is_numnum_metrics=(numnum_meth_alpha_above is not None)
            not_numnum_metrics=(numnum_meth_alpha_above is None)

            if is_cat_columns and is_numcat_metrics:                
                numcat_method = numcat_meth_alpha_above[0] 
                if numcat_method=='kruskal':
                    min_number_of_bins_categorical,column_binning_metrics_categorical=self.find_min_bins( data, col, x_cat_columns, self.one_way_kruskal_wallis, numcat_meth_alpha_above[1], direction_of_relationship=numcat_direction_of_relationship)
                elif numcat_method=='anova':
                    min_number_of_bins_categorical,column_binning_metrics_categorical=self.find_min_bins( data, col, x_cat_columns, self.one_way_ANOVA, numcat_meth_alpha_above[1], direction_of_relationship=numcat_direction_of_relationship)  
            if is_num_columns and is_numnum_metrics:
                numnum_method = numnum_meth_alpha_above[0]
                if numnum_method in ('pearson','spearman','kendall'):
                    min_number_of_bins_numerical,column_binning_metrics_numerical=self.find_min_bins( data, col, x_num_columns, self.get_abs_coefficient_stat, numnum_meth_alpha_above[1], direction_of_relationship=numnum_direction_of_relationship,method=numnum_method)
                elif numnum_method=='welch':
                    min_number_of_bins_numerical,column_binning_metrics_numerical=self.find_min_bins( data, col, x_num_columns, self.welchs_t_test, numnum_meth_alpha_above[1], direction_of_relationship=numnum_direction_of_relationship,method=numnum_method)
                elif numnum_method=='student':
                    min_number_of_bins_numerical,column_binning_metrics_numerical=self.find_min_bins( data, col, x_num_columns, self.students_t_test, numnum_meth_alpha_above[1], direction_of_relationship=numnum_direction_of_relationship,method=numnum_method)            
            if ( not_num_columns or not_numnum_metrics ) and ( not_cat_columns or not_numcat_metrics ):
                #warnings.warn(f"For {col}, There are no potential solutions at these thresholds.", UserWarning)
                continue
            #update threshold metrics
            elif (is_num_columns and is_numnum_metrics) and (is_cat_columns and is_numcat_metrics):
                column_metrics=column_binning_metrics_categorical | column_binning_metrics_numerical
                y_relation_to_x_col_thresholds[col]=column_metrics
                #update the minimum bin that retains   ALL tested relationships
                if min_number_of_bins_categorical is None and min_number_of_bins_numerical is None:
                    minimum = None
                elif min_number_of_bins_categorical is None:
                    minimum = min_number_of_bins_numerical
                elif min_number_of_bins_numerical is None:
                    minimum = min_number_of_bins_categorical
                else:
                    minimum = max(min_number_of_bins_numerical,min_number_of_bins_categorical)
                minimums[col]=minimum
            elif ( is_num_columns and is_numnum_metrics ) and ( not_cat_columns or not_numcat_metrics ):
                y_relation_to_x_col_thresholds[col]=column_binning_metrics_numerical
                #update the minimum bin that retains   ALL tested relationships
                if min_number_of_bins_numerical is None:
                    minimum = None
                else:
                    minimum = min_number_of_bins_numerical
                minimums[col]=minimum
            elif ( not_num_columns or not_numnum_metrics ) and (is_cat_columns and is_numcat_metrics):
                y_relation_to_x_col_thresholds[col]=column_binning_metrics_categorical
                #update the minimum bin that retains   ALL tested relationships
                if min_number_of_bins_categorical is None:
                    minimum = None
                else:
                    minimum = min_number_of_bins_categorical
                minimums[col]=minimum
        return minimums, y_relation_to_x_col_thresholds



    #=============================================================================================================================================================
    #this needs edge case when xfeature columns is none and/or ytarget columns is none, presently, it attempts to detect datatypes
    #=============================================================================================================================================================

    
    def relational_binner(self,
                        df,
                        numnum_meth_alpha_above:tuple|None=('welch',0.05,False),
                        numcat_meth_alpha_above:tuple|None=('kruskal',0.05,False),
                        original_value_count_threashold:int=5,
                        numeric_columns:list|tuple|None=None,
                        categoric_columns:list|tuple|None=None,
                        numeric_target:str|list|None=None,
                        categoric_target:str|list|None=None ):
        """
        This calculated before statistics and uses those to elect bin cantidates
        where original_value_count_threashold=5 is the default threashold. columns with <=threashold unique values won't be considered
        
        call coeff or p-value functions based on 
        numnum_method = one of ('spearman','kendall','pearson','welch','student')
        and numcat_method = one of ('kruskal','anova')
        
        calling this function will create class object that give metrics about bins

        parameters:
            df : a pandas dataframe
            numnum_meth_alpha_above:tuple|None=('welch',0.05,False),
            numcat_meth_alpha_above:tuple|None=('kruskal',0.05,False)
                where each tuple is (test_method, threshold, keep >= threshold): (str,float,bool)
            original_value_count_threashold : sets the minimum number of unique values a column can have to be considered for binning. default==5
            numeric_columns : allows input of numeric columns. default is autodetect. it is an all or none approach. either input columns or autodetect
            categoric_columns : allows input of categoric columns. default is autodetect. it is an all or none approach. either input columns or autodetect
            categoric_target : a categoric column that can be tested against every numeric column [but only one numeric at a time]
            numeric_target :  a numeric that can be tested against all other columns
                    Targets are optional. if left as None, ALL combinations are considered
                    Targets are exclusive, such that no consiteration is made for any pair of columns that doesn't involve a target

        """
        if (numnum_meth_alpha_above is not None) and (not isinstance(numnum_meth_alpha_above[2],bool)):
            raise ValueError("Variable numnum_meth_alpha_above for Numeric to Numeric at index 2 should be a boolean value.", ValueError)
        if (numcat_meth_alpha_above is not None) and (not isinstance(numcat_meth_alpha_above[2],bool)):
            raise ValueError("Variable numcat_meth_alpha_above for Categoric to Numeric at index 2 should be a boolean value.", ValueError)
        num_num_df,num_cat_df=self.pre_bin_relationships(df,
                                                         numnum_meth_alpha_above=numnum_meth_alpha_above,
                                                         numcat_meth_alpha_above=numcat_meth_alpha_above,
                                                         numeric_columns=numeric_columns,
                                                         categoric_columns=categoric_columns,
                                                         numeric_target=numeric_target,
                                                         categoric_target=categoric_target )
        num_num_pairs, cat_num_pairs=self.pair_column_headers(num_num_df,
                                                              num_cat_df)
        self.numeric_target_column_minimums, self.numeric_feature_col_thresholds = self.determine_min_number_of_bins(df, 
                                                                                                                     num_num_pairs, 
                                                                                                                     cat_num_pairs, 
                                                                                                                     original_value_count_threashold, 
                                                                                                                     numnum_meth_alpha_above=numnum_meth_alpha_above,
                                                                                                                     numcat_meth_alpha_above=numcat_meth_alpha_above,
                                                                                                                     categoric_target=categoric_target,
                                                                                                                     numeric_target=numeric_target )
        return self.numeric_target_column_minimums
    #=============================================================================================================================================================
    #=============================================================================================================================================================
    