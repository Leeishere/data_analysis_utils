import pandas as pd
import numpy as np
import scipy 
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None


class MuEstimator:
   

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Z AND T SCORES :--> 1 for proportion estimation and  2 tail for linear estimation
    # one_tailed_t_score_lookup(), two_tailed_t_score_lookup(), one_tailed_z_score_lookup(), two_tailed_z_score_lookup()


    def two_tailed_z_score_lookup(self, confidence_level: float = 0.95):
        two_tail = (1 + confidence_level) / 2
        z_score = scipy.stats.norm.ppf(two_tail)    
        return z_score

    def one_tailed_z_score_lookup(self, confidence_level: float = 0.95):
        # For one-tailed, we use the confidence level directly
        one_tail = confidence_level
        z_score = scipy.stats.norm.ppf(one_tail)        
        return z_score


    def two_tailed_t_score_lookup(self, dof,confidence_level: float = 0.95):
        """
        Return the critical t-score for a two-tailed confidence interval.
        
        Parameters:
            confidence_level (float): Desired confidence level (default 0.95).
            df (int): Degrees of freedom (sample size - 1).
        
        Returns:
            float: Critical t-score.
        """
        alpha = 1 - confidence_level
        return scipy.stats.t.ppf(1 - alpha/2, dof)        
        
    
    def one_tailed_t_score_lookup(self, dof,confidence_level: float = 0.95):
        """
        Return the critical t-score for a one-tailed confidence interval.
        
        Parameters:
            confidence_level (float): Desired confidence level (default 0.95).
            df (int): Degrees of freedom (sample size - 1).
        
        Returns:
            float: Critical t-score.
        """
        alpha = 1 - confidence_level
        return scipy.stats.t.ppf(1 - alpha, dof)        
        

    #--------------------------------------------------------------------------------------------------------------------------------------------------
    # PROPORTION PROBABILITIES FOR SPARSE AND OTHERWISE 
    # PROPORTION STANDARD ERROR(SE) 
    # LINEAR STANDARD ERROR(SE)

    def proportion_one_hot(self,vec):
        """
        accepts a vector of [True,False]'s or [1,0]'s
        Compute the proportion of True/False (or 1/0) in a vector.
        Works with numpy arrays, pandas Series, and cudf Series.
        Parameters
        ----------
        vec : numpy.ndarray, pandas.Series, or cudf.Series
            Sparse vector containing [1,0] or [True,False].
        Returns
        -------
        dict
            {"True": proportion_true, "False": proportion_false}
        """
        # Convert to numpy array for uniform handling
        try:
            arr = vec.values  # pandas or cudf
        except AttributeError:
            arr = vec         # numpy already
        # normalize to boolean
        arr = np.asarray(arr).astype(bool)  
        total = arr.size
        if total == 0:
            return {"True": 0.0, "False": 0.0}
        true_count = arr.sum()
        false_count = total - true_count
        return {
            "True": true_count / total,
            "False": false_count / total        }
    

    def proportion_successes(self,denominator, successes):
        """
        Compute the proportion of successes to denominator.
        Works with numpy arrays, pandas Series, and cudf Series.
        Parameters
        ----------
        denominator : numpy.ndarray, pandas.Series, or cudf.Series
            Vector of denominator counts (must be >0).
        successes : numpy.ndarray, pandas.Series, or cudf.Series
            Vector of success counts (must be <= denominator).
        Returns
        -------
        numpy.ndarray
            Array of proportions (successes / denominator).
        """
        # Convert both inputs to numpy arrays for uniform handling
        try:
            denominator_arr = denominator.values  # pandas or cudf
        except AttributeError:
            denominator_arr = denominator
        try:
            successes_arr = successes.values
        except AttributeError:
            successes_arr = successes
        denominator_arr = np.asarray(denominator_arr, dtype=float)
        successes_arr = np.asarray(successes_arr, dtype=float)
        # Safety checks
        if denominator_arr.shape != successes_arr.shape:
            raise ValueError("denominator and successes must have the same shape.")
        if np.any(denominator_arr < 0):
            raise ValueError("No denominator can be < 0")
        if np.any(successes_arr > denominator_arr):
            raise ValueError("Successes cannot exceed denominator.")
        
        no_denominator_indexes=denominator_arr==0
        if np.any(no_denominator_indexes):
            warnings.warn("Warning: some denominator are zero; proportions will include NaN values.")
        #create an empty array for index masking
        proportions = np.empty_like(denominator_arr, dtype=float)
        proportions[~no_denominator_indexes] = successes_arr[~no_denominator_indexes] / denominator_arr[~no_denominator_indexes]
        proportions[no_denominator_indexes] = np.nan 
        return proportions    

    #standard error for continious or discrete data
    def se_mean(self,sample_std, n_of_population):
        """
        Compute standard error of the mean: s / sqrt(n).        
        Parameters
        ----------
        s : scalar, list, numpy.ndarray, pandas.Series, or cudf.Series ==>Standard deviation(s).
        n : scalar, list, numpy.ndarray, pandas.Series, or cudf.Series ==>Sample size(s).        
        Returns -------> Same type as input --------> Standard error(s).
        """
        s_arr = np.asarray(sample_std, dtype=float)
        n_arr = np.asarray(n_of_population, dtype=float)
        result = s_arr / np.sqrt(n_arr)
        return result

    #standard error for proporitons
    def se_proportion(self,proportion, denominator):
        """
        Compute standard error of a proportion: sqrt(p*(1-p)/n).    
        Parameters
        ----------
        p : scalar, list, numpy.ndarray, pandas.Series, or cudf.Series ==>Proportion(s).
        n : scalar, list, numpy.ndarray, pandas.Series, or cudf.Series ==>Sample size(s).    
        Returns -------> Same type as input  -------> Standard error(s).
        """
        p_arr = np.asarray(proportion, dtype=float)
        n_arr = np.asarray(denominator, dtype=float)
        result = np.sqrt((p_arr * (1 - p_arr)) / n_arr)
        return result
    
    def margin_of_error(self,statistic, standard_error):
        """
        Compute margin of error: statistic * standard error.        
        Parameters
        ----------
        statistic : scalar, list, numpy.ndarray, pandas.Series, or cudf.Series ==> Statistic(s) (e.g., z-score or t-score).
        se : scalar, list, numpy.ndarray, pandas.Series, or cudf.Series ==>  Standard error(s).        
        Returns   ------>   Same type as input   ------> Margin of error(s).
        """
        stat_arr = np.asarray(statistic, dtype=float)
        se_arr = np.asarray(standard_error, dtype=float)

        if stat_arr.shape != se_arr.shape:
            raise ValueError("Statistic and SE must have the same shape.")

        result = abs(stat_arr * se_arr)
        return result
    
    #-----------------------------------------------------------------------------------------------------------------------------------------------------
    #mu estimators

    #mu of continuous or discrete data
    def mean_estimator(self, mean, sample_std, num_observations, confidence_level,MOE_only=False):
        """

        """
        try:
            mean_arr = mean.values  # pandas or cudf
        except AttributeError:
            mean_arr = mean
        try:
            sample_std_arr = sample_std.values
        except AttributeError:
            sample_std_arr = sample_std
        try:
            num_observations_arr = num_observations.values
        except AttributeError:
            num_observations_arr = num_observations
        mean_arr = np.asarray(mean_arr, dtype=float)
        sample_std_arr = np.asarray(sample_std_arr, dtype=float)
        num_observations_arr = np.asarray(num_observations_arr, dtype=float)
        
        #build an empty statistic vector
        statistics=np.empty_like(num_observations,dtype=float)
        #identify indexes with num observations <30
        obs_under_30=num_observations_arr<30
        #steps:
        #determine t scores for low observation datapoints
        #a scalar z 
        statistics[obs_under_30]=self.two_tailed_t_score_lookup( num_observations_arr[obs_under_30]-1,confidence_level)
        z_statistic=self.two_tailed_z_score_lookup(confidence_level)
        statistics[~obs_under_30]=z_statistic

        #make call to get standard errors
        standard_errors = self.se_mean(sample_std_arr, num_observations_arr)
        #make call to get MOE
        MOE = self.margin_of_error(statistics, standard_errors)
        if MOE_only==True:
            return MOE
        #calculate bounds
        lower , upper = mean_arr-MOE , mean_arr+MOE
        """
        #return according to mean input datatype
        if isinstance(mean, pd.Series):
            return pd.DataFrame({"lower": lower, "upper": upper}, index=mean.index)
        try:
            if isinstance(mean, cudf.Series):
                return cudf.DataFrame({"lower": lower, "upper": upper}, index=lower.index)
        except NameError:
            pass
        if isinstance(mean, list):
            return list(zip(lower, upper))"""
        return np.column_stack((lower, upper))  # default NumPy



    #mu of proportion data
    def proportion_estimator(self, proportions, num_observations, confidence_level,MOE_only=False):
        """

        """
        try:
            proportions_arr = proportions.values  # pandas or cudf
        except AttributeError:
            proportions_arr = proportions
        try:
            num_observations_arr = num_observations.values
        except AttributeError:
            num_observations_arr = num_observations
        proportions_arr = np.asarray(proportions_arr, dtype=float)
        num_observations_arr = np.asarray(num_observations_arr, dtype=float)
        
        #build an empty statistic vector
        statistics=np.empty_like(num_observations,dtype=float)
        #identify indexes with num observations <30
        obs_under_30=(num_observations_arr<30)&~np.isnan(proportions_arr)
        obs_equal_to_or_over_30=(num_observations_arr>=30)&~np.isnan(proportions_arr)
        isnull_indexes=np.isnan(proportions_arr)    
        #create empty arrays for masking values into
        standard_errors = np.empty_like(proportions_arr,dtype=float)
        MOE = np.empty_like(statistics,dtype=float)
        lower,upper=np.empty_like(proportions_arr,dtype=float),np.empty_like(proportions_arr,dtype=float)
        #retrieve z stat        
        z_statistic=self.two_tailed_z_score_lookup(confidence_level)
        #steps:
        #determine t scores for low observation datapoints
        #a scalar z
        #make call to get standard errors
        #make call to get MOE
        #calculate bounds
        statistics[obs_under_30]=self.two_tailed_t_score_lookup(num_observations_arr[obs_under_30]-1,confidence_level)
        statistics[obs_equal_to_or_over_30]=z_statistic
        standard_errors[~isnull_indexes] = self.se_proportion(proportions_arr[~isnull_indexes], num_observations_arr[~isnull_indexes]) 
        MOE[~isnull_indexes] = self.margin_of_error(statistics[~isnull_indexes], standard_errors[~isnull_indexes])
        if MOE_only==True:
            return MOE
        lower[~isnull_indexes] , upper[~isnull_indexes] = proportions_arr[~isnull_indexes]-MOE[~isnull_indexes] , proportions_arr[~isnull_indexes]+MOE[~isnull_indexes]
        """
        #return according to mean input datatype
        if isinstance(proportions, pd.Series):
            return pd.DataFrame({"lower": lower, "upper": upper}, index=proportions.index)
        try:
            if isinstance(proportions, cudf.Series):
                return cudf.DataFrame({"lower": lower, "upper": upper}, index=lower.index)
        except NameError:
            pass
        if isinstance(proportions, list):
            return list(zip(lower, upper))
        """
        return np.column_stack((lower, upper))  # default NumPy

    #============================================================================================================
    #============================================================================================================

    #functions to return grouped dataframes with mu estimates
    #will need MuEstimator
    #
    #
    #


    def filter_data_by_season(self,df,season_column:str,season:str):
        return df.loc[df[season_column]==season]

    # these 3 call MuEstimator()

    def get_single_variable_proportions(self,product_column: pd.Series) -> pd.DataFrame:
        """
        takes a series as input and returns a dataframe with proportions
        """
        counts = product_column.value_counts()
        num_observations=[counts.sum()]*len(counts.index)    
        proportions = self.proportion_successes(num_observations,counts)
        cat_var='category_variable' if product_column.name is None else product_column.name
        return pd.DataFrame({cat_var:counts.index,'proportion':proportions,'num_observations':num_observations,'successes':counts.values})


    def get_bivariable_proportions(self,dataframe, partition_variable:str|list, category_variable:str, proportion_within_partition:bool=True):
        """
        takes a dataframe, a partition columns, and a target column as input. returns a new dataframe with proportions
        proportion_within_partition:bool=True/False determins the denominator is sum within partition or sum through category
        """
        if isinstance(partition_variable,str):
            partition_variable=[partition_variable]
        data=dataframe.copy()
        data=data.groupby(partition_variable+[category_variable],as_index=False,observed=True).size().rename(columns={'size':'successes'})
        if proportion_within_partition==True:
            sizes=data.groupby(partition_variable,as_index=False,observed=True)['successes'].sum().rename(columns={'successes':'num_observations'})
            data = data.merge(sizes,how='left',right_on=partition_variable,left_on=partition_variable)
        else:
            sizes=data.groupby(category_variable,as_index=False,observed=True)['successes'].sum().rename(columns={'successes':'num_observations'})
            data = data.merge(sizes,how='left',right_on=category_variable,left_on=category_variable)
        data['proportion'] = self.proportion_successes(data['num_observations'],data['successes'])
        data=data.reset_index(drop=False)
        return data[partition_variable+[category_variable,'proportion','num_observations','successes']]


    #-----------------------------------------------------------------------------------------------------------------------------------------
    # Create grouped mu estimate range dataframes


    def get_proportion_estimate_df(self,dataframe:pd.DataFrame,target_col:str,confidence_level:float=0.95,partition_by:str|list=None, proportion_within_partition:bool=True):
        """
        takes a dataframe, target column, confidence interval[default 0.95], optional partition column(s)[default None], and sort[default None] as input.
        returns a new dataframe with <partition column>, target column, num_observations column, upper column, and lower column.
        where upper and lower are the mu estimate intervals
        """
        if isinstance(partition_by,str):
            partition_by=[partition_by]
        if partition_by is None or partition_by == []:
            estimate_df=self.get_single_variable_proportions(dataframe[target_col])
        else:
            estimate_df=self.get_bivariable_proportions(dataframe, partition_by, target_col, proportion_within_partition)
        if confidence_level>0.999999:
            raise ValueError('Confidence Level Out of Bounds\nexceeds-->0.999999')
        estimate=self.proportion_estimator(estimate_df['proportion'],estimate_df['num_observations'],confidence_level)
        estimate_df['lower'],estimate_df['upper'] = estimate[:,0],estimate[:,1]
        if partition_by==None or partition_by==[]: 
            estimate_df=estimate_df.sort_values(by=['successes',target_col,'proportion'],ascending=[True,False,True]).reset_index(drop=True)        
        else: 
            estimate_df=estimate_df.sort_values(by=[target_col]+['successes','proportion']+partition_by,ascending=[False,True,True]+[False for i in partition_by]).reset_index(drop=True)
        return estimate_df



    def get_mean_estimate_df(self,df,target_col:str,confidence_level:float=0.95,partition_cols:list=None):
        """
        takes dataframe, a target column, a confidence interval[default 0.95], a list of 0 to n partition columns to group by, and sort[default None].
        return a grouped dataframe with aggregated columns: 'min','mean','median','max','std','size','lower','upper'
        """
        if type(partition_cols)==str:
            partition_cols=[partition_cols]
        if partition_cols==None or partition_cols==[]:
            estimate_df = pd.DataFrame({target_col:[target_col],'min':[df[target_col].min()],'mean':[df[target_col].mean()],'median':[df[target_col].median()],'max':[df[target_col].max()],'std':[df[target_col].std()],'size':[df[target_col].count()]})
        else:
            estimate_df = df.groupby(partition_cols,as_index=False,observed=True)[target_col].agg(['min','mean','median','max','std','size'])
        estimates=self.mean_estimator(estimate_df['mean'],estimate_df['std'],estimate_df['size'],confidence_level)
        estimate_df['lower'],estimate_df['upper'] = estimates[:,0],estimates[:,1]
        if partition_cols==None or partition_cols==[]: 
            estimate_df=estimate_df.sort_values(by=['size','mean'],ascending=[True,True]).reset_index(drop=True)
        else:
            estimate_df=estimate_df.sort_values(by=['size','mean']+partition_cols,ascending=[True,True]+[False for i in partition_cols]).reset_index(drop=True)
        return estimate_df

    # get plot data
    def get_floating_mu_hbar_plot_data(self,plot_df_:pd.DataFrame,target_col:str,partition_cols:str|list|None):
        """
        return lefts,mu,median,widths,Y_ticks,X_label,intersects,target_col
        where lefts is lower bounds, mu is mu, media is median or none in case of proportions, widths is MOE*2, 
        Y_ticks is concatenated partition feature labels "[partition_cols]-> target_col" or "target_col", 
        X_label is target col w/ partitions by feature label, 
        intersects is a list of feature partitions [it is partition_cols, and it is unused downstream. It is retained for features that aren't yet supported], 
        target col is the string title of the target col that represents the mu
        """
        if type(partition_cols)==str:
            partition_cols=[partition_cols]
        plot_df=plot_df_.copy()
        if target_col not in plot_df.columns:
            plot_df[target_col]=target_col
        partitions=False if partition_cols is None or partition_cols==[] else True
        if partitions==True:
            Y_ticks="] -> "+plot_df[target_col].astype(str)
            first=True  # to help with string format
            for col in partition_cols: 
                if first==False:
                    Y_ticks="'"+plot_df[col].astype(str)+"', "+Y_ticks.astype(str)
                else:
                    Y_ticks="'"+plot_df[col].astype(str)+"'"+Y_ticks.astype(str)
                    first=False
            if partitions==True:
                    Y_ticks="["+Y_ticks.astype(str)
        else: Y_ticks=plot_df[target_col].astype(str)        
        Y_ticks=Y_ticks.values
        #X_label    
        X_label=target_col+'\nPartitioned By\n'+str(partition_cols[::-1]) if partitions==True else target_col.title()
        

        #extract plot data  {upper and lower are used to calculate lefts and widths
        #check categorical or continuous
        mu_is_mean=True if 'mean' in plot_df_.columns else False     
        if mu_is_mean==True: 
            lefts,mu,median,widths,counts =  plot_df['lower'].values, plot_df['mean'].values, plot_df['median'].values, (plot_df['upper']-plot_df['lower']).values, plot_df['size'].values
        else:
            lefts,mu,widths,counts=plot_df['lower'].values, plot_df['proportion'].values, (plot_df['upper']-plot_df['lower']).values, plot_df['successes'].values 
            median=None
        intersects = partition_cols if partition_cols is not None else []
        return lefts,mu,median,widths,Y_ticks,X_label,intersects,counts,target_col

    def plot_floating_mu_hbar(self,lefts,mu,median,widths,Y_ticks,X_label,intersects,counts,target_col,confidence_level,legend_median=True,plot_title=None,streamlit=False):
        """
        accepts data produced direct from get_floating_mu_hbar_plot_data(); plus plot_title
        
        """  
        if median is None: mu_is_mean=False
        else: mu_is_mean=True
        if plot_title is None: 
                plot_title=f"{target_col.title()}"
        common_fontsize = 16
        scatter_dot_size=3.1415926535 * (common_fontsize / 2) ** 2   #####need a formula to convert fontsize to to area of a circle 
        mu_color="#09145A"
        median_color="#A61313"
        mu_median_edgecolor="white"
        bar_color="#646B7BED"
        legend_text_color='white'
        figure_color='black'
        axis_color='black'  
        num_y_ticks=len(Y_ticks)
        count_color="#4B0707"
        text_color='white'

        row_height_in = common_fontsize / 72 * 1.6   # points → inches
        fig_height = max(6, (len(Y_ticks) * row_height_in)+2)

        sns.set_theme(
            rc={
                "figure.figsize": (20, fig_height),
                "figure.titlesize": common_fontsize*1.75,
                "font.size": common_fontsize,
                "axes.labelsize": common_fontsize,
                "axes.titlesize": common_fontsize*1.5,
                "xtick.labelsize": common_fontsize,
                "ytick.labelsize": common_fontsize,
                "legend.fontsize": common_fontsize,
                "axes.formatter.limits": (-3, 5),
                "axes.formatter.useoffset": True,
                "axes.titlecolor": text_color,
                "xtick.color": text_color,
                "ytick.color": text_color,
                "axes.labelcolor": text_color,
                "axes.edgecolor": "black",
                "legend.labelcolor": text_color,
            }   )

        fig, ax = plt.subplots(1,2)
        plt.subplot(1,2,1)
        ax[0].barh(
            y=np.arange(num_y_ticks),     # numeric positions = better control
            width=widths,
            left=lefts,
            color=bar_color,
            edgecolor=bar_color,
            label="Confidence Interval"
        )

        if mu_is_mean and legend_median==True:
            ax[0].scatter(
                median, np.arange(num_y_ticks),
                color=median_color,
                edgecolor=mu_median_edgecolor,
                s=scatter_dot_size,
                label="Median"
            )

        center_label = "Mean" if mu_is_mean else "Proportion"
        ax[0].scatter(
            mu, np.arange(num_y_ticks),
            color=mu_color,
            edgecolor=mu_median_edgecolor,
            s=scatter_dot_size,
            label=center_label
        )

        ax[0].set_yticks(np.arange(num_y_ticks))
        ax[0].set_yticklabels(Y_ticks)
        ax[0].tick_params(axis="y", pad=6)
        ax[0].set_ylim(-0.5, num_y_ticks - 0.5)

        ax[0].set_xlabel(X_label)
        ax[0].tick_params(axis="x", labelrotation=45)
        for label in ax[0].get_xticklabels():
            label.set_ha("right")
        if not mu_is_mean:
            ax[0].set_xlim((0,1))

        ax[0].set_facecolor(axis_color)
        ax[0].grid(linewidth=0.3,which='both')
        ax[0].set_title(f'{center_label} @ {confidence_level}% Confidence Level')
        
        plt.subplot(1,2,2)
        center_label = "Mean" if mu_is_mean else "Proportion"
        sns.barplot(
            x=counts,
            y=Y_ticks,
            orient="h",
            ax=ax[1],
            color=count_color,
            edgecolor=count_color,
            label='Counts'
        )
        ax[1].legend_.remove()

        ax[1].set_yticks(np.arange(num_y_ticks))
        ax[1].set_yticklabels(['']*num_y_ticks)
        ax[1].tick_params(axis="y", pad=6)
        ax[1].set_ylim(-0.5, num_y_ticks - 0.5)

        ax[1].set_xlabel('Count Within Group(s)')
        ax[1].tick_params(axis="x", labelrotation=45)
        for label in ax[1].get_xticklabels():
            label.set_ha("right")
    
        ax[1].set_facecolor(axis_color)
        ax[1].grid(linewidth=0.3,which='both')
        ax[1].set_title('Count(s)')

        fig.suptitle(
            f"\n\n{plot_title}\nSorted by Count",
            color=text_color,
            y=1
        )

        fig.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1),
            ncol=3 if not mu_is_mean or legend_median==False else 4,
            frameon=False
        )
        
        fig.patch.set_facecolor(figure_color)
        plt.tight_layout()
        if streamlit==False:
            plt.show()
        else:
            fig = plt.gcf()
            st.pyplot(fig,clear_figure=True)  
        
        plt.rcdefaults()



    def get_floating_proportion_hbar(self,dataframe:pd.DataFrame,target_col:str,confidence_level:float=0.95,partition_by:list=[],plot_title=None,streamlit=False, proportion_within_partition:bool=True):
        """
        
        """
        mu_data = self.get_proportion_estimate_df(dataframe,target_col,confidence_level,partition_by, proportion_within_partition)
        plot_data = self.get_floating_mu_hbar_plot_data(mu_data,target_col,partition_by)
        #generate plot
        self.plot_floating_mu_hbar(*plot_data,confidence_level,plot_title=plot_title,streamlit=streamlit)

    
    def get_floating_mean_hbar(self,dataframe:pd.DataFrame,target_col:str,confidence_level:float=0.95,partition_by:list=None,plot_title=None,median=False,streamlit=False):
        """
        
        """
        mu_data = self.get_mean_estimate_df(dataframe,target_col,confidence_level,partition_by)
        plot_data = self.get_floating_mu_hbar_plot_data(mu_data,target_col,partition_by)
        #generate plot
        self.plot_floating_mu_hbar(*plot_data,confidence_level,plot_title=plot_title,legend_median=median,streamlit=streamlit)


