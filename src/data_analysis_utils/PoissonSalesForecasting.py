import random
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-dark')

import seaborn as sns
import pandas as pd
import numpy as np

import warnings

from data_analysis_utils.ProbabilisticModeling import ProbabilisticModeling


### streamlit import
try:
    import streamlit as st
except ModuleNotFoundError:
    st=None

#Note that due in part to the nature of the Consumer_Habits.csv data, support is limited to returning forecasts based on week, month, or season

class PoissonSalesForecasting(ProbabilisticModeling):
    def __init__(self):
        self.frequency_of_purchase_col_header='Frequency of Purchases'   # a column header in Consumer_Habbits.csv: any of:['Annually','Quarterly','Monthly','Bi-Weekly','Fortnightly','Weekly']
        self.total_prev_purchases_col_header='Previous Purchases'    # a column header in Consumer_Habbits.csv:  the number of previous purchases
        self.individual_sale_purchase_amount='Purchase Amount (USD)'    #used to calculate the average single item purchase price for each season
        self.season_col_header_data_is_partitioned_by='Season'   # a column header in Consumer_Habbits.csv: 'Winter','Summer','Sprint','Fall'
        self.days_of_patronage_col_header='Days of Patronage'   # returned by add_day_column(): total customer lifetime so far
        self.days_between_purchases_col_header='Days Between Purchases'   # returned by add_days_between_purchase_column()
        self.seasons=None    #seasons=predict_total_sales_per_season(self,df)
        #get period for periods <= season size
        self.period_details = {'w':None,'m':None,'q':None} #[partitioned_seasons, period_to_aggrigate, period_size_in_days, periods_per_season]+[period_title,total_periods,zero_to_max,starts,periods]



    def freq_factor(self,x):
        if x=='Every 3 Months': return 365/4
        if x=='Annually': return 365/1
        if x=='Quarterly': return 365/4
        if x=='Monthly': return 365/12
        if x=='Bi-Weekly': return 7/2
        if x=='Fortnightly': return 14
        if x=='Weekly': return 7


#------------------------------------------------------------------------------------------------------------------------------------
    def add_days_between_purchase_column(self,df,purchase_frequency=None,new_column_name=None):
        """ returns the dataframe with an added column 
        parameters: dataframe,purchase_frequency,total_previous_purchases,new_column_name
        where purchase frequency should be of:  'Every 3 Months', 'Annually', 'Quarterly', 'Monthly', 'Bi-Weekly', 'Fortnightly', 'Weekly'
        """
        if purchase_frequency==None:
            purchase_frequency=self.frequency_of_purchase_col_header
        else:
            #warnings.warn(f"The purchases frequency column is being fit as {purchase_frequency}\nThat may or may not be the default for the Consumer_Habits Dataset.")
            self.frequency_of_purchase_col_header=purchase_frequency
        if new_column_name==None:
            new_column_name=self.days_between_purchases_col_header
        else:
            self.days_between_purchases_col_header=new_column_name
        df[new_column_name]=df[purchase_frequency].map(self.freq_factor)
        return df
    
    
    
    def add_day_column(self,df,total_previous_purchases=None,new_column_name=None):
        """ returns the dataframe with an added column 
        parameters: dataframe,total_previous_purchases,new_column_name
        """

        if total_previous_purchases is None:
            total_previous_purchases = self.total_prev_purchases_col_header
        else:
            #warnings.warn(f"The total_prev_purchases_col_header is being fit as {total_previous_purchases}\nThat may or may not be the default for the Consumer_Habits Dataset.")
            self.total_prev_purchases_col_header=total_previous_purchases
        if new_column_name is None:
            new_column_name = self.days_of_patronage_col_header
        else:
            self.days_of_patronage_col_header=new_column_name            

        df[new_column_name]=df[self.days_between_purchases_col_header]*df[total_previous_purchases]
        return df    
    
    def add_relevant_columns(self,data,total_prev_purchases_col_header=None,frequency_of_purchase_col_header=None):
        """
        Input dataframe should have columns: frequency_of_purchase and total_previous_purchases, the season column to partition by, and a column of sales amounts
        if frequency_of_purchase or total_previous_purchases are left as None, they will default to the ones in the Consumer_Habits Dataset.
        frequency should be of:  'Every 3 Months', 'Annually', 'Quarterly', 'Monthly', 'Bi-Weekly', 'Fortnightly', 'Weekly'
        """
        consumer_habits_dataset=data.copy()
        if frequency_of_purchase_col_header==None:
            frequency_of_purchase_col_header=self.frequency_of_purchase_col_header
        else:
            warnings.warn(f"The frequency_of_purchase_col_header is being fit as {frequency_of_purchase_col_header}\nThat may or may not be the default for the Consumer_Habits Dataset.")
            self.frequency_of_purchase_col_header=frequency_of_purchase_col_header
        if total_prev_purchases_col_header==None:
            total_prev_purchases_col_header=self.total_prev_purchases_col_header
        else:
            warnings.warn(f"The total_prev_purchases_col_header is being fit as {total_prev_purchases_col_header}\nThat may or may not be the default for the Consumer_Habits Dataset.")
            total_prev_purchases_col_header=self.total_prev_purchases_col_header
        consumer_habits_dataset=self.add_days_between_purchase_column(consumer_habits_dataset,purchase_frequency=frequency_of_purchase_col_header)
        consumer_habits_dataset=self.add_day_column(consumer_habits_dataset,total_previous_purchases=total_prev_purchases_col_header)
        return consumer_habits_dataset



    def simulate_sales_by_season(self,modified_consumer_habits_dataset,occurrence_multiplier:float,season_:str,period_size_in_days=int(365/12),random_seed=None):
        """
        where season_ refers to a season in the the consumer_habbits dataset "Season" column.
        calls on consumer_habits dataset columns headers
        other datasets are not presently supported
        returns cdf of P(num purchases per month) rowwise  
        """
        random.seed(a=random_seed, version=2)
        data=modified_consumer_habits_dataset.copy()
        occurrences = data[self.total_prev_purchases_col_header]
        periods    = data[self.days_of_patronage_col_header]/period_size_in_days
        season_mean_purchase=data.loc[data[self.season_col_header_data_is_partitioned_by]==season_][self.individual_sale_purchase_amount].mean()
        del data
        self.fit_poisson(periods,occurrences) 
        sales_per_period_probabilities=self.predict_poisson_cdf(occurrences*occurrence_multiplier/periods,True,True)   ## as is, it is conservative. occurrences here could be multiplied by 1.1 to bring total p of the cdf closer to 1
        del periods, occurrences
        def weighted_predictor(probability_list):
            k=len(probability_list)
            rng=range(k)
            return random.choices(rng,probability_list,k=1)[0]    
        hypothosized_sales=pd.Series(sales_per_period_probabilities).map(weighted_predictor)    
        return hypothosized_sales*season_mean_purchase


    def predict_total_sales_per_season(self,consumer_habits_dataframe,occurrence_multiplier,partition_column=None):
        """
        Input dataframe should have columns: frequency_of_purchase and total_previous_purchases, the season column to partition by, and a column of sales amounts
        if frequency_of_purchase or total_previous_purchases are left as None, they will default to the ones in the Consumer_Habits Dataset.
        frequency should be of:  'Every 3 Months', 'Annually', 'Quarterly', 'Monthly', 'Bi-Weekly', 'Fortnightly', 'Weekly'
        """
        df=consumer_habits_dataframe.copy()
        df=self.add_relevant_columns(df)
        if partition_column==None:
            partition_column=self.season_col_header_data_is_partitioned_by
        else: 
            self.season_col_header_data_is_partitioned_by=partition_column
        winter=self.simulate_sales_by_season(df,occurrence_multiplier,'Winter',365/4).sum() 
        spring=self.simulate_sales_by_season(df,occurrence_multiplier,'Spring',365/4).sum() 
        summer=self.simulate_sales_by_season(df,occurrence_multiplier,'Summer',365/4).sum() 
        fall=self.simulate_sales_by_season(df,occurrence_multiplier,'Fall',365/4).sum() 
        self.seasons = [winter,spring,summer,fall]

#------------------------------------------------------------------------------------------------------------------------------------
    def occurrence_per_season_data(self,df,occurrence_multiplier,period_to_aggrigate:str='Months'):
        """
        this takes the period to aggrigate: one of ['month','week','quarter','season']
        """
        #edge case where input might be plural
        if period_to_aggrigate[-1] in ('s','S'):
            period_to_aggrigate=period_to_aggrigate[:-1]
        #edge case to make period non-case sensitive
        period_to_aggrigate=period_to_aggrigate.lower()    
        #calculate period size    
        period_sizes = {'week':365/52,'month':365/12,'quarter':365/4,'season':365/4}
        period_size_in_days = period_sizes[period_to_aggrigate]
        #calculate numer of periods in each season: season_days/period_days
        periods_per_season= int( period_sizes['quarter'] / period_size_in_days )
        #call to get sums of predicted sales for given periods   
        if self.seasons is None:
            self.predict_total_sales_per_season(df,occurrence_multiplier)#                                          
        partitioned_seasons=[seasoni/periods_per_season for seasoni in self.seasons]
        return partitioned_seasons, period_to_aggrigate, period_size_in_days, periods_per_season
        

    # retrieve the predicted sales per period by season, name of the period such as week or month, the period size in days, and number of periods per season(quarter)
    def update_period_details(self,df,occurrence_multiplier,period:str='Months'):
        period=period.lower()
        if period in ['quarter','quarters','season','seasons','month','months','week','weeks']:
            if period[0]=='s':
                p='q'
            else: p=period[0]
        else:
            raise ValueError(f"Please enter one of: Months, Weeks, Quarters, or Seasons")
        self.period_details[p]=list(self.occurrence_per_season_data(df,occurrence_multiplier,period))
#------------------------------------------------------------------------------------------------------------------------------------
 
    def get_plot_data(self,df,occurrence_multiplier,period_:str='Month'):
        """

        """
        #4 seasons per year
        num_seasons=4

        #retrieve or create period data
        #get key
        period_=period_.lower()
        p=period_[0]
        if p=='s':p='q'
        #access info
        if self.period_details[p] is None:
            self.update_period_details(df,occurrence_multiplier,period_)
            res=self.period_details[p]
            partitioned_seasons, period_to_aggrigate, period_size_in_days, periods_per_season=res[0],res[1],res[2],res[3]
        else:
            res=self.period_details[p]
            partitioned_seasons, period_to_aggrigate, period_size_in_days, periods_per_season=res[0],res[1],res[2],res[3]  
            #iterate through seasons and periods within seasons
            #capture period totals and start points for later plot
            periods=[]
            starts=[]
            start=0
            for  seas in partitioned_seasons:
                for period in range(int(periods_per_season)):
                    if len(periods)<=0:
                        periods.append(seas)
                        starts.append(start)
                        start+=seas
                    else:
                        starts.append(start)
                        start+=seas
                        periods.append(seas)
            period_title=period_to_aggrigate.title()
            total_periods=int(num_seasons*periods_per_season)
            zero_to_max=(0,start)
            starts,periods=starts,periods
            self.period_details[p]+=[period_title,total_periods,zero_to_max,starts,periods]

#------------------------------------------------------------------------------------------------------------------------------------  
    def update_data_model_as_needed(self,df,occurrence_multiplier,period_:str='Month',    total_prev_purchases: str | None = None,
                                                                    freq_of_purchases: str | None = None,
                                                                    season_header_to_partition_sales_amounts: str | None = None,
                                                                    individual_sale_amounts: str | None = None,):
        """
        checks model objects for None and computes values as needed
        takes period_ as an argument and used defaults for the Consumer_Habit dataset internaly for seasonal forecasting
        """

        if total_prev_purchases is not None:
            self.total_prev_purchases_col_header=total_prev_purchases
        if freq_of_purchases is not None:
            self.frequency_of_purchase_col_header=freq_of_purchases
        if season_header_to_partition_sales_amounts is not None:
            self.season_col_header_data_is_partitioned_by=season_header_to_partition_sales_amounts
        if individual_sale_amounts is not None:
            self.individual_sale_purchase_amount=individual_sale_amounts

        if self.seasons is None:
            self.predict_total_sales_per_season(df,occurrence_multiplier)
        period_=period_.lower()
        p=period_[0]
        if p=='s':p='q'
        #access info
        try:
            if self.period_details[p][8] is not None:
                return
            else:self.get_plot_data(df,occurrence_multiplier,period_)
        except:
            self.get_plot_data(df,occurrence_multiplier,period_)

        
#------------------------------------------------------------------------------------------------------------------------------------

    def y_lims_ticks(self,y,interval=0.25):
        mn,mx=y[0],y[-1]
        spots=1
        places=0
        while spots<=mx:
            spots*=10
            places+=1
        spots//=10
        spots//=10
        mx+=spots
        mx//=spots
        mx*=spots
        whole_tick=spots*10
        ticks=[0]
        while ticks[-1]<mx:
            tick=ticks[-1]+whole_tick*interval
            ticks.append(tick)
        return (mn//spots,mx),ticks




#------------------------------------------------------------------------------------------------------------------------------------
    # CALLABLE-----------------------------------------------------------
    def floating_bar_plot(self,df,period_:str="Month",occurrence_multiplier:float=1.0,y_tick_aggregate_3rd_highest_nplace=0.5,  total_prev_purchases: str | None = None,
                                                                                                freq_of_purchases: str | None = None,
                                                                                                season_header_to_partition_sales_amounts: str | None = None,
                                                                                                individual_sale_amounts: str | None = None,
                                                                                                figure_figsize:tuple|None=None,
                                                                                                streamlit:bool=False,
                                                                                                auto_detect_height:bool=True):
        """
        returns a floating barplot that illustrates period over period sales forecasts based on the poisson distribution seasonal avg purchase amounts
        where y_tick_aggregate_3rd_highest_nplace sets the interval of ticks based on 3 place from left of int (such as decimal when in sci notation)
        where period_ is the seasonal period to plot: week, month, or season/quarter
        where occurrance_multiplier can be used to vary number of occurrences used in poisson.predict

        the imput data frame should contain the four base columns:  
            total_prev_purchases,
            freq_of_purchases,
            season_header_to_partition_sales_amounts,
            individual_sale_amounts
        the 4 base are input here as str(column headers) or  default to Consumer_Habits dataset column headers, but can be changed
        season_header_to_partition_sales_amounts should be of 'Winter', 'Spring', 'Summer', 'Fall'
        freq_of_purchases should be of 'Every 3 Months', 'Annually', 'Quarterly', 'Monthly', 'Bi-Weekly', 'Fortnightly', 'Weekly'
        figure_figsize should be of tuple(width:int,height:int) | None. 
        if auto_detect_height==True: autodetect will override figure_figsize[-1] if not None, if None, width will be set to 20 and height autodetected (min height = 4)
        
        each time it is called it stores period data as a class object, hence it will keep 'week', 'month', and 'season' in RAM
        """
        # a function to update stored data
        self.update_data_model_as_needed(df,occurrence_multiplier,period_,total_prev_purchases,freq_of_purchases,season_header_to_partition_sales_amounts,individual_sale_amounts)

        if period_[-1] in ['s','S']:
            period_=period_[:-1]
        #retrieve or create period data
        period_=period_.lower()
        if period_ not in ['month','week','quarter','season']:
            raise ValueError(f"Please enter one of: Month, Week, Quarter, or Season")
        p=period_[0]
        if p=='s':p='q'
        try:
            res=self.period_details[p]
            periods_per_season,period_title, total_periods, zero_to_max,starts,periods=res[3],res[4],res[5],res[6],res[7],res[8]
        except:
            self.get_plot_data(df,occurrence_multiplier,period_)
            res=self.period_details[p]
            periods_per_season,period_title, total_periods, zero_to_max,starts,periods=res[3],res[4],res[5],res[6],res[7],res[8]

        # x ticks
        if period_title!='Week':
            x_ticks = [f"{period_title} {i+1}" for i in range(len(periods))]
            xtick_fontsize=15
        else:
            x_ticks = [f"{period_title} {i+1}" if int((i+1)%13)==1 else str(i+1) for i in range(len(periods))]
            xtick_fontsize=8
        # calculate y ticks and y lim
        y_lim,y_ticks=self.y_lims_ticks(zero_to_max,y_tick_aggregate_3rd_highest_nplace)
        # y tick fontsize
        ytick_fontsize = 15
        # height
        if auto_detect_height==True:
            y_height_inches = ytick_fontsize / 72 * 1.6   # points → inches
            fig_height = max(4, (len(y_ticks) * y_height_inches)+2)
            if figure_figsize is None:
                figure_figsize=(20,fig_height)
            else:
                figure_figsize=(figure_figsize[0],fig_height)
        # set figsize
        if figure_figsize is not None:
            plt.figure(figsize=figure_figsize)
        # Set styles
        plt.rcParams['axes.formatter.limits'] = (-3, 3)   # force sci notation when needed
        plt.rcParams['axes.formatter.useoffset'] = True   # show offset text
        plt.rcParams['xtick.color'] = 'white'             # x-axis ticks
        plt.rcParams['ytick.color'] = 'white'             # y-axis ticks
        plt.rcParams['axes.labelcolor'] = 'white'         # axis labels
        plt.rcParams['axes.edgecolor'] = 'white'          # axis spines

        #use partitions and starts to form the plot
        sns.lineplot(x=x_ticks,y=np.array(starts)+np.array(periods),linewidth=1,label='Accumulated Total',color='red')
        sns.lineplot(x=x_ticks,y=periods,color='blue',linewidth=3,label=f"Per {period_title}")

        # as per output of predict_total_sales_per_season(df), seasons are ordered as winter, spring, summer, fall
        season_colors=['#EBE3CF','#1c68c0','#FDB131','#BC0B40']
        season_names=['Winter','Spring','Summer','Autumn']
        color_index=0
        increment_step=int(periods_per_season)
        for seas in range(0,total_periods,increment_step):
            sea=season_names[color_index]
            color=season_colors[color_index]
            color_index+=1
            plt.bar(x=x_ticks[seas:seas+increment_step], height=periods[seas:seas+increment_step], bottom=starts[seas:seas+increment_step],label=f"{sea+' '+period_title}ly Increments",color=color)
        plt.xticks(x_ticks,rotation=45,fontsize=xtick_fontsize,ha='right',color='white')
        plt.yticks(y_ticks,rotation=20,fontsize=ytick_fontsize,color='white')
        plt.title(f"Sales Forecast\nHypothetical Sales Per {period_title}\nBased on the Poisson Distribution and Average Per-Sale Amounts by Season",fontweight='bold',color='white')
        plt.ylim(y_lim)
        plt.gcf().set_facecolor("black")   # parchment background
        plt.gca().set_facecolor("black") 
        #format legend
        plt.rcParams['legend.fontsize']=15
        leg = plt.legend(frameon=False)
        for text in leg.get_texts():
            text.set_color("white")
        plt.grid(linewidth=.1)
        if streamlit==False:
            plt.show()
        else:
            fig = plt.gcf()
            st.pyplot(fig,clear_figure=True) 
        plt.rcdefaults()




    # CALLABLE-----------------------------------------------------------
    def plot_mean_seasonal_sales(self,df,period_:str="Month",occurrence_multiplier:float=1.0,  total_prev_purchases: str | None = None,
                                                                freq_of_purchases: str | None = None,
                                                                season_header_to_partition_sales_amounts: str | None = None,
                                                                individual_sale_amounts: str | None = None,
                                                                figure_figsize:tuple|None=(3,3),
                                                                streamlit:bool=False):
        """
        returns a pie chart that illustrates period sales forecasts based on the poisson distribution and seasonal avg purchase amounts
        where period_ is the seasonal period to plot: week, month, or season/quarter
        where occurrance_multiplier can be used to vary number of occurrences used in poisson.predict

        the imput data frame should contain the four base columns: 
            total_prev_purchases,
            freq_of_purchases,
            season_header_to_partition_sales_amounts,
            individual_sale_amounts
        the 4 base are input here as str(column headers) or default to Consumer_Habits dataset column headers, but can be changed
        season_header_to_partition_sales_amounts should be of 'Winter', 'Spring', 'Summer', 'Fall'
        freq_of_purchases should be of 'Every 3 Months', 'Annually', 'Quarterly', 'Monthly', 'Bi-Weekly', 'Fortnightly', 'Weekly'
        """ 
        # a function to update stored data
        self.update_data_model_as_needed(df,occurrence_multiplier,period_,total_prev_purchases,freq_of_purchases,season_header_to_partition_sales_amounts,individual_sale_amounts)

        period=period_.lower()
        if period in ['quarter','quarters','season','seasons','month','months','week','weeks']:
            if period[0]=='s': 
                p='q'
            else: p=period[0]
        else:
            raise ValueError(f"Please enter one of: Months, Weeks, Quarters, or Seasons")
        try:
            if p in ('q','s'):
                values=self.seasons
            else:
                values=self.period_details[p][0]
        except:
            self.period_details[p]=list(self.occurrence_per_season_data(df,occurrence_multiplier,period))
            values=self.period_details[p][0]
        labels = ['Winter','Spring','Summer','Fall']
        if figure_figsize is not None:
            plt.figure(figsize=figure_figsize)
        wedges,texts,autotexts = plt.pie(
            values,
            labels=[f"{labels[i]} (${values[i]:,.2f})" for i in range(len(values))],
            autopct='%1.1f%%',              
            startangle=120,               
            colors=['#EBE3CF','#1c68c0','#FDB131','#BC0B40'], 
            wedgeprops={'edgecolor':'black','linewidth':2},
            explode=[0.04]*4,
            shadow=True
        )
        for text in texts:
            text.set_color('white')
        title_=f"{period_.title()} According to Season" if not period_.title().startswith('Sea') and not period_.title().startswith('Qua') else period_.title()
        plt.title(f"Forecasted Sales Per {title_}",
                fontsize=16, fontweight='bold',color='white')
        plt.gcf().set_facecolor('black')
        if streamlit==False:
            plt.show()
        else:
            fig = plt.gcf()
            st.pyplot(fig,clear_figure=True)
        plt.rcdefaults()




