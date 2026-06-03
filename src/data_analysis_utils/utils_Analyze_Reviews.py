import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# a function for plotting pareto charts
def plot_cat_to_review_levels(data,
                              category_column,
                              review_column,
                              probabilities:bool,
                              detail_plots:bool,
                              n_axis_columns,
                              dist_overview:bool,
                              detail_plot_probabilities:bool):
    """
    where category column is a categorical columns such as 'Item Purchased'
    review_colunm is a review rating level such as Review Rating level 1, 2, or 3
    probabilities is boolean such as count/pareto charts vs probabilities
    detail_plots T/F determines if detail plots are wanted
    n_axis_columns is used if detail_plots is True. It is the number of plots to fit on one row when filling the figure
    dist_overview True proceeds other plots with a distribution of each category
    """
    if dist_overview==True:
        plt.figure(figsize=(20,3.5))
        plt.title(f"Overview Distribution Of {category_column}")
        count_data=data[category_column].value_counts()
        sns.barplot(x=count_data.values,y=count_data.index,orient='h')
        plt.show()

    if probabilities==False and detail_plots==False and dist_overview==False:
        raise ValueError('one of (probabilities, detail_plots, dist_overview) needs to be True or there is nothing to return')
    # aggrigate and get probabilities
    grouped = data.groupby([category_column, review_column],as_index=False, observed=True).size().rename(columns={'size':'Count'})
    grouped[review_column] = grouped[review_column].astype(int)

    # get probabilites if True
    if probabilities==True:
        grouped['ReviewCount'] = grouped.groupby(review_column).transform('size')
        grouped['CategoryCount'] = grouped.groupby(category_column).transform('size')
        grouped["P(Rev|Cat)"]  = grouped['Count'] / grouped['ReviewCount'] 
        grouped["P(Cat|Rev)"] = grouped['Count'] / grouped['CategoryCount'] 
        grouped = grouped.drop(columns=['ReviewCount','CategoryCount'])

        fig = plt.figure(figsize=(20,5))
        plt.title(f'Probability of Review Level Given {category_column}')
        sns.barplot(data=grouped, x=category_column, y="P(Rev|Cat)", 
                    hue=review_column, hue_order=sorted(list(data[review_column].unique())), legend='auto')
    
    if detail_plots==True:
        num_axises = data[review_column].nunique()
        cols=n_axis_columns
        rows=int(np.ceil(num_axises/cols))
        fig = plt.figure(figsize=(20,(rows*3)+3))
        if detail_plot_probabilities==True:
            plt.suptitle(f"Probabilities of Review Level given Category and Category given Review Level\nWhere 1 is Lowest Review Rating Level\nDistributed Across {category_column}\n\n",fontsize=20)
        else:
            plt.suptitle(f"Pareto Plots Per Level of Review Rating\nWhere 1 is Lowest Rating Level and {data[review_column].max()} is Highest\nDistributed Across {category_column}\n\n",fontsize=20)
        for level in range(1,num_axises+1):
            if detail_plot_probabilities==True:
                plot_data=grouped.loc[grouped[review_column]==int(level)].sort_values(by="P(Cat|Rev)",ascending=False)
                plot_data = plot_data.melt( id_vars=[category_column], value_vars=[ "P(Rev|Cat)", "P(Cat|Rev)" ],var_name='Probability_Type', value_name='Probability' )
                ax = plt.subplot(rows,cols,level)
                ax.set_title(f"Review Rating Level {level}")
                legend='auto' if level in (1,3,5) else False
                sns.barplot( data=plot_data, x=category_column, y='Probability', hue='Probability_Type', ax=ax , legend=legend)
                ax.tick_params(axis='x', rotation=45)
                ax.set_ylabel('Probability')
                ax.set_xlabel(f"P's accross lvl {level}")
            else:
                plot_data=grouped.loc[grouped[review_column]==int(level)].sort_values(by='Count',ascending=False)
                plot_data['Percent']=((plot_data['Count'].cumsum()) / plot_data['Count'].sum())*100
                ax = plt.subplot(rows,cols,level)
                ax.set_title(f"Review Rating Level {level}")
                x = np.arange(len(plot_data))

                ax.bar(x, plot_data['Count'])
                ax.set_xticks(x)
                ax.set_xticklabels(plot_data[category_column], rotation=45, ha='right')
                ax.set_ylabel('Count')
                ax.set_xlabel(f"Counts Per Category in Level {level}")
                ax2 = ax.twinx()
                ax2.plot(x, plot_data['Percent'], marker='o',color='orange',linewidth=2.5)
                ax2.set_ylim(0, 100)
                ax2.set_ylabel('Cumulative Percent')

        plt.tight_layout()
        plt.show()