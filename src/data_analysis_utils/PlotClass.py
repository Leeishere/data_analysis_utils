
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
try:
    import streamlit as st
except:
    import warnings
    warnings.warn('streamlit import unsuccessful')


class PlotClass:


    # =====================================================================================================================================
    # Helper functions to fit pots to figures. Especially verticle bar plots
    # =====================================================================================================================================

    # -----------------------------------------------------------------------------------------------
    # axis sizes measured in column widths  
    #a func to return a map list for fitting to the plot, and and a dict for plotting 
    #plotting functions
    def _bar_count(self,
                data, 
                column_headers:list|tuple|None=None, 
                max_bars_on_row:int=40, 
                alternate_plot_when_max_bars_is_exceeded:bool=False,
                num_columns_per_row:int=3,
                univariate:bool=False): 
        """
        uses nunique() to determing num bars 
        column_headers should be an arraylike of nested arraylike: [(header_a, header_b, y_header), ...] or [(univariate, y_header), ...] or [(univariate), ...] or ["univariate", ...]
        where y_header is the y axis and not not used here, but passed into this funcs output dict.
        max_bars_on_row is total of all bars that fit on the row, 
        max_columns_per_row is the numver of spots reserved on the row columnwise
        univariat is default false, but can be set True ** if True, then only one column will be used to count num of vertical bars
        returns 
        a list of axis counts such as [index 0 is count of 1 column wide axises, index 1 is 2 column wide ...
        and dict {str(int(num_cols_wide)):[(header,header, y_header),(),...]}
        Plus a 3rd element is returned: a separate dict of alternates: {str(int(num_cols_wide)):[(header,header, y_header),(),...]}
            which is empty if alternate_plot_when_max_bars_is_exceeded is false, or if true but bax bars is never exceeded
        """
        if not column_headers:
            column_headers = []
        # count each axis size in column width
        axis_counts_by_size = [0]*(num_columns_per_row)
        # store column header(s) with axis width in columns as key
        columns_per_axis_dict={}
        alternate_plots={}
        # bars per columns
        vert_bars_per_axis=int(max_bars_on_row//num_columns_per_row)  # the number of bars per axis slot
        # check to see if headers are passed as univariate strings not enclosed in nested arrays and otherwise, len nested is >1
        len_uni=False
        if column_headers:
            if isinstance(column_headers[0],str) or ( (not isinstance(column_headers[0],str)) and (len(column_headers[0])<=1) ):
                if univariate==False:
                    raise ValueError('Variable(s) are not combined in nested arraylike structrures, but univariate==False.',ValueError)
                if isinstance(column_headers[0],str):
                    column_headers=[[col] for col in column_headers]
                len_uni=True

        #loop through headers
        for column_headers_ in column_headers:
            # flag for when alternate plots are used in case of too many bars: alternate_plot_when_max_bars_is_exceeded==True
            alternate_plot=False
            #get n bars
            if univariate==False:
                n_bars = data[column_headers_[0]].nunique() * data[column_headers_[1]].nunique()
                if (n_bars>max_bars_on_row) and (alternate_plot_when_max_bars_is_exceeded==True):
                    n_bars = max(data[column_headers_[0]].nunique(),data[column_headers_[1]].nunique())
                    alternate_plot=True
            else: 
                if len_uni==False: # make sure there is a second header in nested array
                    n_bars = data[column_headers_[1]].nunique() # This is because _get_columns_from_df() puts numeric headers in column [0] by default. Hence, case of numcat, categorical would be in [1]
                else:
                    n_bars = data[column_headers_[0]].nunique() #

            #get num_columns_wide of plot
            n_cols_wide=int((n_bars//vert_bars_per_axis)+1) if (n_bars%vert_bars_per_axis)>0 else int(n_bars//vert_bars_per_axis)

            #update axis_counts_by_size list and columns_per_axis_dict dict
            if n_cols_wide>num_columns_per_row:
                axis_counts_by_size[-1]+=1   # update the index at the end of the list to count a full row sized chart
                if alternate_plot==True:
                    if str(num_columns_per_row) not in alternate_plots.keys():
                        alternate_plots[str(num_columns_per_row)]=[column_headers_]
                    else: alternate_plots[str(num_columns_per_row)].append(column_headers_)
                else:
                    if str(num_columns_per_row) not in columns_per_axis_dict.keys():
                        columns_per_axis_dict[str(num_columns_per_row)]=[column_headers_]
                    else: columns_per_axis_dict[str(num_columns_per_row)].append(column_headers_) 
            else:
                axis_counts_by_size[n_cols_wide-1]+=1   # update the index sized chart position (index 0 = 1 col, index 1 = 2
                if alternate_plot==True:
                    if str(n_cols_wide) not in alternate_plots.keys():
                        alternate_plots[str(n_cols_wide)]=[column_headers_]
                    else: alternate_plots[str(n_cols_wide)].append(column_headers_) 
                else:
                    if str(n_cols_wide) not in columns_per_axis_dict.keys():
                        columns_per_axis_dict[str(n_cols_wide)]=[column_headers_]
                    else: columns_per_axis_dict[str(n_cols_wide)].append(column_headers_)   
        return axis_counts_by_size, columns_per_axis_dict, alternate_plots    # returns a count list and lookup dict(s)





    #-------------------------------------------------------------------------------------------------------------------------------------------
    # a helper function for finding combinations
    def _one_rotation(self,
                      curr_available_space_in_row, curr_grid, row_map): 
        """
        this is used inside _get_all_satisfying_combinations() to execute rolls
        parameters are updated and returned after a roll, or returned as val,None,None, if no rolls left
        where row map is a countdown from max_iter to 0 for each row [calculated at creation of any new grid and passed untill curr_grid is replaced]
        curr_grid is input grid of remainding combo possibilites [where it is recalculated outside this func whenever possibilities are removed]
        curr_available_space_in_row is the number of spaces to fill [where it is decrimented outside this func when all possibilities are exausted]
        """
        max_rolls=curr_grid.shape[1]

        #update row_map and determine which row to roll
        row_to_iterate=-1   #start at right side
        while row_map[row_to_iterate]==1 and ((row_to_iterate+len(row_map))>0): # a loop for when a row is completed -> loop moves action to the next row that needs to be rolled
            row_map[row_to_iterate]=max_rolls
            if row_to_iterate==-1:
                curr_grid[row_to_iterate:,:]=np.roll(curr_grid[row_to_iterate:,:], 1, axis=1)
            elif row_to_iterate<-1:
                curr_grid[row_to_iterate:row_to_iterate+1,:]=np.roll(curr_grid[row_to_iterate:row_to_iterate+1,:], 1, axis=1)
            row_to_iterate-=1

        # a check to see if all combos have been considered
        if (row_to_iterate+len(row_map))==0:
            return curr_available_space_in_row, None, None  ##------------------> all combos have been considered, to curr_available_space_in_row needs to be decrimented and the grid and map need to be reconstructed
        
        if row_to_iterate==-1:
            row_map[row_to_iterate]-=1
            curr_grid[row_to_iterate:,:]=np.roll(curr_grid[row_to_iterate:,:], 1, axis=1)
        elif row_to_iterate<-1:
            row_map[row_to_iterate]-=1
            curr_grid[row_to_iterate:row_to_iterate+1,:]=np.roll(curr_grid[row_to_iterate:row_to_iterate+1,:], 1, axis=1)

        return curr_available_space_in_row, curr_grid, row_map



    #-------------------------------------------------------------------------------------------------------------------------------------------
    #a helper that finds the max plots that can be on a row based on available plots and their sizes
    def _get_max_num_plots_on_a_row(self,
                                    axis_counts_by_size):
        """
        this is used inside greedy_plot_placemen() with _get_all_satisfying_combinations()
        takes output[0] from _bar_count() as input, but in use, the output[0] is mutated in between
        #it returns a value based on available plots -> (lowest lenghts accumulate while <= len row and > len row not included) not including 0
        where axis counts by size are (index+1)=axis_width in columns it consumes
        num_columns_per_row is self explanitory
        """
        axis_map=axis_counts_by_size.copy()
        plot_counter=0
        column_widths_remainding=len(axis_map)
        pointer=0
        while (column_widths_remainding>0) and (pointer < len(axis_map)):
            # if there are plot(s) that size and room for them
            if axis_map[pointer]>0 and ((column_widths_remainding-(pointer+1))>=0):
                axis_map[pointer]-=1
                plot_counter+=1
                column_widths_remainding-=(pointer+1)
                continue
            # plots from this point and forward are too big, so break
            elif ((column_widths_remainding-(pointer+1))<0):
                break
            # there are no plots at this size
            else:
                pointer+=1
        return plot_counter



    #-------------------------------------------------------------------------------------------------------------------------------------------
    # a function that returns satifying combinations for the present state: n plots & n spots
    def _get_all_satisfying_combinations(self,
                                         curr_available_space_in_row:int,
                                        curr_grid,
                                        maximum_number_of_plots_can_be_used_to_fill_a_row:int,
                                        result_updates_between_drop_duplicates:int=10000):
        """
        this is used in _get_max_num_plots_on_a_row() in cooperation with a greedy approach
        where:
        curr_available_space_in_row is the number of gridspec spaces to fill in row
        curr_grid is first passed as a 1d array of unique gridspec lenghts in plot list + 0 (if 0 is not present, it will be added). Rows are added inside the function based on gridspec-row lenght and the new formed grid is iterated on rowwise to capture all combinations that sum to curr_available_space_in_row
        maximum_number_of_plots_can_be_used_to_fill_a_row is the maximum number of plots that can fill a row. it is based on available plots -> (lowest lenghts accumulate until<= len row) not including 0
        result_updates_between_drop_duplicates is an updater that duplicates some drop_duplicate time complexity, but reduces RAM consumption
        """
        if (curr_available_space_in_row>1)==False:
            raise ValueError(f"Current available spots on the row are <2. No combination can satisfy.")
        elif (curr_available_space_in_row>7)==True:
            raise ValueError(f"Unable to place more than 6 columns in one row. Doing so can consume excessive resources.")
        curr_grid=np.asarray(curr_grid)
        if curr_grid.ndim != 1:
            raise ValueError(f"The grid is the wrong shape. 1d array expected, but recieved {curr_grid.ndim}d")
        
        #ensure there are non plot slots (0's) in the combos to allow large plots to fill in
        if 0 not in curr_grid:
            curr_grid=np.hstack((curr_grid,0))

        #add rows until they == maximum_number_of_plots_can_be_used_to_fill_a_row
        temp_grid=curr_grid.copy()
        while curr_grid.shape[0]<maximum_number_of_plots_can_be_used_to_fill_a_row or curr_grid.ndim==1:
            curr_grid=np.vstack((temp_grid,curr_grid))
        # create the map _one_rotation() will use to track each row's iterations (rolls)
        row_map=[len(curr_grid)]*curr_grid.shape[0]# <----- row_map tracks iterations(rolls) for each vecor(row) 

        # collect combinations that satisfy the conditon->(sum(col) == curr_available_space_in_row)
        possible=[]
        count=0# used with result_updates_between_drop_duplicates
        while curr_grid is not None: #_one_rotation() will return curr_grid, row_map = None,None when all iterations (rolls) are complete
            if (curr_grid.sum(axis=0)==curr_available_space_in_row).any():#the condition is met ->(sum(col) == curr_available_space_in_row)
                count+=1
                mask=curr_grid.sum(axis=0)==curr_available_space_in_row
                pos=[sorted(curr_grid[:,mask][:,i].ravel())[::-1] for i in range(curr_grid[:,mask].shape[1])] # sort the combination and make a
                if len(possible)>0:
                    possible=np.vstack((possible,pos)) 
                else:
                    possible=pos
                    possible=np.array(possible)
                if (count%result_updates_between_drop_duplicates)==0 and len(pos)>1:
                    possible=np.unique(possible,axis=0)
            #update curr_available_space_in_row,curr_grid,row_map
            curr_available_space_in_row,curr_grid,row_map = self._one_rotation(curr_available_space_in_row,curr_grid,row_map)
        possible=np.unique(possible,axis=0)
        if (possible.ndim==2) and (possible.shape[1] >= 1):
            #https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html
            sort_index=np.lexsort(tuple([possible[:, i] for i in range(possible.shape[1]-1,-1,-1)]))
            sort_index=sort_index[::-1]
            possible=possible[sort_index]
        return possible






    #-------------------------------------------------------------------------------------------------------------------------------------------

    # a hybrid of greedy approach and optimal combinations that processes output[0] from _bar_count()
    def _greedy_plot_placement(self,
                               axis_counts_by_size:list):
        """
        """
        max_combo_computable=6  # for a call to _get_all_satisfying_combinations 6 is ussually ok, 7 can be slow
        all_rows = []
        curr_row = []
        curr_sum = 0
        num_columns_per_row=len(axis_counts_by_size)
        i =  num_columns_per_row -1   # start from largest size
        while sum(axis_counts_by_size) > 0:
            #print('map at outer iter: ', axis_counts_by_size)
            """if (4+3)==7:
                prnt=True
            if prnt==True:
                print('new iter')
                print('loop',axis_counts_by_size)
                print('sum',curr_sum)"""
            # if index out of bounds → finalize row and reset
            if (i < 0) or (curr_sum==num_columns_per_row):
                """if prnt==True:
                    print('first if updater')"""
                if curr_row:
                    all_rows.append(curr_row)
                    #print('all i<0 or match: ',all_rows)
                curr_row = []
                curr_sum = 0
                i = len(axis_counts_by_size) - 1
                continue
            # if no pieces of this size → move down
            if axis_counts_by_size[i] == 0:
                """if prnt==True:
                    print('first if decrimentor')"""
                i -= 1
                continue
            # the size to append
            size = i + 1
            # fits exactly → close row
            if (curr_sum + size) == num_columns_per_row:            
                """if prnt==True:
                    print('exact')"""
                curr_row.append(size)
                axis_counts_by_size[i] -= 1
                all_rows.append(curr_row)
                #print('all rows at exact greedy match: ',all_rows)
                # reset row
                curr_row = []
                curr_sum = 0
                i = len(axis_counts_by_size) - 1
                continue
            # try to grab the best fill options without too much more compute
            # uses max_combo_computable to determine if calling the combo function is too compute intensive or not
            elif ((num_columns_per_row-curr_sum)<=max_combo_computable) and ((num_columns_per_row-curr_sum)>1):            
                """if prnt==True:
                    print('combo')
                    print('sum: ',curr_sum,'cols per row: ',num_columns_per_row)"""
                num_cols_to_fill=num_columns_per_row-curr_sum
                num_cols_to_fill_decriment_variable=num_cols_to_fill
                while (num_cols_to_fill_decriment_variable>1) and (sum(axis_counts_by_size[:num_cols_to_fill_decriment_variable])>0):
                    array_of_sizes=[value for value in range(1,num_cols_to_fill_decriment_variable+1) if axis_counts_by_size[value-1]!=0]
                    abs_max_possible_combo_size = self._get_max_num_plots_on_a_row(axis_counts_by_size[:num_cols_to_fill_decriment_variable])
                    possible_combos = self._get_all_satisfying_combinations(num_cols_to_fill_decriment_variable,
                                                                    array_of_sizes,
                                                                    abs_max_possible_combo_size,
                                                                    result_updates_between_drop_duplicates=10000)
                    if possible_combos.shape==(0,):
                        num_cols_to_fill_decriment_variable-=1
                        continue
                    else:
                        """print('before'.upper())
                        print('possible combos', possible_combos, 'shape',possible_combos.shape)
                        print('num cols targeting: ',num_cols_to_fill_decriment_variable)
                        print('real map: ',axis_counts_by_size)"""
                        #iterate through the first match
                        temp_map=axis_counts_by_size.copy()
                        row_has_been_identified=False
                        # start from row 0 in order to get the greediest answer, such as the largest plots first
                        for row in range(possible_combos.shape[0]):
                            # possible row
                            row_fill = possible_combos[row,:]
                            # check possibility
                            row_works=True
                            #print('row fill',row_fill,'temp_map',temp_map)
                            for val in row_fill:
                                # condition that checks availability and disqualifies rows that need too many of any one plotsize 
                                if (val!=0) and (temp_map[val-1]==0):
                                    row_works=False
                                    break
                                else:
                                    temp_map[val-1]-=1
                            # the row hasn't been disqualified, so it's the greediest option 
                            if row_works==True:
                                for val in row_fill:
                                    if (val!=0):
                                        curr_row.append(int(val))                        
                                        axis_counts_by_size[val-1]-=1
                                        curr_sum+=val                            
                                #if curr_rows:  Removed because of reduncancy
                                # append the greedy solution and reset for next iteration
                                all_rows.append(curr_row)
                                #print('all rows at combos: ',all_rows)
                                # reset row
                                curr_row = []
                                curr_sum = 0
                                i = len(axis_counts_by_size) - 1
                                row_has_been_identified=True
                                break
                            else:
                                # try the next combo
                                temp_map=axis_counts_by_size.copy()
                        # break if a row has been found
                        if row_has_been_identified==True:
                            break
                        else:
                            # if no row found, then look for one fewer columns to satisfy combinations
                            num_cols_to_fill_decriment_variable-=1
                            continue
                        
                        """print('after'.upper())
                        print('possible combos', possible_combos, 'shape',possible_combos.shape)
                        print('num cols targeting: ',num_cols_to_fill_decriment_variable)
                        print('real map: ',axis_counts_by_size)"""
                # the while conditions aren't met
                # so the rest of the outer loop is implemented here inside this inner elif: while loop to avoid complicated conditional logic
                # fits partially
                else:
                    if ((curr_sum + size) < num_columns_per_row):
                        curr_row.append(size)
                        axis_counts_by_size[i] -= 1
                        curr_sum += size
                        continue
                    # too large → try smaller
                    # will eventually reset curr_row when i<0
                    else:
                        i -= 1
                continue

            # fits partially
            elif ((curr_sum + size) < num_columns_per_row):
                curr_row.append(size)
                axis_counts_by_size[i] -= 1
                curr_sum += size
                continue
            # too large → try smaller
            # will eventually reset curr_row when i<0
            else:
                i -= 1
        if curr_row:
            all_rows.append(curr_row)
            #print('all rows at final if: ',all_rows)
        return all_rows





    # --------------------------------------------------------------------------------------------------------------

    # function to map columns to rows
    def _columns_in_rows_map(self,
                             df, 
                            columns,
                            max_bars_on_row:int=40, 
                            alternate_plot_when_max_bars_is_exceeded:bool=False,
                            num_columns_per_row:int=6, 
                            univariate:bool=False):
        """
        Where at least one column is presumed categorical
        where df is s pandas DataFrame
        columns is the columns such as [(col_a,col_b),(col_r,col_k), ...] such as output from _get_columns_from_df() or there can be more than 2 cols in groups, but only the first two are considered when fitting to plot based on number of verticle bars
        where columns are considered as col[1] on X axis and col[0] on Y by default. This is because _get_columns_from_df() puts numeric headers in column [0] by default. Hence, case of numcat, categorical would be in [1]
        exception: in case of univariate, cols can be passed as ["col", ...] or [(col), ...]. otherwise if (col1,col2), col2 will be considered X axis
        max_bars_on_row is the ideal max sum of vertical bars in barplots (in case of hue) or sum of x ticks. Such as to summ all xticks or bars on one row irrespective of y ticks and labels.
        num_columns_per_row: the number of [plot_axis]columns allocated for each row. It is presumed some will share plot axises.
        univariate False is bivariate, true is univariate. It determines the number of x axis ticks or bars. Such as variable_1.nunique()*variable_2.nunique(). If True, only one column will be used to set number of x axis bars
        Suggested Uses:
            univariate:   count plots , 
                        pre-binned histplots
            bivariate:    barplot w/hue, 
                        barplot stacked,
                        boxen plot,
                        box plot
        """
        axis_counts_by_size,header_dict,alternate_plots=self._bar_count(df, 
                                                columns, 
                                                max_bars_on_row=max_bars_on_row, 
                                                alternate_plot_when_max_bars_is_exceeded=alternate_plot_when_max_bars_is_exceeded,
                                                num_columns_per_row=num_columns_per_row, 
                                                univariate=univariate)
        map_cols_to_rows = self._greedy_plot_placement(axis_counts_by_size)

        return map_cols_to_rows, header_dict, alternate_plots

    # --------------------------------------------------------------------------------------------------------------
    # a function that takes output from CompareColumns Hyp-tests_&_Coeffients

    def _get_columns_from_df(self,
                             dataframe:pd.DataFrame,pairs:bool=True):
        """
        where dataframe can be output from statistic analysis dataframes produced by CompareColumns
        where statistic columns [0,1] are column headers of columns that have been statisticaly analyzed
        if pairs True the first 2 columns are taken
            output is a list of header pairs: [(header_a,header_b),(header_a,header_c), ...]
        if pairs False the first column is taken
            output is a list of headers: [(header_a),(header_c), ...]
        """
        if pairs==True:
            return dataframe[dataframe.columns[:2]].to_numpy().tolist()
        else:
            return dataframe[dataframe.columns[:1]].to_numpy().tolist()




    # =====================================================================================================================================
    # bivariate_categorical_snapshot
    # =====================================================================================================================================




    def bivariate_categorical_snapshot(self,
                            data:pd.DataFrame,
                            column_combinations:list,                        
                            n_wide:int|tuple|list,
                            stacked_bars_when_max_bars_is_exceeded:bool=True,
                            sorted:bool=True,
                            super_title:str|None="Bivariate Analysis of Categoric and Categoric" ,
                   streamlit_:bool|None = None):
        """
        categorical should be a list of column pairs: [[cola,colb],[colc,cold], ...]
        where n_wide determines horizontal or vertical bars
            if type(n_wide)==int, horizontal bars and n_wide and indicates n cols of plots, 
            otherwise vertical bars and n_wide is a list or tuple where 
                n_wide[0]=n cols on row, # number of plot axises available per row
                and n_wide[1]=ideal_max bars on row, 
                and optional <n_wide[2]=row_height or default is 5>
        where stacked_bars_when_max_bars_is_exceeded==True uses stacked barcharts for cases when there would otherwise be too many bars
        where sorted sorts x and hue if set to True

        """        
        plt.rcdefaults()
        if streamlit_ is None:
            streamlit_ = False 
        if isinstance(n_wide,int):  # case where plots should have horizontal bars.
            raise ValueError("Sorry horizontal bars are not yet supported. Please provide an array like input for n_wide")
        else:                       # case where plots should have vertical bars.
            row_height=5 if (len(n_wide)<3) else n_wide[2]
            map_cols_to_rows, header_dict, stacked_plots = self._columns_in_rows_map(data, 
                                                                column_combinations, 
                                                                max_bars_on_row=n_wide[1], 
                                                                alternate_plot_when_max_bars_is_exceeded=stacked_bars_when_max_bars_is_exceeded,
                                                                num_columns_per_row=n_wide[0], 
                                                                univariate=False)
            

        #create figure
        num_rows=max(1,len(map_cols_to_rows))
        plot_height=(row_height*num_rows)+(2 if num_rows <=4 else 0)
        fig = plt.figure(figsize=(20,plot_height))#,constrained_layout=True)
        # create gridspec
        gs = GridSpec(
                    nrows=num_rows,
                    ncols=n_wide[0],
                    height_ratios=None,#[1 for i in range(num_rows)], 
                    hspace=0.75,
                    wspace=0.45
                )
        # super title
        if map_cols_to_rows: 
            plt.suptitle(super_title+'\n\n',fontsize=25)
        else:
            plt.suptitle("Nothing to Plot\n"+super_title,fontsize=12)
        # fill rows
        curr_row=0
        available_row_start_index = 0
        for index in range(len(map_cols_to_rows)):
            curr_row=index
            row_map = map_cols_to_rows[index]
            # fill columns on each row
            for plot_width in row_map:
                # fit axis for plot
                ax = fig.add_subplot(gs[curr_row, available_row_start_index : available_row_start_index+plot_width])
                # start position
                available_row_start_index+=plot_width
                # get plot data
                # flag for plot creation type
                stacked=False
                # if there are plots in the alternate plot type dict and if this size is not in the primary plot type dict
                if  ( (stacked_bars_when_max_bars_is_exceeded==True) and (stacked_plots) ) and ( (str(plot_width) not in header_dict.keys()) or (not header_dict[str(plot_width)]) ):
                    plot_headers = stacked_plots[str(plot_width)].pop()
                    #flage for plot creation type
                    stacked=True
                # esle the primary plot type dict
                else:
                    plot_headers = header_dict[str(plot_width)].pop()
                
                if data[plot_headers[0]].nunique()>data[plot_headers[1]].nunique():
                    primary_col,secondary_col=plot_headers[0],plot_headers[1]
                else:                
                    primary_col,secondary_col=plot_headers[1],plot_headers[0]
                if sorted==True:
                    primary_order, secondary_order = data[primary_col].value_counts().index , data[secondary_col].value_counts().index
                else:
                    primary_order, secondary_order = None , None
                # plot
                plot_title = primary_col.title()+'\n'+secondary_col.title()
                plt.title(plot_title)
                # if this is a secondary plot type
                if stacked==True:
                    plot_data=data[[primary_col,secondary_col]].groupby([primary_col,secondary_col],as_index=False,observed=True).size()
                    plot_data=plot_data.set_index([primary_col,secondary_col]).unstack()
                    plot_data.columns=[i[1] for i in plot_data.columns]
                    if sorted==True:
                        plot_data.loc[primary_order, secondary_order].plot(kind='bar',stacked=True,ax=ax)
                    else:
                        plot_data.plot(kind='bar',stacked=True,ax=ax)
                # if it's a primary plot type
                else:
                    sns.countplot(data=data, x=primary_col,hue=secondary_col,order=primary_order,hue_order=secondary_order,ax=ax)
                #ax.legend(frameon=False)
                
                # add a draggable legend
                leg = ax.get_legend()
                if leg:
                    leg.set_draggable(True)
                    leg.get_frame().set_alpha(0.0)
                
                ax.tick_params(axis='x',rotation=55)
                for label in ax.get_xticklabels():
                    label.set_ha('right') 
                y_label='Count' 
                ax.set_ylabel(y_label)
                if stacked==False:
                    plt.grid()
            curr_row+=1
            available_row_start_index = 0
        #plt.tight_layout()
        if streamlit_==False:
            plt.show()
        else:
            fig = plt.gcf()
            st.pyplot(fig,clear_figure=True)  
        plt.rcdefaults()
        return


    # =====================================================================================================================================
    # numeric_to_categorical_snapshot
    # =====================================================================================================================================



    def numeric_to_categorical_snapshot(self,
                                        data:pd.DataFrame,
                                        column_combos:list,
                                        plot_type:str='boxen',
                                        n_wide:int|tuple|list=(6,40,4),
                                        super_title:str|None="Bivariate Analysis of Categoric and Numeric" ,
                   streamlit_:bool|None = None):
        """
        column_combos should be a list of column pairs: [[numeric_col,categoric_col],[ ..., ... ], ...]
        plot_type should be one of box, boxen, violin
        where n_wide determines horizontal or vertical bars
            if type(n_wide)==int, horizontal bars and n_wide and indicates n rows, 
            otherwise vertical bars and n_wide is a list or tuple where n_wide[0]=n cols on row, and n_wide[1]=ideal_max bars on row, <n_wide[2]=row_height or default is 5>

        """
        plt.rcdefaults()
        if streamlit_ is None:
            streamlit_ = False 
        if isinstance(n_wide,int):  # case where plots should have horizontal bars.
            raise ValueError("Sorry horizontal bars are not yet supported. Please provide an array like input for n_wide")
        else:                       # case where plots should have vertical bars.
            row_height=5 if (len(n_wide)<3) else n_wide[2]
            # where alternate plots is unused
            map_cols_to_rows, header_dict, alternate_plots = self._columns_in_rows_map(data, 
                                                                column_combos, 
                                                                max_bars_on_row=n_wide[1], 
                                                                num_columns_per_row=n_wide[0], 
                                                            univariate=True)  # univariate True because only cat cols are counted as n_bars
        #create figure
        num_rows=max(1,len(map_cols_to_rows))
        plot_height=(row_height*num_rows)+(2 if num_rows <=4 else 0)
        fig = plt.figure(figsize=(20,plot_height))#,constrained_layout=True)
        # create gridspec
        gs = GridSpec(
                    nrows=num_rows,
                    ncols=n_wide[0],
                    height_ratios=None,#[1 for i in range(num_rows)], 
                    hspace=0.6,
                    wspace=0.6
                )
        # super title
        if map_cols_to_rows: 
            plt.suptitle(super_title+'\n\n',fontsize=25)
        else:
            plt.suptitle("Nothing to Plot\n"+super_title,fontsize=12)
        # adjust suptitle position to avoid excessive white space issues
        fig.subplots_adjust(top=0.92)
        # fill rows
        curr_row=0
        available_row_start_index = 0
        for index in range(len(map_cols_to_rows)):
            curr_row=index
            row_map = map_cols_to_rows[index]
            if row_map:
                # fill columns on each row
                for plot_width in row_map:
                    # fit axis for plot
                    ax = fig.add_subplot(gs[curr_row, available_row_start_index : available_row_start_index+plot_width])
                    # start position
                    available_row_start_index+=plot_width
                    # get plot data
                    plot_headers = header_dict[str(plot_width)].pop()
                    num_col, cat_col = plot_headers[0], plot_headers[1]
                    plot_title = num_col.title()+'\n'+cat_col.title()
                    plt.title(plot_title)
                    if plot_type=='box':
                        sns.boxplot(data=data,x=cat_col,y=num_col,ax=ax)
                    elif plot_type=='boxen':
                        sns.boxenplot(data=data,x=cat_col,y=num_col,ax=ax)
                    elif plot_type=='violin':
                        sns.violinplot(data=data,x=cat_col,y=num_col,ax=ax)
                    else:
                        raise ValueError("plot_type must be one of box, boxen, violin")
                    ax.tick_params(axis='x',rotation=45)
                    for label in ax.get_xticklabels():
                        label.set_ha('right') 
                    y_label=num_col
                    ax.set_ylabel(y_label)
                    if len(ax.get_xticks()) <= 60:
                        plt.grid()
            available_row_start_index = 0
        #plt.tight_layout()
        if streamlit_==False:
            plt.show()
        else:
            fig = plt.gcf()
            st.pyplot(fig,clear_figure=True)  
        plt.rcdefaults()
        return

    # =====================================================================================================================================
    # bivariate_numeric_numeric_snapshot
    # =====================================================================================================================================




    def _jointplot_gridspec(self,
        x,
        y,
        linreg:bool=False,
        fig=None,
        gridspec=None,
        scatter_kwargs=None,
        xhist_kwargs=None,
        yhist_kwargs=None,
        linreg_kwargs=None,
        ratio=4,
        space=0.05
    ):
        """
        Create a joint plot layout using matplotlib GridSpec.

        Parameters
        ----------
        x, y : array-like
            Data to plot.
        fig : matplotlib.figure.Figure, optional
            Existing figure. If None, a new one is created.
        gridspec : matplotlib.gridspec.SubplotSpec, optional
            If provided, plot is embedded inside a larger layout.
        scatter_kwargs : dict, optional
            Passed to ax.scatter().
        hist_kwargs : dict, optional
            Passed to ax.hist().
        ratio : int
            Size ratio between main plot and marginals.
        space : float
            Space between axes.

        Returns
        -------
        ax_joint, ax_marg_x, ax_marg_y
        """

        if scatter_kwargs is None:
            scatter_kwargs = {}
        if xhist_kwargs is None:
            xhist_kwargs = {}
        if yhist_kwargs is None:
            yhist_kwargs = {}
        if linreg_kwargs is None:
            linreg_kwargs = {}

        if fig is None:
            fig = plt.figure()

        if gridspec is None:
            gs = fig.add_gridspec(
                ratio + 1,
                ratio + 1,
                hspace=space,
                wspace=space
            )
        else:
            gs = gridspec.subgridspec(
                ratio + 1,
                ratio + 1,
                hspace=space,
                wspace=space
            )

        ax_joint = fig.add_subplot(gs[1:, :-1])
        ax_marg_x = fig.add_subplot(gs[0, :-1], sharex=ax_joint)
        ax_marg_y = fig.add_subplot(gs[1:, -1], sharey=ax_joint)

        # main scatter
        ax_joint.scatter(x, y, **scatter_kwargs)
        if linreg==True:
            sns.regplot(x=x,y=y,ax=ax_joint, **linreg_kwargs)

        # marginal histograms
        ax_marg_x.hist(x, **xhist_kwargs)
        ax_marg_y.hist(y, orientation="horizontal", **yhist_kwargs)

        # clean marginal axes
        ax_marg_x.tick_params(labelbottom=False)
        ax_marg_y.tick_params(labelleft=False)

        return ax_joint, ax_marg_x, ax_marg_y




    def bivariate_numeric_numeric_snapshot(self,
                                           data:pd.DataFrame,
                                            column_combos:list,
                                            plot_type:str='scatter',
                                            linreg:bool=True,                        
                                            super_title:str|None="Bivariate Analysis of Numeric and Numeric",
                                            plot_type_kwargs:dict|None=None,
                                            linreg_kwargs:dict|None=None ,
                   streamlit_:bool|None = None):
        """
        column_combos should be a list of column combinations: [(col_a,col_b), (col_c,col_d), ...]
        plot_type should be of 'joint' or 'scatter' and specifies plot type
        linreg is boolean whether or not to plot a linear regulation line
        plot_type_kwargs 
                scatter and linreg: sns.scatterplot(x=x_vector, y=y_vector, ax=ax, **plot_type_kwargs) sns.regplot(x=x_vector, y=y_vector, ax=ax, **linreg_kwargs)
                joint (linreg_kwargs should be included in plot type kwargs):   self._jointplot_gridspec(x_vector, y_vector, linreg=linreg, fig=fig, gridspec=outer[curr_row,curr_col], **plot_type_kwargs)  
                                scatter_kwargs=None,==> ax_joint.scatter(x, y, **scatter_kwargs)
                                xhist_kwargs=None,==> ax_marg_x.hist(x, **xhist_kwargs)
                                yhist_kwargs=None,==> ax_marg_y.hist(y, orientation="horizontal", **yhist_kwargs)
                                linreg_kwargs=None,==> sns.regplot(x=x,y=y,ax=ax_joint, **linreg_kwargs)
                                ratio=4,
                                space=0.05
        """
        plt.rcdefaults()
        if streamlit_ is None:
            streamlit_ = False 
        if plot_type_kwargs==None: plot_type_kwargs={}
        if  linreg_kwargs==None: linreg_kwargs={}

        row_height=9 
            
        #create figure
        num_rows=max(1,(len(column_combos)//2)+(len(column_combos)%2))
        num_cols=2 if len(column_combos)>1 else 1
        plot_height=(row_height*num_rows)+(2 if num_rows <=4 else 0)
        fig = plt.figure(figsize=(20,plot_height))#,constrained_layout=True)
        # create gridspec
        outer = fig.add_gridspec(
                    nrows=num_rows,
                    ncols=num_cols,
                    height_ratios=None,#[1 for i in range(num_rows)], 
                    hspace=0.3,
                    wspace=0.3
                    )

        # super title
        if column_combos: 
            plt.suptitle(super_title+'\n\n',fontsize=25)
        else:
            plt.suptitle("Nothing to Plot\n"+super_title,fontsize=12)
        # fill rows
        for index in range(len(column_combos)):
            x_vector=data[column_combos[index][0]]
            y_vector=data[column_combos[index][1]]
            curr_row, curr_col = index//2 , index%2
            if plot_type == 'scatter':
                ax = fig.add_subplot(outer[curr_row,curr_col])
                sns.scatterplot(x=x_vector, y=y_vector, ax=ax, **plot_type_kwargs)
                if linreg==True:
                    sns.regplot(x=x_vector, y=y_vector, ax=ax, **linreg_kwargs)
            elif plot_type == 'joint':
                ax, ax_marg_x, ax_marg_y = self._jointplot_gridspec(x_vector, y_vector, linreg=linreg, fig=fig, gridspec=outer[curr_row,curr_col], **plot_type_kwargs)   
            ax.tick_params(axis='x',rotation=45)
            for label in ax.get_xticklabels():
                label.set_ha('right')   
            #   title not supported at this time           
            #title_ = column_combos[index][0]+'\n&\n'+column_combos[index][1]
            #ax.set_title(title_)
            ax.set_ylabel(column_combos[index][1])
            ax.set_xlabel(column_combos[index][0])
            ax.grid()
        #plt.tight_layout()
        if streamlit_==False:
            plt.show()
        else:
            fig = plt.gcf()
            st.pyplot(fig,clear_figure=True)  
        plt.rcdefaults()
        return

    # =====================================================================================================================================
    # univariate_categorical_snapshot
    # =====================================================================================================================================




    def univariate_categorical_snapshot(self,
                                        data:pd.DataFrame,
                                        categorical:list|None= None,
                                        proportions:bool=False,
                                        n_wide:int|tuple|list=(6,40,4),
                                        super_title:str|None="Univariate Analysis of Categorical Variables" ,
                   streamlit_:bool|None = None):
        """
        categorical should be a list of columns or None. If None, categories will be autodetected
        proportions should be boolean. If False, then counts will be used
        where n_wide determines horizontal or vertical bars
            if type(n_wide)==int, horizontal bars and n_wide and indicates n plot_cols, 
            otherwise vertical bars and n_wide is a list or tuple where 
                    n_wide[0]=n cols on row, and 
                    n_wide[1]=ideal_max bars on row, 
                    <n_wide[2]=row_height or default is 5>

        """
        plt.rcdefaults()
        if streamlit_ is None:
            streamlit_ = False 
        columns=categorical if categorical is not None else list(data.select_dtypes('object').columns)

        if isinstance(n_wide,int):  # case where plots should have horizontal bars.
            raise ValueError("Sorry horizontal bars are not yet supported. Please provide an array like input for n_wide")
        else:                       # case where plots should have vertical bars.
            row_height=5 if (len(n_wide)<3) else n_wide[2]
            map_cols_to_rows, header_dict, alternate_plots = self._columns_in_rows_map(data, 
                                                                columns, 
                                                                max_bars_on_row=n_wide[1], 
                                                                num_columns_per_row=n_wide[0], 
                                                            univariate=True)
        #create figure
        num_rows=max(1,len(map_cols_to_rows))
        plot_height=(row_height*num_rows)+(2 if num_rows <=4 else 0)
        fig = plt.figure(figsize=(20,plot_height))#,constrained_layout=True)
        # create gridspec
        gs = GridSpec(
                    nrows=num_rows,
                    ncols=n_wide[0],
                    height_ratios=None,#[1 for i in range(num_rows)], 
                    hspace=0.5,
                    wspace=0.45
                )
        # super title
        if map_cols_to_rows: 
            plt.suptitle(super_title+'\n\n',fontsize=25)
        else:
            plt.suptitle("Nothing to Plot\n"+super_title,fontsize=12)
        # fill rows
        curr_row=0
        available_row_start_index = 0
        for index in range(len(map_cols_to_rows)):
            curr_row=index
            row_map = map_cols_to_rows[index]
            # fill columns on each row
            for plot_width in row_map:
                # fit axis for plot
                ax = fig.add_subplot(gs[curr_row, available_row_start_index : available_row_start_index+plot_width])
                # start position
                available_row_start_index+=plot_width
                # get plot data
                plot_header = header_dict[str(plot_width)].pop()
                if not isinstance(plot_header,str):
                    plot_header=plot_header[0]
                plot_data = data[plot_header].value_counts(normalize=proportions)
                # plot
                plot_title = plot_header
                plt.title(plot_title)
                sns.barplot(x=plot_data.index,y=plot_data.values,ax=ax)
                ax.tick_params(axis='x',rotation=45)
                for label in ax.get_xticklabels():
                    label.set_ha('right') 
                y_label='Count' if proportions==False else 'Proportion'
                ax.set_ylabel(y_label)                    
                if len(ax.get_xticks()) <= 60:
                    plt.grid()
            curr_row+=1
            available_row_start_index = 0
        #plt.tight_layout()
        if streamlit_==False:
            plt.show()
        else:
            fig = plt.gcf()
            st.pyplot(fig,clear_figure=True)  
        plt.rcdefaults()
        return

    # =====================================================================================================================================
    # univariate_numerical_snapshot
    # =====================================================================================================================================


    def univariate_numerical_snapshot(self,
                                        data:pd.DataFrame,
                                        numerical:list|None=None,
                                        n_wide:int|tuple|list=(6,40,4),
                                        kde:bool|None=None,
                                        super_title:str|None="Univariate Analysis of Numerical Variables",
                                        proportions:bool|None=None,
                                        keep_bins_significant:dict|None|bool=None ,
                   streamlit_:bool|None = None):
        """
        plots a histplot
            by default bins are decided by np.histogram_bin_edges bins='auto' which chooses the min(Sturges' Method, Freedman-Diaconis Rule)
            ## custom binning is supported, such as if the varaible has failed to reject null relationships in AnalyzeDataset module, 
            ## for custom bins, the BinnerClass should be used to provide a dictionary of header:bins in the **keep_bins_significant** parameter: {str(header):np.array|pd.Series(bins)}
        kde==True includes a kernel density estimate of the data

        numerical should be a list of columns or None. If None, numerical will be autodetected
        proportions should be boolean. If False, then counts will be used
        where n_wide determines horizontal or vertical bars
            if type(n_wide)==int, horizontal bars and n_wide and indicates n plot_cols, 
            otherwise vertical bars and n_wide is a list or tuple where 
                    n_wide[0]=n cols on row, and 
                    n_wide[1]=ideal_max bars on row, 
                    <n_wide[2]=row_height or default is 5>

        """
        plt.rcdefaults()
        if streamlit_ is None:
            streamlit_ = False 
        if kde is None: 
            kde=False
        if proportions is None:
            proportions=False

        stat = 'count' if proportions in (None,False) else 'probability'

        columns=numerical if numerical is not None else list(data.select_dtypes(['number',np.number]).columns)

        if (not keep_bins_significant) or (keep_bins_significant==False) or (keep_bins_significant is None):
            keep_bins_significant = {}

        # precompute bins for better compatability with self._columns_in_rows_map()
        bin_dict = {}
        for col in columns:
            if col in keep_bins_significant.keys():
                vals = pd.Series(keep_bins_significant[col])
                bin_dict.update({col:vals})
            else:
                try:
                    vals = pd.Series(np.histogram_bin_edges(data[col],bins='auto'))
                except:
                    vals = pd.Series(np.histogram_bin_edges(data[col], range = (data[col].min(),data[col].max())))
                bin_dict.update({col:vals})
        
        if isinstance(n_wide,int):  # case where plots should have horizontal bars.
            raise ValueError("Sorry horizontal bars are not yet supported. Please provide an array like input for n_wide")
        else:                       # case where plots should have vertical bars.
            row_height=5 if (len(n_wide)<3) else n_wide[2]
            map_cols_to_rows, header_dict, alternate_plots = self._columns_in_rows_map(bin_dict, 
                                                                columns, 
                                                                max_bars_on_row=n_wide[1], 
                                                                num_columns_per_row=n_wide[0], 
                                                                univariate=True)
        #create figure
        num_rows=max(1,len(map_cols_to_rows))
        plot_height=(row_height*num_rows)+(2 if num_rows <=4 else 0)
        fig = plt.figure(figsize=(20,plot_height))#,constrained_layout=True)
        # create gridspec
        gs = GridSpec(
                    nrows=num_rows,
                    ncols=n_wide[0],
                    height_ratios=None,#[1 for i in range(num_rows)], 
                    hspace=0.5,
                    wspace=0.45
                )
        # super title
        if map_cols_to_rows: 
            plt.suptitle(super_title+'\n\n',fontsize=25)
        else:
            plt.suptitle("Nothing to Plot\n"+super_title,fontsize=12)
        # fill rows
        curr_row=0
        available_row_start_index = 0
        for index in range(len(map_cols_to_rows)):
            curr_row=index
            row_map = map_cols_to_rows[index]
            # fill columns on each row
            for plot_width in row_map:
                # fit axis for plot
                ax = fig.add_subplot(gs[curr_row, available_row_start_index : available_row_start_index+plot_width])
                # start position
                available_row_start_index+=plot_width
                # get plot data
                plot_header = header_dict[str(plot_width)].pop()
                if not isinstance(plot_header,str):
                    plot_header=plot_header[0]
                plot_data = data[plot_header]
                plot_bins = bin_dict[plot_header]
                # plot
                plot_title = plot_header
                plt.title(plot_title)
                sns.histplot(x=plot_data,
                              bins=plot_bins,
                              ax=ax,kde=kde, 
                              stat=stat,
                              label='Observed',
                              legend=True,)
                ax.set_xticks(plot_bins)
                bincount = len(plot_bins)
                tenth= plot_bins[int(bincount*.1)]
                if tenth<100:
                    ax.set_xticklabels([f"{edge:,.2f}" for edge in plot_bins], rotation=45, ha='right')
                else:
                    ax.set_xticklabels([f"{edge:,.0f}" for edge in plot_bins], rotation=45, ha='right')
                '''ax.tick_params(axis='x',rotation=45)
                for label in ax.get_xticklabels():
                    label.set_ha('right') '''
                y_label='Count' if proportions==False else 'Proportion'
                ax.set_ylabel(y_label)
                # avoid crowded gridlines
                if bincount<=60:
                    plt.grid()
            curr_row+=1
            available_row_start_index = 0
        #plt.tight_layout()
        if streamlit_==False:
            plt.show()
        else:
            fig = plt.gcf()
            st.pyplot(fig,clear_figure=True)  
        plt.rcdefaults()
        return


    # =====================================================================================================================================
    # plot supercategories to subcategories W/ a function to map them to a figure   
    # =====================================================================================================================================


    # helper function called in prep_super_subcat_figure_maps()
    def _map_subcats_supercat_to_fig(self,
                                    dataframe:pd.DataFrame, 
                                    supercat_subcat_pair:list|tuple, 
                                    row_height:int=3, 
                                    cols_per_row:int=3, 
                                    y_tick_fontsize:int=17
                                    ):
        """
        where supercat_subcat_pair is [SUPERCAT, SUBCAT]
        row height is row approximate height of plot
        cols per row is absolute, not meant to be overlapped except for the supercat which will take a whole row at the top.
        y_tick_fontsize is used to calculate num gridspec rows each subcat used
        returns 
            n_rows: the total number of rows includeing the supercategory univariate plot
            supercat_placement_tracker: a list of nested lists of nested lists [ [[SUPERCATEGORY, INT(HEIGHT)],...],[[...]],...]
                where the index of the second level is the index of the intended column
            presently, it also returns 3 parameters it took as inputs
                row_height, cols_per_row, y_tick_fontsize
        """
        n_rows_for_supercat_dict = {}
        for supercat in dataframe[supercat_subcat_pair[0]].unique(): 
            n_subcats = dataframe[supercat_subcat_pair].loc[dataframe[supercat_subcat_pair[0]]==supercat][supercat_subcat_pair[1]].nunique()
            """ use this math:  figsize (inches) × dpi = pixels; fontsize (points) × dpi / 72 = pixels; where default dpi is 100;  fonstize*1.n for spacing"""
            # determine number of rows the plot needs
            num_rows_this_plot = ((y_tick_fontsize*1.15)/72)*n_subcats
            if (num_rows_this_plot%row_height)<0.005:
                num_rows_this_plot = int(num_rows_this_plot//row_height)
            else:
                num_rows_this_plot = int((num_rows_this_plot//row_height)+1)
            # convert to STRING
            # & add row(s) plotwise to make room for titles etcetera
            num_rows_this_plot = str(int(num_rows_this_plot+1))
            if num_rows_this_plot not in n_rows_for_supercat_dict.keys():
                n_rows_for_supercat_dict[num_rows_this_plot]=[supercat]
            else:
                n_rows_for_supercat_dict[num_rows_this_plot].append(supercat)
        # indexes in row_len_tracker match column indexes in rows. Num rows used is incremented for each column
        # grabbing biggest plots greedily and placing it in the shortest column  
        row_len_tracker = [0]*cols_per_row
        # indexes the supercats are placed in match the index of the figure column
        supercat_placement_tracker =[[] for i in range(cols_per_row)]
        while n_rows_for_supercat_dict:
            max_key=max(n_rows_for_supercat_dict.keys())
            if not n_rows_for_supercat_dict[max_key]:
                n_rows_for_supercat_dict.pop(max_key)
                continue
            else:
                super_category = n_rows_for_supercat_dict[max_key].pop()
                shortest_index_place = row_len_tracker.index(min(row_len_tracker))
                row_len_tracker[shortest_index_place]+=int(max_key)
                super_cat_and_height = [super_category,int(max_key)]
                supercat_placement_tracker[shortest_index_place].append(super_cat_and_height)
        # add one row to plot the supercategory vector
        n_rows = max(row_len_tracker)+1
        supercat_column, subcat_header = supercat_subcat_pair[0], supercat_subcat_pair[1]
        return n_rows, supercat_column, subcat_header, supercat_placement_tracker, row_height, cols_per_row, y_tick_fontsize

    # a function that takes [(SUPERCAT, SUBCAT),(),...] pairs and maps them to figure for function: plot_supercats_subcats
    def _prep_super_subcat_figure_maps(self,
                                        df: pd.DataFrame , 
                                        supercat_subcat_pairs:list|tuple, 
                                        row_height:int=3, 
                                        cols_per_row:int=3, 
                                        y_tick_fontsize:int=17
                                        ):
        """
        where subcat is used to describe the column partitioned by the supercat column
        takes a dataframe
        and list of supercat subcat pairs: [[SUPERCAT, SUBCAT],[SUPERCAT, SUBCAT],[SUPERCAT, SUBCAT],...]
        returns a map: [  [num rows this pair, 'SUPERCAT_column', 'SUBCAT_column', [
                                                                                [['SUPER_PARTITION_VALUE_1', int(height in rows)],
                                                                                    ['SUPER_PARTITION_VALUE_2', int(height in rows)],
                                                                                    ... ],
                        [num rows next pair, ...]
        and params for plotting: (row_height_in_inches, cols_per_row, y_tick_fontsize)
        """
        my_map = []
        my_params = None
        for cols in supercat_subcat_pairs:
            meta = self._map_subcats_supercat_to_fig(dataframe=df, 
                                    supercat_subcat_pair=cols, 
                                    row_height=row_height, 
                                    cols_per_row=cols_per_row, 
                                    y_tick_fontsize=y_tick_fontsize
                                    )
            if my_params == None:
                my_params = tuple(meta[-3:])
            map_data = list(meta[:4])
            my_map.append(map_data)
        return my_map, my_params

    # output this should adhere to: n_rows, supercat_column, subcat_header, supercat_placement_tracker, row_height, cols_per_row, y_tick_fontsize

    def plot_supercats_subcats(self,
                               data, 
                               figure_map:list, 
                               row_height, 
                               cols_per_row, 
                               y_tick_fontsize,
                               super_title:str|None=None ,
                   streamlit_:bool|None = None):
        """
        where figure map is ouptut[0] from prep_super_subcat_figure_maps()
        where figure map should be a list of lists: [
                                                    [n_rows_for_supercat_var, supercat_header, subcat_header, [[supercat_1,n_rows],[supercat_2,n_rows]],
                                                    [n_rows_for_supercat_var, supercat_header, subcat_header, [[supercat_1,n_rows],[supercat_2,n_rows]],
                                                    ...]
                                                    where each 1st level nest is output from _map_subcats_supercat_to_fig()[:3] for one supercat_header*subcat_header combo
        where supercats take one row and are included in n_rows_for_supercat_var
        row_height, cols_per_row, and y_tick_fontsize are passed from _map_subcats_supercat_to_fig()
        """
        plt.rcdefaults()
        if streamlit_ is None:
            streamlit_ = False 
        if super_title is None:
            super_title= 'Super and Sub Categories - One Categoric Variable Partitions Another'
        true_rows = sum(i[0] for i in figure_map)
        total_rows = max(1,true_rows)
        n_cols = len(figure_map[0][3])
        fig = plt.figure(figsize=(20,(row_height*total_rows)+2))#,constrained_layout=True)
        if true_rows>=1: 
            plt.suptitle(super_title+'\n\n',fontsize=25)
        else:
            plt.suptitle("Nothing to Plot\n"+super_title,fontsize=12)
        grid_specs = GridSpec(
                    nrows=total_rows,
                    ncols=n_cols,
                    height_ratios=None,#[1 for i in range(total_rows)], 
                    hspace=0.7,
                    wspace=0.45
                    )
        
        supercat_header_start_row = 0
        for supercat_map in figure_map:
            curr_supercat_header = supercat_map[1]
            ax_super = fig.add_subplot(grid_specs[supercat_header_start_row,:]) 
            super_plot_data=data[curr_supercat_header].value_counts()
            plt.title(curr_supercat_header)
            sns.barplot(x=super_plot_data.index,y=super_plot_data.values,ax=ax_super)
            ax_super.set_xlabel('')
            if len(ax_super.get_xticks()) <= 60:
                plt.grid()
            curr_subcat_header=supercat_map[2]
            # increment to the next row
            subcat_header_start_row = supercat_header_start_row+1
            for column_index,supercat_partitions in enumerate(supercat_map[3]):
                # stack plots based on map
                colum_stacker_start_row = subcat_header_start_row
                for supercat_partition in supercat_partitions:
                    partition=supercat_partition[0]
                    # partition and count subcats
                    sub_plot_data=data.loc[data[curr_supercat_header]==partition][curr_subcat_header].value_counts()
                    ax_sub = fig.add_subplot(grid_specs[colum_stacker_start_row:colum_stacker_start_row+supercat_partition[1],column_index])
                    # increment stacker start position
                    colum_stacker_start_row+=supercat_partition[1]
                    subcat_title='Partitioned by '+str(partition)
                    ax_sub.set_title(subcat_title)
                    sns.barplot(y=sub_plot_data.index,x=sub_plot_data.values,orient='h',ax=ax_sub)
                    ax_sub.tick_params(axis='y', labelsize=y_tick_fontsize)
                    plt.grid()

            # increment the start spot for the next supercat column
            supercat_header_start_row+=supercat_map[0]
        #plt.tight_layout()
        if streamlit_==False:
            plt.show()
        else:
            fig = plt.gcf()
            st.pyplot(fig,clear_figure=True)  
        plt.rcdefaults()
        return




    # =====================================================================================================================================
    # 
    # =====================================================================================================================================




    # =====================================================================================================================================
    # 
    # =====================================================================================================================================




    # =====================================================================================================================================
    # 
    # =====================================================================================================================================


