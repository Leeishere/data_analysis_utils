
import streamlit as st



def chunk_plotables_mutate_titles(
                                list_of_nested_lists_of_iterable_plottables:list,
                                list_of_unmutated_titles_where_indexes_match_plotables:list,
                                chunk_size:int
                                ):
    """
    list_of_nested_lists_of_iterable_plottables: a list of lists [numeric_reject_null, numeric_numeric_above_correlation, ...]
    list_of_unmutated_titles_where_indexes_match_plotables: a list of titles that corespond to list_of_nested_lists_of_iterable_plottables
    chunk_size controls max plottables in each chunk
    
    Create: mutated_group_plotable_titles: a list of titles with a count of plots added as suffix. index 2 of return statement
        and   group_plot_options: a list of chunked lists: [ [[],[],[]], [], [[],[],[],[]] ]
    """
    mutated_group_plotable_titles = list_of_unmutated_titles_where_indexes_match_plotables.copy()
    group_plot_options = []
    not_present_titles = []
    plottable_indexes = []
    for i in range(len( mutated_group_plotable_titles)):
        # fill a list with dict params to iterate over
        curr_plottables = []
        if list_of_nested_lists_of_iterable_plottables[i]:
            plottable_indexes.append(i)
            num_curr_pairs = len(list_of_nested_lists_of_iterable_plottables[i])
            if num_curr_pairs > chunk_size:
                start = 0
                stop  = chunk_size
                # create an iterable list at the index position
                while start < num_curr_pairs:
                    chunk = list_of_nested_lists_of_iterable_plottables[i][start:stop]
                    start = stop
                    stop = stop + chunk_size
                    appendable = chunk
                    curr_plottables.append(appendable)
                suffix = f": {num_curr_pairs} Plots"
                mutated_group_plotable_titles[i] += suffix
                group_plot_options.append(curr_plottables)
            else:
                # else enclose/append it in a list: [ original_list ], so it can be iterated 1 time
                curr_plottables.append(list_of_nested_lists_of_iterable_plottables[i])
                suffix = f": {num_curr_pairs} Plots"
                mutated_group_plotable_titles[i] += suffix
                group_plot_options.append(curr_plottables)
        else:
            group_plot_options.append(curr_plottables)
            not_present_titles.append(mutated_group_plotable_titles[i])

    return  group_plot_options  ,    mutated_group_plotable_titles  ,  not_present_titles , plottable_indexes



def plot_one_title(
                    data,
                    variable_to_plot:list|str,
                    curr_params:dict,  
                    var_type_plot:str,
                    plot_func_):

    """
    variable_to_plot is string header to be ploted, or a list pair of bivariate headers
    curr_params are parameters passed to the plot function
    """
    plotly_express = True
    if var_type_plot in ('num_univar', 'cat_univar'):
        par = {'x': variable_to_plot}
        curr_params.update(par)
        fig = plot_func_(data, **curr_params)
    elif var_type_plot in ('numnum_bivar', 'numcat_bivar'):
        par = {'x': variable_to_plot[1], 'y': variable_to_plot[0]}
        curr_params.update(par)
        fig = plot_func_(data, **curr_params)
    elif var_type_plot in ('catcat_bivar'):
        plot_func, par = plot_func_(variable_to_plot[0], variable_to_plot[1])
        curr_params.update(par)
        fig = plot_func(data_frame = data, **curr_params)
    elif var_type_plot in ('super_subcat_pairs'):
        par = {'super_subcat_pairs': [variable_to_plot],
                'cat_univar':False,   
                'num_univar':False, 
                'catcat_bivar':False,
                'numnum_bivar':False,
                'numcat_bivar':False}
        plot_params = {'super_subcat_pairs_params': curr_params ,'streamlit_':True}
        plot_params.update(par)
        #### THIS 'plotly_express' FLAG EXCLUDES SUPERCATEGORY SUBCATAGORY PARTIION PLOTS WHERE A MATPLOLIB FIGURE IS RETURNED INTERNALLY
        plotly_express = False  # set to false because unlike the plotly.express functions, this renders the figure without needing to call st.write(fig)
        fig = plot_func_(data, **plot_params)
    else:
        fig = None
    # to catch plots where functions display figures internally
    if (plotly_express):
        if fig is None:
            st.warning('No figure generated for this tab.')
        elif hasattr(fig, 'to_plotly_json'):
            st.plotly_chart(fig, use_container_width=True)
        elif isinstance(fig, tuple) and fig and hasattr(fig[0], 'savefig'):
            st.pyplot(fig[0], use_container_width=True)
        elif hasattr(fig, 'savefig'):
            st.pyplot(fig, use_container_width=True)
        elif hasattr(fig, 'figure') and hasattr(fig.figure, 'savefig'):
            st.pyplot(fig.figure, use_container_width=True)
        else:
            st.write(fig)
                