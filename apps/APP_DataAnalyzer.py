import numpy as np
import pandas as pd
import pathlib
import streamlit as st
import plotly.express as px
import datetime

from data_analysis_utils import AnalyzeDataset
from data_analysis_utils.BinnerClass import Bin
from data_analysis_utils.CompareColumns import CompareColumns
from data_analysis_utils.MuEstimator import MuEstimator


DEFAULT_DATASET_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "shopping_behavior_updated.csv"
NAVIGATION_PAGES = ["Data Upload & Processing", "Group Visualizations", "Target Visualizations", "Feedback", "Binning Tool"]


def chunk_plotables_mutate_titles(
                                list_of_nested_lists_of_iterable_plottables:list,
                                list_of_unmutated_titles_where_indexes_match_plotables:list,
                                chunk_size:int
                                ):
    """
    list_of_nested_lists_of_iterable_plottables: a list of lists [numeric_reject_null, numeric_numeric_above_correlation, ...]
    list_of_unmutated_titles_where_indexes_match_plotables: a list of titles that corespond to list_of_nested_lists_of_iterable_plottables
    chunk_size controls max plottables in each chunk
    """
    mutated_group_plotable_titles = list_of_unmutated_titles_where_indexes_match_plotables.copy()
    group_plot_options = []
    not_present_titles = []
    plottable_indexes = []
    for i in range(len(mutated_group_plotable_titles)):
        curr_plottables = []
        if list_of_nested_lists_of_iterable_plottables[i]:
            plottable_indexes.append(i)
            num_curr_pairs = len(list_of_nested_lists_of_iterable_plottables[i])
            if num_curr_pairs > chunk_size:
                start = 0
                stop = chunk_size
                while start < num_curr_pairs:
                    chunk = list_of_nested_lists_of_iterable_plottables[i][start:stop]
                    start = stop
                    stop = stop + chunk_size
                    curr_plottables.append(chunk)
                mutated_group_plotable_titles[i] += f": {num_curr_pairs} Plots"
                group_plot_options.append(curr_plottables)
            else:
                curr_plottables.append(list_of_nested_lists_of_iterable_plottables[i])
                mutated_group_plotable_titles[i] += f": {num_curr_pairs} Plots"
                group_plot_options.append(curr_plottables)
        else:
            group_plot_options.append(curr_plottables)
            not_present_titles.append(mutated_group_plotable_titles[i])

    return group_plot_options, mutated_group_plotable_titles, not_present_titles, plottable_indexes


def plot_one_title(
                    data,
                    variable_to_plot:list|str,
                    curr_params:dict,
                    var_type_plot:str,
                    plot_func_):
    """
    variable_to_plot is string header to be ploted, or a list pair of bivariate headers.
    curr_params are parameters passed to the plot function.
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
        fig = plot_func(data_frame=data, **curr_params)
    elif var_type_plot in ('super_subcat_pairs'):
        par = {'super_subcat_pairs': [variable_to_plot],
                'cat_univar':False,
                'num_univar':False,
                'catcat_bivar':False,
                'numnum_bivar':False,
                'numcat_bivar':False}
        plot_params = {'super_subcat_pairs_params': curr_params, 'streamlit_':True}
        plot_params.update(par)
        plotly_express = False
        fig = plot_func_(data, **plot_params)
    else:
        fig = None

    if plotly_express:
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


# SESSION STATE
# ======================|
# DATA MANIPULATION     | 
# MODEL PARAMETERS      |
# UI CONTROL            |
# PLOT PARAMS           |
# PLOT LISTS            |
# CALL_BACKS            |
# ======================|


# DATA MANIPULATION
# =======================================================================================================================================
# =======================================================================================================================================


if 'indexed_col_meta_df' not in st.session_state:
    st.session_state.indexed_col_meta_df = None        

if 'user_mutated_data' not in st.session_state:
    st.session_state.user_mutated_data = None

if "datatypes_df" not in st.session_state:
    st.session_state.datatypes_df = None

if 'ui_df' not in st.session_state:
    st.session_state.ui_df = None

if 'result_df' not in st.session_state:
    st.session_state.result_df = None

if 'data' not in st.session_state:
    st.session_state.data = None

if 'one_not_recognized' not in st.session_state:
    st.session_state.one_not_recognized = None

if 'not_the_same' not in st.session_state:
    st.session_state.not_the_same = None

if "original_len" not in st.session_state:
    st.session_state.original_len = None

if "original_shape" not in st.session_state:
    st.session_state.original_shape = None

if 'max_date_cols_used' not in st.session_state:
    st.session_state.max_date_cols_used = None

if 'min_pct_non_null_to_propose_a_dtype' not in st.session_state:
    st.session_state.min_pct_non_null_to_propose_a_dtype = None

if 'max_unique_pct_of_total_ie_identifier_variable_ID' not in st.session_state:
    st.session_state.max_unique_pct_of_total_ie_identifier_variable_ID = None

if 'max_pct_unique_for_numtype_cattype_threshold' not in st.session_state:
    st.session_state.max_pct_unique_for_numtype_cattype_threshold = None

if 'propose_drop_pct_nan_int' not in st.session_state:
    st.session_state.propose_drop_pct_nan_int = None

if 'max_frac_droppable_nan_cat' not in st.session_state:
    st.session_state.max_frac_droppable_nan_cat = None

# MODEL PARAMETERS
# =======================================================================================================================================
# =======================================================================================================================================


if "kruskal_assumption_check_params" not in st.session_state:
    st.session_state.kruskal_assumption_check_params = {"levene_alpha": 0.01, "ks_alpha": 0.01, "return_pseudo": False, "pseudo_test_max_global_ties_ratio": 0.7, "full_pseudo": False, "dropna": True, "n_jobs": 4, "guesstimate": {"rej_max_pct_in_group": 0.2, "max_num_outlier_all_reject": 3, "max_pct_reject_total": 0.2}}

if "anova_assumption_check_params" not in st.session_state:
    st.session_state.anova_assumption_check_params = {"normality_alpha": 0.01, "homogeneity_alpha": 0.01, "min_n": 5, "iqr_multiplier": 2, "dropna": True}

if "chi2_assumption_check_params" not in st.session_state:
    st.session_state.chi2_assumption_check_params = {"dropna": True}

if "supercat_subcat_params" not in st.session_state: 
    st.session_state.supercat_subcat_params = {"max_evidence": 0.2, "isolate_super_subs": False}

if "multivariate_params" not in st.session_state:
    st.session_state.multivariate_params = {"max_n_combination_size": 2, "max_n_combinations": 20_000, "min_combo_size": 2}

if "multivariate_concatenation_delimiter" not in st.session_state:
    st.session_state.multivariate_concatenation_delimiter = "_|&|_"

if "numnum_meth_alpha_above_instructions" not in st.session_state:
    st.session_state.numnum_meth_alpha_above_instructions = [["pearson", 0.6, None], ["spearman", 0.6, None], ["kendall", 0.6, None]]

if "numcat_meth_alpha_above_instructions" not in st.session_state:
    st.session_state.numcat_meth_alpha_above_instructions = [["kruskal", 0.05, None], ["anova", 0.05, None]]

if "catcat_meth_alpha_above_instructions" not in st.session_state:
    st.session_state.catcat_meth_alpha_above_instructions = [["chi2", 0.05, None]]

if "good_of_fit_uniform_test_instructions" not in st.session_state:
    st.session_state.good_of_fit_uniform_test_instructions = [0.05, None]

if "normal_test_instructions" not in st.session_state:
    st.session_state.normal_test_instructions = [0.05, None]

# UI CONTROL
# =======================================================================================================================================
# =======================================================================================================================================
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

if "page" not in st.session_state:
    st.session_state.page = "Data Upload & Processing"

if 'early_stop' not in st.session_state:
    st.session_state.early_stop = None

if 'step' not in st.session_state:
    st.session_state.step = 1

if 'use_default_dataset' not in st.session_state:
    st.session_state.use_default_dataset = False

if 'default_filepath' not in st.session_state:
    st.session_state.default_filepath = DEFAULT_DATASET_PATH

if 'unbinned_columns' not in st.session_state:
    st.session_state.unbinned_columns = None

if 'binned_columns' not in st.session_state:
    st.session_state.binned_columns = []

if 'binned_exists_as' not in st.session_state:
    st.session_state.binned_exists_as = None

if 'curr_original_unbinned' not in st.session_state:
    st.session_state.curr_original_unbinned = 'Review Rating'

if "numnum_metrics" not in st.session_state:
    st.session_state.numnum_metrics = ['pearson', 0.6, True]

if "catnum_metrics" not in st.session_state:
    st.session_state.catnum_metrics = ['kruskal', 0.05, False]

if "abs_min_bins" not in st.session_state:
    st.session_state.abs_min_bins = None

if "col_by_col_min_bins" not in st.session_state:
    st.session_state.col_by_col_min_bins = None

if "custom_min_bins" not in st.session_state:
    st.session_state.custom_min_bins = {}

if "binning_processed" not in st.session_state:
    st.session_state.binning_processed = False

# PLOT PARAMS
# =======================================================================================================================================
# =======================================================================================================================================

if 'n_wide' not in st.session_state:
    st.session_state.n_wide = {'n_wide': [8, 40, 4]}  # axises per row, total bars per row, row height in inches

if "cat_univar_params" not in st.session_state:
    st.session_state.cat_univar_params = {"histfunc":'count'}

if "catcat_bivar_params" not in st.session_state:
    st.session_state.catcat_bivar_params = {}

if "numnum_bivar_params" not in st.session_state:
    st.session_state.numnum_bivar_params = {'marginal_x':"histogram", 'marginal_y':"histogram", 'trendline':"ols"}  
    
if "numcat_bivar_params" not in st.session_state:
    st.session_state.numcat_bivar_params = {}

if "super_subcat_pairs_params" not in st.session_state:
    st.session_state.super_subcat_pairs_params = {"row_height": 3, "cols_per_row": 2, "y_tick_fontsize": 12,  
                                                  "super_title": "Supercategory-Subcategory - One Categoric Variable Partitions Another"}

if "num_univar_params" not in st.session_state:
    st.session_state.num_univar_params = {}

if 'plot_params_dict' not in st.session_state:
    st.session_state.plot_params_dict = {
        'num_univar_params':st.session_state.num_univar_params,
        'cat_univar_params':st.session_state.cat_univar_params,
        'numnum_bivar_params':st.session_state.numnum_bivar_params,
        'numcat_bivar_params':st.session_state.numcat_bivar_params,
        'catcat_bivar_params':st.session_state.catcat_bivar_params,
        'catcat_bivar_params':st.session_state.catcat_bivar_params,
        'super_subcat_pairs_params':st.session_state.super_subcat_pairs_params,
        }

if "catunivarparams" not in st.session_state:
    st.session_state.catunivarparams = {"proportions": False, "n_wide": (8, 30, 4), "super_title": "Univariate Categorical Variables - Reject Good-Of-Fit for Uniform"}

if "catcatbivarparams" not in st.session_state:
    st.session_state.catcatbivarparams = {"n_wide": (8, 30, 4), "stacked_bars_when_max_bars_is_exceeded": True, "sorted": False, "super_title": "Categoric-To-Categoric Bivariates - Reject Null"}

if "numnumbivarparams" not in st.session_state:
    st.session_state.numnumbivarparams = {"plot_type": "joint", "linreg": False, "plot_type_kwargs": None, "linreg_kwargs": None, "super_title": "Numeric Bivariates With Significant Correlation(s)"}

if "numcatbivarparams" not in st.session_state:
    st.session_state.numcatbivarparams = {"plot_type": "boxen", "n_wide": (8, 30, 4), "super_title": "Numeric-to-Categoric Bivariates  - Reject Null"}

if "supersubcatpairsparams" not in st.session_state:
    st.session_state.supersubcatpairsparams = {"row_height": 3, "cols_per_row": 2, "y_tick_fontsize": 12, "super_title": "Supercategory-Subcategory - One Categoric Variable Partitions Another"}

if "numunivarparams" not in st.session_state:
    st.session_state.numunivarparams = {"kde": None, "proportions": False, "n_wide": (8, 30, 4), "super_title": "Univariate Numerical Variables - Reject Normal Distribution", "force_significant_bin_edges": None, "minimize_significant_bins": None, "include_multivariate": True}


# PLOT LISTS
# =====================================================================================================================================
# =====================================================================================================================================
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 5

if 'group_plot_options' not in st.session_state:
    st.session_state.group_plot_options = None

if 'target_plot_options' not in st.session_state:
    st.session_state.target_plot_options = None

if 'mutated_group_plot_titles' not in st.session_state:
    st.session_state.mutated_group_plot_titles = None

if 'mutated_target_plot_titles' not in st.session_state:
    st.session_state.mutated_target_plot_titles = None

if 'plottable_group_indexes' not in st.session_state:
    st.session_state.plottable_group_indexes = None

if 'plottable_target_indexes' not in st.session_state:
    st.session_state.plottable_target_indexes = None

if 'plot_list_keys' not in st.session_state:
    st.session_state.plot_list_keys = [
        'num_univar', 'num_univar', 
        'cat_univar', 'cat_univar',
        'numnum_bivar', 'numnum_bivar',
        'numcat_bivar', 'numcat_bivar',
        'catcat_bivar', 'catcat_bivar',
        'super_subcat_pairs',
        'catcat_bivar', 'numnum_bivar', 'numcat_bivar', 
        'num_univar', 'cat_univar'
        ]
    
if 'plot_param_keys' not in st.session_state:
    st.session_state.plot_param_keys = [
                                            'num_univar_params' , 'num_univar_params' ,
                                            'cat_univar_params' , 'cat_univar_params' ,
                                            'numnum_bivar_params' , 'numnum_bivar_params' ,
                                            'numcat_bivar_params' , 'numcat_bivar_params' ,
                                            'catcat_bivar_params' , 'catcat_bivar_params' ,
                                            'super_subcat_pairs_params' ,
                                            'catcat_bivar_params' , 'numnum_bivar_params' , 'numcat_bivar_params' , 
                                            'num_univar_params' , 'cat_univar_params'
                                            ]
    
if 'unmutated_title_list' not in st.session_state:
    # un mutated plot options
    st.session_state.unmutated_title_list = [
                'Numerical Non-Normal', 'Numerical Normal',
                'Categorical Non-Uniform', 'Categorical Uniform',
                'Numeric-Numeric With Correlation','Numeric-Numeric Without Correlation',
                'Numeric-Categoric Reject Null', 'Numeric-Categoric Fail to Reject Null',
                'Categoric-Categoric Reject Null','Categoric-Categoric Fail to Reject Null',
                'Categoric Partitioned by Another Categoric',
                'Assumptions Not Met - Categoric-Categoric',
                'Assumptions Not Met - Numeric-Numeric',
                'Assumptions Not Met - Numeric-Categoric',
                'Assumptions Not Met - Numerical',
                'Assumptions Not Met - Categorical'
                ]
    
if ('plot_function' not in st.session_state):
    st.session_state.plot_function = None
    
if 'curr_group_plot_selection' not in st.session_state:
    st.session_state.curr_group_plot_selection = None

if 'curr_target_plot_selection' not in st.session_state:
    st.session_state.curr_target_plot_selection = None

if 'master_group_plot_index' not in st.session_state:
    st.session_state.master_group_plot_index = None

if 'master_target_plot_index' not in st.session_state:
    st.session_state.master_target_plot_index = None

if 'curr_target_selection' not in st.session_state:
    st.session_state.curr_target_selection = None

if 'curr_target_plot_lists' not in st.session_state:
    st.session_state.curr_target_plot_lists = None
    
# CALL_BACKS
# =======================================================================================================================================
# =======================================================================================================================================  
def set_page():
    st.session_state.page = st.session_state.navigation_control

if 'group_plot_selection' not in st.session_state:
    st.session_state.group_plot_selection = None
def group_plot():
    st.session_state.group_plot_selection = st.session_state.group_plot_control

if 'target_plot_selection' not in st.session_state:
    st.session_state.target_plot_selection = None
def target_plot():
    st.session_state.target_plot_selection = st.session_state.target_plot_control

if 'group_selection' not in st.session_state:
    st.session_state.group_selection = None
def  group_chunk_selector():
    st.session_state.group_selection = st.session_state.group_chunk_select

if 'group_targ_selection' not in st.session_state:
    st.session_state.group_targ_selection = None
def  group_targ_chunk_selector():
    st.session_state.group_targ_selection = st.session_state.group_targ_chunk_select

# functions
# =======================================================================================================================================
# =======================================================================================================================================  
# process bivariate categorical input to determine plot type: group vs stack
def return_catcat_plottype(x, y):
    _df = st.session_state.indexed_col_meta_df
    xniq = int(_df.loc[x, 'n_unique'])
    yniq = int(_df.loc[y, 'n_unique'])

    curr_x_curr_color = (x, y) if xniq > yniq else (y, x)

    if (xniq * yniq) <= st.session_state.n_wide['n_wide'][1]:
        params = {
            'x': curr_x_curr_color[0],
            'color': curr_x_curr_color[1],
            'histfunc': 'count',
            'barmode': 'group',
        }
    else:
        params = {
            'x': curr_x_curr_color[0],
            'color': curr_x_curr_color[1],
            'histfunc': 'count',
            'barmode': 'stack',
        }

    return px.histogram, params


def reset_binning_state():
    st.session_state.unbinned_columns = None
    st.session_state.binned_columns = []
    st.session_state.binned_exists_as = None
    st.session_state.curr_original_unbinned = 'Review Rating'
    st.session_state.abs_min_bins = None
    st.session_state.col_by_col_min_bins = None
    st.session_state.custom_min_bins = {}
    st.session_state.binning_processed = False


@st.cache_data(show_spinner=False)
def cached_min_bins(dataframe, numnum_metrics, catnum_metrics, numeric_columns_key=None):
    cached_bin = Bin()
    numeric_columns = list(numeric_columns_key) if numeric_columns_key is not None else None
    min_bin_dict = cached_bin.relational_binner(
        dataframe,
        numnum_meth_alpha_above=tuple(numnum_metrics),
        numcat_meth_alpha_above=tuple(catnum_metrics),
        original_value_count_threshold=5,
        numeric_columns=numeric_columns,
        categoric_columns=None,
        numeric_target=None,
        categoric_target=None,
    )
    return min_bin_dict, cached_bin.numeric_feature_col_thresholds


def get_dict_for_min_bins(numeric_columns=None):
    numeric_columns_key = tuple(numeric_columns) if numeric_columns is not None else None
    return cached_min_bins(
        st.session_state.data,
        tuple(st.session_state.numnum_metrics),
        tuple(st.session_state.catnum_metrics),
        numeric_columns_key,
    )


def update_data_with_binned_columns(data_, bin_dict):
    bin_model = Bin()
    data = data_.copy()
    binned_list = []
    for k, v in bin_dict.items():
        header = f"{k} -> Binned"
        data[header] = bin_model.binner(data[k], v, rescale=True, return_bins=False)
        data[header] = round(data[header], 3).astype(float)
        binned_list.append(header)
    return data, binned_list


def ensure_default_binning():
    has_custom_bins = isinstance(st.session_state.custom_min_bins, dict) and len(st.session_state.custom_min_bins) > 0
    needs_binning = (not has_custom_bins) or any(
        f"{col} -> Binned" not in st.session_state.data.columns
        for col in st.session_state.custom_min_bins.keys()
    )
    if needs_binning:
        with st.spinner("Preparing default binning for analysis views..."):
            st.session_state.abs_min_bins, st.session_state.col_by_col_min_bins = get_dict_for_min_bins()
            st.session_state.custom_min_bins = st.session_state.abs_min_bins.copy()
            data = st.session_state.data.copy()
            st.session_state.data, st.session_state.binned_columns = update_data_with_binned_columns(
                data,
                st.session_state.custom_min_bins,
            )
            st.session_state.unbinned_columns = list(st.session_state.custom_min_bins.keys())
            st.session_state.binning_processed = True
            if st.session_state.unbinned_columns and (
                st.session_state.curr_original_unbinned not in st.session_state.unbinned_columns
            ):
                st.session_state.curr_original_unbinned = st.session_state.unbinned_columns[0]

    if st.session_state.unbinned_columns and st.session_state.curr_original_unbinned is not None:
        st.session_state.binned_exists_as = f"{st.session_state.curr_original_unbinned} -> Binned"


def render_navigation(label="Navigation"):
    with st.sidebar:
        st.header("Navigation")
        st.session_state.page = st.radio(
            label,
            NAVIGATION_PAGES,
            index=NAVIGATION_PAGES.index(st.session_state.page),
            key='navigation_control',
            on_change=set_page,
        )


def render_binning_tool():
    render_navigation("Navigation")

    if st.session_state.data is None or st.session_state.step < 6:
        st.info("Please complete Data Upload & Processing before using the Binning Tool.")
        return

    st.title("Bin Variables Based on Hypothesis Tests and/or Correlation Coefficients.", text_alignment='center')

    st.markdown('...', text_alignment='center')
    st.markdown("Minimum Bin Sizes that Retain Statistical Relationships", text_alignment='center')
    st.markdown('...', text_alignment='center')

    coco = CompareColumns()
    ensure_default_binning()

    examin_bins = st.columns([1], gap='large', vertical_alignment='top', border=True)
    pre_binned_lineplot_cell, pre_bin_relationships = st.columns([.45, .55], gap='large', vertical_alignment='top', border=True)
    post_binned_countplot_cell, post_binned_relationships = st.columns([.45, .55], gap='large', vertical_alignment='top', border=True)
    binned_mu_plot = st.columns([1], gap='large', vertical_alignment='top', border=True)

    examin_bins = examin_bins[0]
    with examin_bins:
        if st.session_state.unbinned_columns:
            st.subheader("Column Selection", divider='grey', anchor=False, text_alignment='center')
            st.markdown("Select the Binned Variable You Want to Examine.")
            try:
                unbinned_default_start_index = st.session_state.unbinned_columns.index(st.session_state.curr_original_unbinned)
            except:
                unbinned_default_start_index = 0
            finally:
                unbinned = st.selectbox(
                    "",
                    st.session_state.unbinned_columns,
                    index=unbinned_default_start_index,
                    key="selected_unbinned",
                    help=None,
                    on_change=None,
                    args=None,
                    kwargs=None,
                    placeholder=None,
                    disabled=False,
                    label_visibility="visible",
                    accept_new_options=False,
                    width="stretch",
                )
                st.session_state.binned_exists_as = f"{unbinned} -> Binned"
                st.session_state.curr_original_unbinned = unbinned

    with pre_binned_lineplot_cell:
        if st.session_state.binned_exists_as is not None:
            st.subheader('Pre Bin Plot', divider='grey', anchor=False, text_alignment='center')
            st.markdown("Line Plot")
            unbinned_lineplot_data = st.session_state.data[st.session_state.curr_original_unbinned].copy().to_frame().reset_index(drop=True).reset_index(drop=False)
            st.line_chart(
                data=unbinned_lineplot_data,
                x=unbinned_lineplot_data.columns[0],
                y=unbinned_lineplot_data.columns[1],
                x_label=None,
                y_label=st.session_state.curr_original_unbinned,
                color=None,
                width="stretch",
                height="content",
                use_container_width=None,
            )

    with pre_bin_relationships:
        if st.session_state.binned_exists_as is not None:
            st.subheader('Pre Bin Relationships', divider='grey', anchor=False, text_alignment='center')
            st.markdown("Metrics Before Binning")
            pre_combined = coco.column_comparison(
                st.session_state.data,
                numnum_meth_alpha_above=st.session_state.numnum_metrics,
                numcat_meth_alpha_above=st.session_state.catnum_metrics,
                catcat_meth_alpha_above=None,
                numeric_columns=None,
                categoric_columns=None,
                numeric_target=st.session_state.curr_original_unbinned,
                categoric_target=None,
            )
            pre_combined = pre_combined.loc[
                (
                    (pre_combined['column_a'] == st.session_state.curr_original_unbinned)
                    | (pre_combined['column_b'] == st.session_state.curr_original_unbinned)
                )
                & (pre_combined['column_a'] != pre_combined['column_b'])
                & ~(
                    (pre_combined['column_a'] + " -> Binned" == pre_combined['column_b'])
                    | (pre_combined['column_a'] == pre_combined['column_b'] + " -> Binned")
                )
            ].round(3)
            pre_target_on_right = pre_combined['column_b'] == st.session_state.curr_original_unbinned
            pre_combined.loc[pre_target_on_right, ['column_a', 'column_b']] = (
                pre_combined.loc[pre_target_on_right, ['column_b', 'column_a']].to_numpy()
            )
            pre_combined = pre_combined.sort_values(by=['column_b'], ascending=[True]).drop_duplicates(subset=['column_a', 'column_b'], keep='first').reset_index(drop=True)
            st.dataframe(
                data=pre_combined[[col for col in pre_combined if col in ['column_b', 'P-value', 'Coefficient']]],
                width="stretch",
                height="auto",
                use_container_width=None,
                hide_index=None,
                column_order=None,
                column_config=None,
                key=None,
                on_select="ignore",
                selection_mode="multi-row",
                row_height=None,
                placeholder=None,
            )

    with post_binned_countplot_cell:
        if st.session_state.binned_exists_as is not None:
            st.subheader('Post Bin Plot', divider='grey', anchor=False, text_alignment='center')
            st.markdown("Count Plot")
            binned_countplot_data = (
                st.session_state.data[st.session_state.binned_exists_as]
                .to_frame()
                .groupby(st.session_state.binned_exists_as, as_index=False, observed=True)
                .size()
                .sort_values(by='size', ascending=False)
                .reset_index(drop=True)
            )
            st.bar_chart(
                data=binned_countplot_data,
                x=binned_countplot_data.columns[0],
                y=binned_countplot_data.columns[1],
                x_label=binned_countplot_data.columns[0],
                y_label="Counts",
                color=None,
                horizontal=False,
                sort=True,
                stack=None,
                width="stretch",
                height="content",
                use_container_width=None,
            )

    with post_binned_relationships:
        if st.session_state.binned_exists_as is not None:
            st.subheader('Post Bin Relationships', divider='grey', anchor=False, text_alignment='center')
            st.markdown("Statistically Significant Relationships After Binning")
            post_combined = coco.column_comparison(
                st.session_state.data,
                numnum_meth_alpha_above=st.session_state.numnum_metrics,
                numcat_meth_alpha_above=st.session_state.catnum_metrics,
                catcat_meth_alpha_above=None,
                numeric_columns=None,
                categoric_columns=None,
                numeric_target=st.session_state.binned_exists_as,
                categoric_target=None,
            )
            post_combined = post_combined.loc[
                (
                    (post_combined['column_a'] == st.session_state.binned_exists_as)
                    | (post_combined['column_b'] == st.session_state.binned_exists_as)
                )
                & (post_combined['column_a'] != post_combined['column_b'])
                & ~(
                    (post_combined['column_a'] + " -> Binned" == post_combined['column_b'])
                    | (post_combined['column_a'] == post_combined['column_b'] + " -> Binned")
                )
            ].round(3)
            target_on_right = post_combined['column_b'] == st.session_state.binned_exists_as
            post_combined.loc[target_on_right, ['column_a', 'column_b']] = (
                post_combined.loc[target_on_right, ['column_b', 'column_a']].to_numpy()
            )
            post_combined = post_combined.sort_values(by=['column_b'], ascending=[True]).drop_duplicates(subset=['column_a', 'column_b'], keep='first').reset_index(drop=True)
            st.dataframe(
                data=post_combined[[col for col in post_combined if col in ['column_b', 'P-value', 'Coefficient']]],
                width="stretch",
                height="auto",
                use_container_width=None,
                hide_index=None,
                column_order=None,
                column_config=None,
                key=None,
                on_select="ignore",
                selection_mode="multi-row",
                row_height=None,
                placeholder=None,
            )

    binned_mu_plot = binned_mu_plot[0]
    with binned_mu_plot:
        if st.session_state.binned_exists_as is not None:
            st.markdown("Bin Means")
            muest = MuEstimator()
            muest.get_floating_mean_hbar(
                st.session_state.data,
                st.session_state.curr_original_unbinned,
                0.95,
                [st.session_state.binned_exists_as],
                plot_title=None,
                median=False,
                streamlit=True,
            )

    r_mu_and_relation_plot = st.columns([1], gap="small", vertical_alignment="bottom", border=True, width="stretch")

    r_mu_and_relation_plot = r_mu_and_relation_plot[0]
    with r_mu_and_relation_plot:
        if st.session_state.binned_exists_as is not None:
            mu_column, partition_column = st.session_state.curr_original_unbinned + " -> Binned", None
            muest2 = MuEstimator()
            muest2.get_floating_proportion_hbar(
                st.session_state.data,
                mu_column,
                0.95,
                partition_column,
                plot_title=None,
                streamlit=True,
                proportion_within_partition=True,
            )



if st.session_state.page == "Data Upload & Processing":
    

    # Sidebar for threshold adjustments
    with st.sidebar:

        # Page navigation #
        st.header("Navigation")        
        st.session_state.page = st.radio("Navigate", 
                                         NAVIGATION_PAGES, 
                                         index=NAVIGATION_PAGES.index(st.session_state.page),
                                         key = 'navigation_control',
                                         on_change = set_page)

        st.header("Threshold Adjustments")
        
        # Correlation thresholds for numeric-numeric
        current_corr = st.session_state.numnum_meth_alpha_above_instructions[0][1]
        corr_options = [current_corr,0.7,0.8,0.9]
        corr_labels = [f"{current_corr}"] + [str(i) for i in corr_options[1:]]
        selected_corr = st.selectbox("Correlation Threshold (Numeric-Numeric)", corr_labels, index=0)
        if selected_corr != corr_labels[0]:
            new_val = corr_options[corr_labels.index(selected_corr)]
            for item in st.session_state.numnum_meth_alpha_above_instructions:
                item[1] = new_val
        
        
        # P-value threshold for numeric-categoric
        current_p_numcat = st.session_state.numcat_meth_alpha_above_instructions[0][1]
        p_numcat_options = [current_p_numcat, 0.025, 0.01]
        p_numcat_labels = [f"{current_p_numcat}"] + [str(i) for i in p_numcat_options[1:]]
        selected_p_numcat = st.selectbox("P-value Threshold (Numeric-Categoric)", p_numcat_labels, index=0)
        if selected_p_numcat != p_numcat_labels[0]:
            new_val = p_numcat_options[p_numcat_labels.index(selected_p_numcat)]
            for item in st.session_state.numcat_meth_alpha_above_instructions:
                item[1] = new_val
        
        # P-value threshold for categoric-categoric
        current_p_catcat = st.session_state.catcat_meth_alpha_above_instructions[0][1]
        p_catcat_options = [current_p_catcat, 0.025, 0.01]
        p_catcat_labels = [f"{current_p_catcat}"] + [str(i) for i in p_catcat_options[1:]]
        selected_p_catcat = st.selectbox("P-value Threshold (Categoric-Categoric)", p_catcat_labels, index=0)
        if selected_p_catcat != p_catcat_labels[0]:
            new_val = p_catcat_options[p_catcat_labels.index(selected_p_catcat)]
            st.session_state.catcat_meth_alpha_above_instructions[0][1] = new_val
        
        # P-value threshold for good-of-fit uniform
        current_p_gof = st.session_state.good_of_fit_uniform_test_instructions[0]
        p_gof_options = [current_p_gof, 0.025, 0.01]
        p_gof_labels = [f"{current_p_gof}"] + [str(i) for i in p_gof_options[1:]]
        selected_p_gof = st.selectbox("P-value Threshold (Good-of-Fit Uniform)", p_gof_labels, index=0)
        if selected_p_gof != p_gof_labels[0]:
            new_val = p_gof_options[p_gof_labels.index(selected_p_gof)]
            st.session_state.good_of_fit_uniform_test_instructions[0] = new_val
        
        # P-value threshold for normal test
        current_p_norm = st.session_state.normal_test_instructions[0]
        p_norm_options = [current_p_norm, 0.025, 0.01]
        p_norm_labels = [f"{current_p_norm}"] + [str(i) for i in p_norm_options[1:]]
        selected_p_norm = st.selectbox("P-value Threshold (Normal Test)", p_norm_labels, index=0)
        if selected_p_norm != p_norm_labels[0]:
            new_val = p_norm_options[p_norm_labels.index(selected_p_norm)]
            st.session_state.normal_test_instructions[0] = new_val

        # decide to isolate partitioning super-subcat pairs
        #curr_isolate = st.session_state.supercat_subcat_params["isolate_super_subs"]
        #isolate_options = [curr_isolate, not curr_isolate]
        #selected_isolate = st.selectbox("Isolate Partitioning Super-Subcategory Counterparts",isolate_options,index=0)
        #if selected_isolate != st.session_state.supercat_subcat_params['isolate_super_subs']:
            #st.session_state.supercat_subcat_params.update({'isolate_super_subs':selected_isolate })
      
        if st.session_state.kruskal_assumption_check_params['full_pseudo']==False:
            # return pseudo{'full_pseudo':False}
            curr_ret_p = st.session_state.kruskal_assumption_check_params['return_pseudo']
            retp_options = [curr_ret_p, not curr_ret_p]
            selected_retp = st.selectbox("Test Similarity Instead When Strict Mean Assumptions Aren't Met\nIn Kruskal-Wallis Test",retp_options,index=0)
            if selected_retp != st.session_state.kruskal_assumption_check_params['return_pseudo']:
                st.session_state.kruskal_assumption_check_params.update({'return_pseudo':selected_retp })
      
        # return pseudo
        curr_ps = st.session_state.kruskal_assumption_check_params['full_pseudo']
        ps_options = [curr_ps, not curr_ps]
        selected_ps = st.selectbox("Only Test Distribution Similarity. Not Mean\nIn Kruskal-Wallis Test",ps_options,index=0)
        if selected_ps != st.session_state.kruskal_assumption_check_params['full_pseudo']:
            st.session_state.kruskal_assumption_check_params.update({'full_pseudo':selected_ps})
        






    # LOAD, PROCESS, AND FIT THE  DATA
    #================================================================================================================================================================= 
    #=================================================================================================================================================================

    # CSV File Upload and Processing
    st.title("Data Analyzer App")
# UI
# ======================================================================================================================================
# ======================================================================================================================================
# ======================================================================================================================================

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], help="File size limit: 10MB")
    with col2:
        if st.button("Use Default Dataset"):
            st.session_state.use_default_dataset = True
            st.rerun()

# ======================================================================================================================================
# ======================================================================================================================================
# ======================================================================================================================================

    if uploaded_file is not None or st.session_state.use_default_dataset:

        # Determine which file to process
        if uploaded_file is not None:
            file_to_process = uploaded_file
            file_identifier = uploaded_file.name
            st.session_state.use_default_dataset = False
        else:
            file_to_process = st.session_state.default_filepath
            file_identifier = "shopping_behavior_updated.csv"

        #   multipliers to help name a dtype
        st.session_state.min_pct_non_null_to_propose_a_dtype                        = 0.99
        st.session_state.max_pct_unique_for_numtype_cattype_threshold               = 0.005 # over for num <= for cat
        st.session_state.max_date_cols_used                                         = 5
        st.session_state.max_unique_pct_of_total_ie_identifier_variable_ID          = 0.2 # percentage of all observations that are unique to a columns, such as ID number or unique locations
        st.session_state.propose_drop_frac_nan_for_num                              = 0.95
        st.session_state.max_frac_droppable_nan_cat                                 = 0.1 # nans over this frac will be converted to str(NAN)


        # Check file size (10MB limit for fly.io) - only for uploaded files
        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue())
            if file_size > 10 * 1024 * 1024:
                st.error("File size exceeds 10MB limit. Please upload a smaller file.")
                file_to_process = None
        
        if file_to_process is not None:

            try:

                if 2 < st.session_state.step <= 7:
                    # display a sample
                    n_sample_rows = 3
                    st.markdown(f'{n_sample_rows} sample rows of values')
                    st.dataframe(st.session_state.data.sample(min(n_sample_rows, len(st.session_state.data))))
                if st.session_state.step == 7:
                    st.markdown(f'Hypothesis and Correlation Results')
                    st.dataframe(st.session_state.result_df)


                # Read CSV to DataFrame
                # Only re-read CSV when a new file is uploaded; preserve session state data across reruns
                if st.session_state.uploaded_file_name != file_identifier:

                    # reset dynamic ploting variables
                    st.session_state.group_plot_options = None
                    st.session_state.target_plot_options = None

                    st.session_state.mutated_group_plot_titles = None
                    st.session_state.mutated_target_plot_titles = None

                    st.session_state.plottable_group_indexes = None
                    st.session_state.plottable_target_indexes = None

                    st.session_state.plot_function = None

                    st.session_state.curr_target_selection = None
                    st.session_state.curr_target_plot_lists = None
                    reset_binning_state()
                    

                    st.session_state.step = 1



                    st.session_state.uploaded_file_name = file_identifier
                    st.session_state.data = pd.read_csv(file_to_process)   
                    if len(st.session_state.data.shape)<2:
                        st.session_state.data = st.session_state.data.to_frame()
                    # capture original len and shape
                    st.session_state.original_len = len(st.session_state.data)
                    st.session_state.original_shape = st.session_state.data.shape

                    st.session_state.step = 2

                    # Display which dataset is active
                    if uploaded_file is None:
                        st.info("📊 Using default dataset: shopping_behavior_updated.csv")

                    # display a sample
                    st.markdown('3 sample rows of existing values')
                    st.dataframe(st.session_state.data.sample(min(3, len(st.session_state.data))))

                if  st.session_state.step==2:
                
                    # Drop nulls
                    # DROPNA NOT REQUIRED. MODULE WILL HANDLE NAN INTERNALLY
                    #st.session_state.data = st.session_state.data.dropna()

                    n_rows = len(st.session_state.data)

                    # store cols and meta
                    cols_and_meta = {}
                    # Robust datatype detection
                    date_columns = []
                    native_date_cols = []
                    for col in st.session_state.data.columns:
                        proposed_dtype = None
                        col_dtype      = st.session_state.data[col].dtype
                        if str(col_dtype).startswith('str'):
                            st.session_state.data[col]=st.session_state.data[col].astype('category')
                            col_dtype      = st.session_state.data[col].dtype
                        niq            = st.session_state.data[col].nunique()
                        nan            = st.session_state.data[col].isna().sum()
                        n_non_nan_rows = n_rows - nan

                    
                        if str(col_dtype).startswith('datetime') or str(col_dtype).startswith('timedelta'):
                            date_columns.append(col)
                            native_date_cols.append(col)
                            proposed_dtype = 'datetime'
                        # Try to detect dates/datetimes
                        elif str(col_dtype) in ('object','category'):
                            test_as_dt = pd.to_datetime(st.session_state.data[col].copy(), errors='coerce').notna().sum()

                            # if st.session_state.min_pct_non_null_to_propose_a_dtype% convert to datetime it can be datetime
                            if (test_as_dt >= (st.session_state.min_pct_non_null_to_propose_a_dtype * n_non_nan_rows)):
                                date_columns.append(col)
                                proposed_dtype = 'datetime'

                            else:
                                # attempt to parse as numeric
                                test_as_num = pd.to_numeric(st.session_state.data[col].copy(), errors='coerce').notna().sum()

                                # if >st.session_state.min_pct_non_null_to_propose_a_dtype% are valid values and over st.session_state.max_pct_unique_for_numtype_cattype_threshold are unique, it can be int or float
                                if (test_as_num>(n_non_nan_rows*st.session_state.min_pct_non_null_to_propose_a_dtype)) and (niq>(n_non_nan_rows*st.session_state.max_pct_unique_for_numtype_cattype_threshold)):
                                    if (pd.to_numeric(st.session_state.data[col].copy(), errors='coerce').dropna() % 1 != 0).any():
                                        proposed_dtype = 'float64'
                                    else:
                                        proposed_dtype = 'int64'
                                else:
                                    proposed_dtype = 'category'

                        elif str(col_dtype).startswith('int') or str(col_dtype).startswith('float') or str(col_dtype).startswith('UInt'):
                            # if not over st.session_state.max_pct_unique_for_numtype_cattype_threshold are unique, it will be categoric
                            if (niq>(n_non_nan_rows*st.session_state.max_pct_unique_for_numtype_cattype_threshold)):
                                if (st.session_state.data[col].dropna() % 1 != 0).any():
                                    proposed_dtype = 'float64'
                                else:
                                    proposed_dtype = 'int64'
                            else:
                                proposed_dtype = 'category'

                        cols_and_meta[col] = [col_dtype,
                                            proposed_dtype,
                                            niq,
                                            False,
                                            nan,
                                            False,
                                            n_non_nan_rows]

                    index = [
                            'original_dtype', 
                            'possible_change',
                            'n_unique', 
                            'accept_change', 
                            'n_nan', 
                            'drop',
                            'non_nan_observations'
                            ]
                    st.session_state.datatypes_df               = pd.DataFrame(cols_and_meta, index=index)

                    st.session_state.step = 3


                if st.session_state.step == 3:


                    # Restore detection results from session state on reruns
                    n_rows                     = len(st.session_state.data)


                    # display datatypes as df
                    # transpose to make columns as index and [ 'original_dtype', 'possible_change', 'n_unique', 'accept_change', 'n_nan', 'drop']
                    # then add columns: 'pct_unique', 'pct_nan', 'nan_to_NAN_if_cat'
                    st.session_state.indexed_col_meta_df               = st.session_state.datatypes_df.copy().T
                    
                    # KEEP columns with (only 1 variable) or (1 var plus OVER/UNDER cat threshold for nan to str(NAN)) put others in auto_drop and remove from st.session_state.indexed_col_meta_df
                    id_mask                     = (
                                                (st.session_state.indexed_col_meta_df['n_unique']>1) 
                                                |
                                                ( (st.session_state.indexed_col_meta_df['n_unique']==1) 
                                                &
                                                (st.session_state.indexed_col_meta_df['n_nan']>(n_rows*st.session_state.max_frac_droppable_nan_cat)))
                                                )
                    
                    auto_drop                   = st.session_state.indexed_col_meta_df.loc[~id_mask].index  
                    st.header("Drop Columns")
                    # drop from data and from st.session_state.indexed_col_meta_df, to keep them out of the process 
                    if len(auto_drop)>0:
                        st.info(f"Some columns have only one unique value and will be dropped by default:  {[i for i in auto_drop]}")
                        st.session_state.indexed_col_meta_df           = st.session_state.indexed_col_meta_df.drop(index=auto_drop)
                        st.session_state.data                          = st.session_state.data.drop(columns=auto_drop)
                        st.session_state.datatypes_df                  = st.session_state.datatypes_df.drop(columns=auto_drop)

                    # colwise analysis
                    # WAYS TO DROP: 
                    #              ID COL; 
                    #              TOO MANY NAN, 
                    #              TOO FEW UNIQUE
                    # pct nunique/df size
                    st.session_state.indexed_col_meta_df['pct_unique_of_non_nan']    = st.session_state.indexed_col_meta_df['n_unique']/st.session_state.indexed_col_meta_df['non_nan_observations']
                    # pct nan
                    st.session_state.indexed_col_meta_df['pct_nan']       = st.session_state.indexed_col_meta_df['n_nan']/n_rows
                    # proposed and existing are both cat | both number
                    id_orig_str                     = st.session_state.indexed_col_meta_df['original_dtype'].copy().astype(str)
                    id_poss_str                     = st.session_state.indexed_col_meta_df['possible_change'].copy().astype(str)

                    orig_date                       = ((id_orig_str.str.startswith('date')) | (id_orig_str.str.startswith('time')))
                    orig_cat                        = (id_orig_str.isin(['category','object']))
                    orig_num                        = (id_orig_str.str[:3].isin(['UIn','flo','int']))
                    orig_neither                    = ( ~( orig_cat) & ~(orig_num) & ~(orig_date) )

                    poss_date                       = ((id_poss_str.str.startswith('date')) | (id_poss_str.str.startswith('time')))
                    poss_cat                        = (id_poss_str.isin(['category','object'])) 
                    poss_num                        = (id_poss_str.str[:3].isin(['UIn','flo','int']))
                    poss_neither                    = ( ~(poss_cat) & ~(poss_num) & ~(poss_date))

                    one_not_recognized              = (poss_neither | orig_neither)
                    not_the_same                    = ( 
                                                        (poss_date & ~(orig_date)) | (orig_date & ~(poss_date))
                                                        |
                                                        (poss_cat & ~(orig_cat)) | (orig_cat & ~(poss_cat))
                                                        |
                                                        (poss_num & ~(orig_num)) | (orig_num & ~(poss_num))
                                                        |
                                                        (poss_neither & ~(orig_neither)) | (orig_neither & ~(poss_neither))
                                                       )
                    st.session_state.one_not_recognized = one_not_recognized
                    st.session_state.not_the_same = not_the_same

                    # APPLY CHECK(S) FOR EACH
                    # threshold checks  
                    over_unique_id_threshold       = st.session_state.indexed_col_meta_df['pct_unique_of_non_nan'] > st.session_state.max_unique_pct_of_total_ie_identifier_variable_ID
                    excessiv_nan_for_numeric       = st.session_state.indexed_col_meta_df['pct_nan']    > st.session_state.propose_drop_frac_nan_for_num         
                    # capture it now before rows are dropped and dtypes changed etcetera
                    st.session_state.indexed_col_meta_df['nan_to_NAN_if_cat']    = st.session_state.indexed_col_meta_df['pct_nan']    > st.session_state.max_frac_droppable_nan_cat



                    #explain thresholds to user
                    st.subheader('Data Manipulation')
                    st.markdown(f"Threshold 'pct_nan' to drop a column - %{round(100*st.session_state.propose_drop_frac_nan_for_num,2)}")
                    st.markdown(f"Threshold 'pct_nan' to convert NaN to string: 'NAN' - %{round(100*st.session_state.max_frac_droppable_nan_cat,2)}")
                    st.markdown(f"Threshold 'pct_unique_of_non_nan' to drop an identifier column - %{round(100*st.session_state.max_unique_pct_of_total_ie_identifier_variable_ID,2)}")
                    st.markdown(f"Threshold 'pct_unique_of_non_nan' to consider a type change - %{round(100*st.session_state.max_pct_unique_for_numtype_cattype_threshold,3)}")


                    # use np.select  to fill drop and cat choices
                    conditions = [
                                  excessiv_nan_for_numeric, # too many nans for numeric, date, or cat                                            Drop
                                  (st.session_state.indexed_col_meta_df['n_unique']<2) & ~(poss_cat | orig_cat), # no hope to save by replacing nan w NAN               DROP,
                                  (st.session_state.indexed_col_meta_df['n_unique']<2) & poss_cat & st.session_state.indexed_col_meta_df['nan_to_NAN_if_cat'], #    try changeing it to 2 variables     CHANGE
                                  over_unique_id_threshold & orig_cat & poss_cat, # too many unique values: adding 1 NAN probably wouldn't help anyways                DROP
                                  st.session_state.indexed_col_meta_df['nan_to_NAN_if_cat'] & poss_cat & orig_num & (st.session_state.indexed_col_meta_df['n_unique']<=60),# convert nan to str(NAN) 60 plots well   CHANGE
                                  poss_date & ~(orig_date),   #              might be date                                                            CHANGE
                                  over_unique_id_threshold & orig_cat & ~(poss_cat), # better chance with not cat                                               CHANGE
                                ]

                    # choices           
                    drop_col   = [True,  True,  False, True,  False, False, False]
                    change_col = [False, False, True,  False, True,  True,  True ]

                    # call np.select in the df, conditions and choices to create 'drop' and 'make_changes' columns
                    st.session_state.indexed_col_meta_df['drop'] = np.select(conditions, drop_col, default=False)
                    st.session_state.indexed_col_meta_df['make_change'] = np.select(conditions, change_col, default=False)


                    st.session_state.datatypes_df = None

                    st.session_state.step = 4

                    st.rerun()

                if st.session_state.step ==4:
           
                    # filter columns and sort based on 'drop' and 'make_change' columns
                    st.session_state.ui_df = (st.session_state.indexed_col_meta_df.copy().loc[( 
                                                        st.session_state.one_not_recognized |
                                                        st.session_state.not_the_same |
                                                        st.session_state.indexed_col_meta_df['drop'].copy() |
                                                        st.session_state.indexed_col_meta_df['make_change'].copy() |
                                                        (st.session_state.indexed_col_meta_df['pct_unique_of_non_nan']>.8)
                                                        )][['original_dtype', 'possible_change', 
                                                            'make_change', 'drop', 
                                                            'pct_unique_of_non_nan','pct_nan', 
                                                            'n_unique','n_nan']]
                                                        .sort_values(by=['make_change','drop'],ascending=False))



# ---------------------------------------------------UI
# ======================================================================================================================================
# ======================================================================================================================================
# ======================================================================================================================================
                    
                    
                    try:
                        drop_cols_one = st.segment_control("Select Any Column to Drop", 
                                                        list(set(st.session_state.data.columns)-set(st.session_state.ui_df.index)),
                                                        selection_mode="multi",
                                                        default = None)
                    except:
                        drop_cols_one = st.multiselect("Select Any Column to Drop", 
                                                        list(set(st.session_state.data.columns)-set(st.session_state.ui_df.index)),
                                                        default = None)

                    st.session_state.user_mutated_data = st.data_editor(st.session_state.ui_df)

                    procede_w_changes_and_drops = st.button('Submit Changes')

                    if procede_w_changes_and_drops:

# ======================================================================================================================================
# ======================================================================================================================================
# ======================================================================================================================================


                        # drop columns
                        cols_to_drop = st.session_state.user_mutated_data.loc[st.session_state.user_mutated_data['drop']].index
                        st.session_state.user_mutated_data = st.session_state.user_mutated_data.drop(index=cols_to_drop)
                        if drop_cols_one:
                            if isinstance(drop_cols_one,str):
                                drop_cols_one = [drop_cols_one]
                            cols_to_drop = list(cols_to_drop) + drop_cols_one
                        st.session_state.data              = st.session_state.data.drop(columns=cols_to_drop)
                        st.session_state.indexed_col_meta_df = st.session_state.indexed_col_meta_df.drop(index=cols_to_drop)

                        

                        # change types and update current: original_dtype
                        datatype_changes = ( st.session_state.user_mutated_data
                                            .loc[st.session_state.user_mutated_data['make_change']][['possible_change']]
                                            .reset_index(drop=False)
                                            .values
                                            )
                        for col_name, new_datatype in datatype_changes:
                            st.session_state.data[col_name] = st.session_state.data[col_name].astype(new_datatype,errors='ignore')
                        st.session_state.user_mutated_data['original_dtype'] = st.session_state.user_mutated_data.apply(lambda x: x['possible_change'] if x['make_change'] else x['original_dtype'],axis=1)

                        # create a df that includes columns not in the UI as index, 'original_dtype' as values
                        never_displayed_index  =   list( set( st.session_state.indexed_col_meta_df.index ) - set( st.session_state.user_mutated_data.index ) )
                        transition_df      = st.session_state.indexed_col_meta_df.copy().loc[never_displayed_index][['original_dtype']]
                        st.session_state.mutated_dtypes   =    pd.concat([transition_df, st.session_state.user_mutated_data[['original_dtype']].copy()])


                        # fillna over all columns that meet the condition
                        st.session_state.mutated_dtypes.columns = ['curr_dtype']
                        st.session_state.indexed_col_meta_df    = pd.merge(st.session_state.indexed_col_meta_df,
                                                                           st.session_state.mutated_dtypes,
                                                                           how = 'inner',
                                                                           left_index=True,
                                                                           right_index=True
                                                                           )
                        st.session_state.indexed_col_meta_df['dtype_as_string'] = st.session_state.indexed_col_meta_df['curr_dtype'].astype(str)

                        fillna_columns = (st.session_state.indexed_col_meta_df
                                            .loc[( 
                                                 (st.session_state.indexed_col_meta_df['dtype_as_string'].isin(['category','object'])) 
                                                 & 
                                                 (st.session_state.indexed_col_meta_df['nan_to_NAN_if_cat'])
                                                 )]
                                            .index)                                            
                        for column in fillna_columns:
                            st.session_state.data[column] = np.where(st.session_state.data[column].isna(),'NAN',st.session_state.data[column])

                        
                        
                        st.session_state.step = 5

                        st.session_state.ui_df = None

                        st.session_state.user_mutated_data = None                
                        
                        if 0 < len(st.session_state.data.shape) < 2:
                            st.session_state.data = st.session_state.data.to_frame()
                        if st.session_state.data.shape[1] == 0:
                            st.warning("⚠️ All columns have been dropped. No data to process.")
                            st.info("💡 Upload a new file or reload the page to start over.")
                            st.session_state.early_stop = True
                            st.session_state.step = 1
                            st.stop()  # ✅ Clean halt with clear next steps



                        if not st.session_state.early_stop:
                            st.rerun()
                        
                        

                if st.session_state.step == 5:                        

                    # capture date type cols
                    is_date = (
                                (   st.session_state.indexed_col_meta_df['dtype_as_string'].str.startswith('date')    ) 
                                | 
                                (   st.session_state.indexed_col_meta_df['dtype_as_string'].str.startswith('time')   )
                                )
                    date_columns = list(st.session_state.indexed_col_meta_df.loc[is_date].index)

                    if date_columns:
                        # DATES
                        
                        st.header("Part Dates\nEg: 'Year', 'Quarter', 'Month', ...\nOr Drop Dates")
                        # side by side buttons
                        procede_w_date_col_left,   drop_dates_col_right   =   st.columns([1,2])

                        # buttons to choose to 'part' dates or drop them
                        procede_w_date,     drop_dates   = None, None
                        if (not procede_w_date) and (not drop_dates):
                            st.info(f"If 'Part Dates' is selected, a max of {st.session_state.max_date_cols_used} will be automatically chosen.")
                        with procede_w_date_col_left:
                            procede_w_date = st.button("Part Dates")
                        with drop_dates_col_right:
                            drop_dates = st.button("Drop Dates")
                        # if either button is pressed
                        if procede_w_date or drop_dates:
                            # drop w/o processing
                            if drop_dates:
                                st.session_state.indexed_col_meta_df = st.session_state.indexed_col_meta_df.drop(index=date_columns)
                                st.session_state.data                = st.session_state.data.drop(columns=date_columns)
                            # or process then drop
                            elif procede_w_date:      

                                # Limit number of additional columns with st.session_state.max_date_cols_used = None date columns
                                cols_to_drop = []
                                if len(date_columns) > st.session_state.max_date_cols_used:
                                    # Keep first 2, drop others
                                    cols_to_drop = date_columns[st.session_state.max_date_cols_used:]
                                    date_columns = date_columns[:st.session_state.max_date_cols_used]

                                # extract useful info from date/datetime columns
                                
                                # Create categorical variables from dates
                                for col in date_columns:
                                    curr_cols = set(st.session_state.data.columns)
                                    try:
                                        new_title = f'{col}_year'
                                        if new_title in curr_cols:
                                            new_title=new_title+'_i'
                                        while new_title in curr_cols:
                                            new_title = new_title + "i"
                                        st.session_state.data[new_title] = st.session_state.data[col].dt.year.astype('category')
                                        if st.session_state.data[new_title].nunique()<=1: st.session_state.data = st.session_state.data.drop(columns=new_title)
                                    except:
                                        pass
                                    try:
                                        new_title = f'{col}_quarter'
                                        if new_title in curr_cols:
                                            new_title=new_title+'_i'
                                        while new_title in curr_cols:
                                            new_title = new_title + "i"
                                        st.session_state.data[new_title] = st.session_state.data[col].dt.quarter.astype('category')
                                        if st.session_state.data[new_title].nunique()<=1: st.session_state.data = st.session_state.data.drop(columns=new_title)
                                    except:
                                        pass
                                    try:
                                        new_title = f'{col}_month'
                                        if new_title in curr_cols:
                                            new_title=new_title+'_i'
                                        while new_title in curr_cols:
                                            new_title = new_title + "i"
                                        st.session_state.data[new_title] = st.session_state.data[col].dt.month.astype('category')
                                        if st.session_state.data[new_title].nunique()<=1: st.session_state.data = st.session_state.data.drop(columns=new_title)
                                    except:
                                        pass
                                    try:
                                        new_title = f'{col}_day'
                                        if new_title in curr_cols:
                                            new_title=new_title+'_i'
                                        while new_title in curr_cols:
                                            new_title = new_title + "i"
                                        st.session_state.data[new_title] = st.session_state.data[col].dt.day.astype('category')
                                        if st.session_state.data[new_title].nunique()<=1: st.session_state.data = st.session_state.data.drop(columns=new_title)
                                    except:
                                        pass
                                    try:
                                        new_title = f'{col}_hour'
                                        if new_title in curr_cols:
                                            new_title=new_title+'_i'
                                        while new_title in curr_cols:
                                            new_title = new_title + "i"
                                        st.session_state.data[new_title] = st.session_state.data[col].dt.hour.astype('category')
                                        if st.session_state.data[new_title].nunique()<=1: st.session_state.data = st.session_state.data.drop(columns=new_title)
                                    except:
                                        pass
                                    try:
                                        new_title = f'{col}_30min'
                                        if new_title in curr_cols:
                                            new_title=new_title+'_i'
                                        while new_title in curr_cols:
                                            new_title = new_title + "i"
                                        st.session_state.data[new_title] = (st.session_state.data[col].dt.minute // 30).astype('category')
                                        if st.session_state.data[new_title].nunique()<=1: st.session_state.data = st.session_state.data.drop(columns=new_title)
                                    except:
                                        pass


                                # Drop original date columns (full date/datetime only, not extracted parts like year/month/day)
                                date_columns = date_columns + cols_to_drop
                                if date_columns:
                                    st.session_state.data = st.session_state.data.drop(columns=date_columns,errors='ignore')
                                    if len(st.session_state.data.shape)<2:
                                        st.session_state.data = st.session_state.data.to_frame()
                                    
                                    st.session_state.indexed_col_meta_df = st.session_state.indexed_col_meta_df.drop(index=date_columns)
                        st.session_state.step = 6
                        st.rerun()
                    else:
                        st.session_state.step = 6
                        st.rerun()

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Please upload a CSV file or click 'Use Default Dataset' to begin analysis.")

    if st.session_state.step==6:
        fit = st.button("Fit the Data")

        if fit:
            st.caption("Fitting the Data Now. Please Wait.")
            st.session_state.AD = AnalyzeDataset(multivariate_params = st.session_state.multivariate_params,
                                                    kruskal_assumption_check_params=st.session_state.kruskal_assumption_check_params,
                                                    anova_assumption_check_params = st.session_state.anova_assumption_check_params,
                                                    chi2_assumption_check_params = st.session_state.chi2_assumption_check_params,
                                                    supercat_subcat_params = st.session_state.supercat_subcat_params,
                                                    multivariate_concatenation_delimiter = st.session_state.multivariate_concatenation_delimiter,
                                                    numnum_meth_alpha_above_instructions=st.session_state.numnum_meth_alpha_above_instructions,
                                                    numcat_meth_alpha_above_instructions=st.session_state.numcat_meth_alpha_above_instructions,
                                                    catcat_meth_alpha_above_instructions=st.session_state.catcat_meth_alpha_above_instructions,
                                                    good_of_fit_uniform_test_instructions=st.session_state.good_of_fit_uniform_test_instructions,
                                                    normal_test_instructions=st.session_state.normal_test_instructions
                                                    )
            st.session_state.AD.fit_full_dataset_analysis(st.session_state.data,  
                                numeric_columns=None,         # None for autodetect
                                categoric_columns=None,       # None for autodetect
                                numeric_target=None,          # None to compute all numeric variables as targets
                                categoric_target=None,        # None to compute all categoric variables as targets                          
                                fit_good_of_fit=True,         # instruct to test categoric variables for uniform distribution
                                fit_normal=True,
                                fit_multivariates=False,       # instruct to test multivariate significance
                                fit_supercat_subcats=True)    # test for super categories with subcategories that partition other variables
            
            # this calls return_catcat_plottype and st.session_state.AD.produce_all_plots, so it is placed here 
            # after st.session_state.AD is fit and return_catcat_plottype is defined
            if ('plot_function' not in st.session_state) or (not st.session_state.plot_function):
                                st.session_state.plot_function = [
                                    px.histogram, px.histogram,        # 0,1  num_univar
                                    px.histogram, px.histogram,        # 2,3  cat_univar
                                    px.scatter, px.scatter,            # 4,5  numnum_bivar
                                    px.box, px.box,                    # 6,7  numcat_bivar
                                    return_catcat_plottype, return_catcat_plottype,  # 8,9  catcat_bivar
                                    st.session_state.AD.produce_all_plots,          # 10   super_subcat_pairs
                                    return_catcat_plottype, px.scatter, px.box,     # 11,12,13 assumptions not met
                                    px.histogram, px.histogram                      # 14,15 assumptions not met
                                ]


            result_df = st.session_state.AD.column_relationships_df(st.session_state.data.columns).reset_index(drop=False)
            st.session_state.result_df = result_df[['Target', 'Distribution', 'FeatureColumn(s)', 'Test(s)']].rename(columns={'Test(s)':'Test:Assumptions_Met'}).set_index(['Target','Distribution'])

            st.session_state.step = 7
            st.rerun()


        
    # PLOT THE DATA
    #================================================================================================================================================================= 
    #=================================================================================================================================================================
elif st.session_state.page in ["Group Visualizations", "Target Visualizations"]:


    # Sidebar for n_wide controls
    with st.sidebar:

        # Page navigation #
        st.header("Navigation")
        st.session_state.page = st.radio("Navigation", 
                                         NAVIGATION_PAGES, 
                                         index=NAVIGATION_PAGES.index(st.session_state.page),
                                         key = 'navigation_control',
                                         on_change = set_page)

        st.header("Plot Controls")
        
        # Number of plot axes per row
        chunk_size_options = [1,5,10]
        selected_chunk_size = st.selectbox("Number of plots per batch", chunk_size_options, index=chunk_size_options.index(st.session_state.chunk_size) if st.session_state.chunk_size in chunk_size_options else 0) 
        
        #st.header("Numeric-Numeric Bivariate Parameters")
        
        # plot_type for numnum_bivar_params
        #numnum_plot_type_options = ["joint", "scatter"]
        #current_numnum_plot_type = st.session_state.numnum_bivar_params.get('plot_type', 'joint')
        #selected_numnum_plot_type = st.selectbox("Plot type for numeric-numeric plots", numnum_plot_type_options, index=numnum_plot_type_options.index(current_numnum_plot_type) if current_numnum_plot_type in numnum_plot_type_options else 0)
        #st.session_state.numnum_bivar_params['plot_type'] = selected_numnum_plot_type
        
        # linreg for numnum_bivar_params
        #linreg_options = ["False", "True"]
        #current_linreg = "True" if st.session_state.numnum_bivar_params.get('linreg', False) else "False"
        #selected_linreg = st.selectbox("Include linear regression line", linreg_options, index=linreg_options.index(current_linreg))
        #st.session_state.numnum_bivar_params['linreg'] = selected_linreg 
        
        #st.header("Supercategory-Subcategory Parameters")
        
        # row_height for super_subcat_pairs_params
        #row_height_options = [2, 3, 4, 5]
        #current_row_height = st.session_state.super_subcat_pairs_params.get('row_height', 3)
        #selected_row_height = st.selectbox("Row height (inches) for supersubcat plots", row_height_options, index=row_height_options.index(current_row_height) if current_row_height in row_height_options else 1)
        #st.session_state.super_subcat_pairs_params['row_height'] = selected_row_height
        
        # cols_per_row for super_subcat_pairs_params
        #cols_per_row_options = [1, 2, 3, 4]
        #current_cols_per_row = st.session_state.super_subcat_pairs_params.get('cols_per_row', 2)
        #selected_cols_per_row = st.selectbox("Columns per row for subcategory plots", cols_per_row_options, index=cols_per_row_options.index(current_cols_per_row) if current_cols_per_row in cols_per_row_options else 1)
        #st.session_state.super_subcat_pairs_params['cols_per_row'] = selected_cols_per_row
        
        # y_tick_fontsize for super_subcat_pairs_params
        #y_tick_fontsize_options = [8, 10, 12, 14, 16]
        #current_y_tick_fontsize = st.session_state.super_subcat_pairs_params.get('y_tick_fontsize', 12)
        #selected_y_tick_fontsize = st.selectbox("Y-axis tick font size", y_tick_fontsize_options, index=y_tick_fontsize_options.index(current_y_tick_fontsize) if current_y_tick_fontsize in y_tick_fontsize_options else 2)
        #st.session_state.super_subcat_pairs_params['y_tick_fontsize'] = selected_y_tick_fontsize

  
    if (st.session_state.step < 7 or 
        st.session_state.data is None or 
        'AD' not in st.session_state or 
        st.session_state.AD is None):
        st.info("The Data Hasn't Been Fit.")
    else:
        # PLOT GROUPS
        # ---------------------------------------------------------------------------------------------------
        if st.session_state.page == "Group Visualizations":



        

            if ( 
                (st.session_state.group_plot_options is None)
                ):

                # unprocessed pairs list returned by model (coresponds to un mutated titles)
                var_list_of_iterable_plottable_group_header_pairs = [
                            list(st.session_state.AD.reject_null_normal),
                            list(st.session_state.AD.fail_to_reject_null_normal),
                            list(st.session_state.AD.reject_null_good_of_fit),
                            list(st.session_state.AD.fail_to_reject_null_good_of_fit),
                            st.session_state.AD.above_threshold_corr_numnum,
                            st.session_state.AD.below_threshold_corr_numnum,
                            st.session_state.AD.reject_null_numcat,
                            st.session_state.AD.fail_to_reject_null_numcat,
                            st.session_state.AD.reject_null_catcat,
                            st.session_state.AD.fail_to_reject_null_catcat,
                            st.session_state.AD.supercategory_subcategory_pairs,
                            st.session_state.AD.assumptions_not_met['catcat'],
                            st.session_state.AD.assumptions_not_met['numnum'],
                            st.session_state.AD.assumptions_not_met['numcat'],
                            list(st.session_state.AD.assumptions_not_met['num']),
                            list(st.session_state.AD.assumptions_not_met['cat']),
                            ]

                (
                    st.session_state.group_plot_options,
                    st.session_state.mutated_group_plot_titles,
                    not_present_titles,
                    st.session_state.plottable_group_indexes
                ) = chunk_plotables_mutate_titles(
                    var_list_of_iterable_plottable_group_header_pairs,
                    st.session_state.unmutated_title_list,
                    st.session_state.chunk_size,
                )
                #if not_present_titles:
                    #st.markdown(f"None of:")
                    #for title in not_present_titles:
                        #st.markdown(f"     {title}")
                    #st.markdown("Are present in the data.")
                
     
            if ( 
                (st.session_state.group_plot_options)
                ):

                # create a list of mutated titles that matches the index and is within the plot selection gate
                option_titles = [st.session_state.mutated_group_plot_titles[i] for i in st.session_state.plottable_group_indexes]
                defa = 0
                if not option_titles:
                    st.session_state.curr_group_plot_selection = None
                    st.session_state.master_group_plot_index = None
                    st.session_state.seen = []
                    st.session_state.seen_for_plot_selection = None
                    st.info("No plottable groups found for the current dataset/settings.")
                    plot_selection = None
                else:
                    if st.session_state.curr_group_plot_selection not in option_titles:
                        st.session_state.curr_group_plot_selection = None
                        st.session_state.master_group_plot_index = None
                        st.session_state.seen = []
                        st.session_state.seen_for_plot_selection = None

                    # Default to first option if no prior selection
                    if st.session_state.group_plot_selection in option_titles:
                        defa = option_titles.index(st.session_state.group_plot_selection)
                    else:
                        defa = 0
                        # Initialize on first run
                        st.session_state.group_plot_selection = option_titles[0]
                        st.session_state.curr_group_plot_selection = option_titles[0]
                        
                try:
                    # Widget with callback - don't assign return value
                    st.segment_control("Select a Group to Plot", 
                                      option_titles,
                                      selection_mode="single",
                                      default=defa,
                                      on_change = group_plot,
                                      key = 'group_plot_control')
                except:
                    # Widget with callback - don't assign return value
                    st.selectbox("Select a Group to Plot", 
                                option_titles,
                                index=defa,
                                on_change = group_plot,
                                key = 'group_plot_control')

                ### capture the index. titles and plots match 
                ### option_titles  is based on st.session_state.mutated_group_plot_titles which was created with 
                ### and is indexed according to st.session_state.group_plot_options
                st.session_state.master_group_plot_index = st.session_state.mutated_group_plot_titles.index(st.session_state.group_plot_selection)
                selected_group_option = st.session_state.group_plot_options[st.session_state.master_group_plot_index]
                

                # use the index to retrieve the corresponding key for plot params
                # note that the plotly_express plots params use underscore. 
                param_key = st.session_state.plot_param_keys[st.session_state.master_group_plot_index]
                iterable_chunks_of_variables = selected_group_option

                # st.session_state.seen resets only when user changes the selected plot group.
                # st.session_state.seen_for_plot_selection is used to determine if the plot group is changed
                if st.session_state.get('seen_for_plot_selection') != st.session_state.group_plot_selection:
                    st.session_state.seen = []
                    st.session_state.seen_for_plot_selection = st.session_state.group_plot_selection
                seen = st.session_state.setdefault('seen', [])
                selected_partition = []
                tabs = []

                # plot one chunk at a time and keep track of which chunks have been plotted 
                if not iterable_chunks_of_variables:
                    st.info('No partitions available for the selected group.')
                else:
                    select_wid, seen_wid = st.columns([1, 1])
                    with select_wid:
                        # identify each chunk as index + 1
                        chunk_options = [i + 1 for i in range(len(iterable_chunks_of_variables))]
                        if not chunk_options:
                            st.session_state.group_selection = None
                        else:
                            # a number that represents a chunk. it is index = number - 1
                            st.session_state.group_selection = st.selectbox('Select a Partition Chunk to Plot', 
                                                            chunk_options, 
                                                            index=0,
                                                            on_change=group_chunk_selector,
                                                            key='group_chunk_select')
                    with seen_wid:
                        st.markdown(f'Seen Partitions: {sorted(seen)}')                         

                    if st.session_state.group_selection is None:
                        st.info('No selectable partition found for the selected group.')
                    else:
                        if st.session_state.group_selection not in seen:
                            seen.append(st.session_state.group_selection)

                        # select a chunk of pairs/univariates to plot
                        selected_partition = iterable_chunks_of_variables[st.session_state.group_selection - 1]

                        def _tab_title(variable_to_plot):
                            if isinstance(variable_to_plot, (list, tuple)):
                                return ' | '.join([str(v) for v in variable_to_plot])
                            return str(variable_to_plot)

                        tab_titles = [_tab_title(v) for v in selected_partition]
                        tabs = st.tabs(tab_titles) if tab_titles else []


                for tab, variable_to_plot in zip(tabs, selected_partition):
                    with tab:
                        # identify variable(s) type of plot numeric-categoric bivariate, numeric univariate etc
                        var_type_plot = st.session_state.plot_list_keys[st.session_state.master_group_plot_index]
                        #retrieve the appropriate plot function
                        plot_func_ = st.session_state.plot_function[st.session_state.master_group_plot_index]
                        # retrieve the plot parameters
                        curr_params = st.session_state.plot_params_dict[param_key].copy()
                        # process and plot one plot for each iteration in the chunk
                        plot_one_title(
                                data = st.session_state.data,
                                variable_to_plot=variable_to_plot,
                                curr_params=curr_params,  
                                var_type_plot=var_type_plot,
                                plot_func_=plot_func_)
                        



                
        # PLOT TARGETS
        # ---------------------------------------------------------------------------------------------------
    
        elif st.session_state.page == "Target Visualizations":

            try:
                variable_selection = st.segment_control("Select Variables to Plot", 
                                        list(st.session_state.data.columns),
                                        selection_mode="single",
                                        default=None)
            except:
                variable_selection = st.selectbox("Select a Variable", 
                                        list(st.session_state.data.columns),
                                        index=None)


            if variable_selection:
        
                if st.session_state.curr_target_selection != variable_selection:

                    st.session_state.target_plot_options = None

                    st.session_state.curr_target_selection = variable_selection

                    meta       = st.session_state.AD.target_key_feature_meta_vals[variable_selection]
                    is_numeric = meta['target_dtype']=='numeric'

                    def targ_first(list_of_variables):
                        return [[variable_selection, feat] for feat in list_of_variables]
                    def targ_second(list_of_variables):
                        return [[feat, variable_selection] for feat in list_of_variables]
                    
                    
                    st.session_state.curr_target_plot_lists = [

                        #'Numerical Non-Normal'
                        [variable_selection] if  is_numeric  and meta['is_normal_or_uniform']=='reject_normal'  else [],

                        #'Numerical Normal'
                        [variable_selection] if  is_numeric and meta['is_normal_or_uniform']=='fail_to_reject_normal' else [],

                        #'Categorical Non-Uniform'
                        [variable_selection] if not is_numeric and meta['is_normal_or_uniform']=='reject_uniform' else [],
                         
                        #'Categorical Uniform'
                        [variable_selection] if not is_numeric and meta['is_normal_or_uniform']=='fail_to_reject_uniform' else [],
                         
                        #'Numeric-Numeric With Correlation'
                        [] if not is_numeric else targ_first(meta['significant_numeric_relationships']),
                        
                        #'Numeric-Numeric Without Correlation'
                        [] if not is_numeric else targ_first(meta['not_significant_numerics']),
                        
                        #'Numeric-Categoric Reject Null'
                        targ_first(meta['significant_categoric_relationships']) if is_numeric else targ_second(meta['significant_numeric_relationships']),

                        #'Numeric-Categoric Fail to Reject Null'
                        targ_first(meta['not_significant_categorics']) if is_numeric else targ_second(meta['not_significant_numerics']),

                        #'Categoric-Categoric Reject Null'
                        [] if is_numeric else targ_first(meta['significant_categoric_relationships']),

                        #'Categoric-Categoric Fail to Reject Null'
                        [] if is_numeric else targ_first(meta['not_significant_categorics']),

                        #'Categoric Partitioned by Another Categoric'
                        targ_second(meta['paired_to_a_supercategory'])+targ_first(meta['paired_to_a_subcategory']),
                        
                        #'Assumptions Not Met - Categoric-Categoric'
                        [] if is_numeric else targ_first(list(meta['assumptions_not_met']['catcat'])),

                        # 'Assumptions Not Met - Numeric-Numeric'
                        [] if not is_numeric else targ_first(list(meta['assumptions_not_met']['numnum'])),

                        #'Assumptions Not Met - Numeric-Categoric'
                        targ_first(list(meta['assumptions_not_met']['numcat'])) if is_numeric else targ_second(list(meta['assumptions_not_met']['numcat'])),

                        #'Assumptions Not Met - Numerical'
                        [] if not is_numeric else [feat for feat in list(meta['assumptions_not_met']['num'])],

                        #'Assumptions Not Met - Categorical'
                        [] if is_numeric else [feat for feat in list(meta['assumptions_not_met']['cat'])],
                    ]
            
                if st.session_state.curr_target_selection == variable_selection:

                    if (
                        (not st.session_state.target_plot_options)
                    ):
                        (
                            st.session_state.target_plot_options,
                            st.session_state.mutated_target_plot_titles,
                            not_present_titles,
                            st.session_state.plottable_target_indexes
                        ) = chunk_plotables_mutate_titles(
                            st.session_state.curr_target_plot_lists,
                            st.session_state.unmutated_title_list,
                            st.session_state.chunk_size,
                        )
                        #if not_present_titles:
                            #st.markdown(f"None of:")
                            #for title in not_present_titles:
                                #st.markdown(f"     {title}")
                            #st.markdown("Are present in the data.")
                    
            
                    if ( 
                        (st.session_state.target_plot_options)
                        ):

                        # create a list of mutated titles that matches the index and is within the plot selection gate
                        option_titles = [st.session_state.mutated_target_plot_titles[i] for i in st.session_state.plottable_target_indexes]
                        defa = 0
                        if not option_titles:
                            st.session_state.curr_target_plot_selection = None
                            st.session_state.master_target_plot_index = None
                            st.session_state.seen_tar = []
                            st.session_state.tar_for_plot_selection = None
                            st.info("No plottable targets found for the current dataset/settings.")
                            plot_selection = None
                        else:
                            if st.session_state.curr_target_plot_selection not in option_titles:
                                st.session_state.curr_target_plot_selection = None
                                st.session_state.master_target_plot_index = None
                                st.session_state.seen_tar = []
                                st.session_state.tar_for_plot_selection = None

                            # Default to first option if no prior selection
                            if st.session_state.target_plot_selection in option_titles:
                                defa = option_titles.index(st.session_state.target_plot_selection)
                            else:
                                defa = 0
                                # Initialize on first run
                                st.session_state.target_plot_selection = option_titles[0]
                                st.session_state.curr_target_plot_selection = option_titles[0]
                                
                        try:
                            # Widget with callback - don't assign return value
                            st.segment_control("Select a Group to Plot", 
                                              option_titles,
                                              selection_mode = "single",
                                              default = defa,
                                              key = 'target_plot_control',
                                              on_change =target_plot)
                        except:
                            # Widget with callback - don't assign return value
                            st.selectbox("Select a Group to Plot", 
                                        option_titles,
                                        index = defa,
                                        key = 'target_plot_control',
                                        on_change = target_plot)

                        ### capture the index. titles and plots match 
                        ### option_titles  is based on st.session_state.mutated_target_plot_titles which was created with 
                        ### and is indexed according to st.session_state.target_plot_options
                        st.session_state.master_target_plot_index = st.session_state.mutated_target_plot_titles.index(st.session_state.target_plot_selection)
                        selected_target_option = st.session_state.target_plot_options[st.session_state.master_target_plot_index]
                    

                        # use the index to retrieve the corresponding key for plot params
                        # note that the plotly_express plots params use underscore. 
                        param_key = st.session_state.plot_param_keys[st.session_state.master_target_plot_index]
                        iterable_chunks_of_variables = selected_target_option

                        # st.session_state.seen_tar resets only when user changes the selected plot group.
                        # st.session_state.tar_for_plot_selection is used to determine if the plot group is changed
                        if st.session_state.get('tar_for_plot_selection') != st.session_state.target_plot_selection:
                            st.session_state.seen_tar = []
                            st.session_state.tar_for_plot_selection = st.session_state.target_plot_selection
                        seen = st.session_state.setdefault('seen_tar', [])
                        selected_partition = []
                        tabs = []

                        # plot one chunk at a time and keep track of which chunks have been plotted 
                        if not iterable_chunks_of_variables:
                            st.info('No partitions available for the selected group.')
                        else:
                            select_wid, seen_wid = st.columns([1, 1])
                            with select_wid:
                                # identify each chunk as index + 1
                                chunk_options = [i + 1 for i in range(len(iterable_chunks_of_variables))]
                                if not chunk_options:
                                    st.session_state.group_targ_selection = None
                                else:
                                    # a number that represents a chunk. it is index = number - 1
                                    st.session_state.group_targ_selection = st.selectbox('Select a Partition Chunk to Plot', 
                                                                                            chunk_options, 
                                                                                            index=0,
                                                                                            on_change = group_targ_chunk_selector,
                                                                                            key = 'group_targ_chunk_select')
                            with seen_wid:
                                st.markdown(f'Seen Partitions: {sorted(seen)}')                         

                            if st.session_state.group_targ_selection is None:
                                st.info('No selectable partition found for the selected group.')
                            else:
                                if st.session_state.group_targ_selection not in seen:
                                    seen.append(st.session_state.group_targ_selection)

                                # select a chunk of pairs/univariates to plot
                                selected_partition = iterable_chunks_of_variables[st.session_state.group_targ_selection - 1]

                                def _tab_title(variable_to_plot):
                                    if isinstance(variable_to_plot, (list, tuple)):
                                        return ' | '.join([str(v) for v in variable_to_plot])
                                    return str(variable_to_plot)

                                tab_titles = [_tab_title(v) for v in selected_partition]
                                tabs = st.tabs(tab_titles) if tab_titles else []


                        for tab, variable_to_plot in zip(tabs, selected_partition):
                            with tab:
                            # identify variable(s) type of plot numeric-categoric bivariate, numeric univariate etc
                                var_type_plot = st.session_state.plot_list_keys[st.session_state.master_target_plot_index]
                                #retrieve the appropriate plot function
                                plot_func_ = st.session_state.plot_function[st.session_state.master_target_plot_index]
                                # retrieve the plot parameters
                                curr_params = st.session_state.plot_params_dict[param_key].copy()
                                # process and plot one plot for each iteration in the chunk
                                plot_one_title(
                                        data = st.session_state.data,
                                        variable_to_plot=variable_to_plot,
                                        curr_params=curr_params,  
                                        var_type_plot=var_type_plot,
                                        plot_func_=plot_func_)
                                





#----------------------------------------------------------------------------------------------------------------------
elif st.session_state.page == "Feedback":

    FEEDBACK_FILE = "data_analyzer_app_feedback.txt"

    with st.sidebar:
        st.header("Navigation")
        st.session_state.page = st.radio("Navigation",
                                         NAVIGATION_PAGES,
                                         index=NAVIGATION_PAGES.index(st.session_state.page),
                                         key="navigation_control",
                                         on_change=set_page)

    st.title("Feedback")
    st.write("Leave a note, bug report, or suggestion below.")

    feedback_text = st.text_area("Your feedback", height=200, max_chars=3000, placeholder="Type your feedback here...")

    if st.button("Submit"):
        if feedback_text.strip():
            try:
                if len(feedback_text.strip()) > 3000:
                    st.error("Feedback exceeds the 3000 character limit. Please shorten your message.")
                else:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    entry = f"[{timestamp}]\n{feedback_text.strip()}\n{'-' * 60}\n"
                    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
                        f.write(entry)
                    st.success("Feedback saved.")
            except OSError as e:
                st.error(f"Could not save feedback: {e}")
        else:
            st.warning("Please enter some feedback before submitting.")

    st.divider()
    st.subheader("Previous Feedback")
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            contents = f.read()
        if contents.strip():
            st.text(contents)
        else:
            st.info("No feedback yet.")
    except FileNotFoundError:
        st.info("No feedback yet.")

elif st.session_state.page == "Binning Tool":
    render_binning_tool()

