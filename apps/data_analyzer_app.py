import numpy as np
import pandas as pd
import streamlit as st

from data_analysis_utils import AnalyzeDataset

# ======================|
# MODEL PARAMETERS      |
# DATA MANIPULATION     | 
# UI CONTROL            |
# CALL_BACKS            |
# ======================|


# MODEL PARAMETERS
# =======================================================================================================================================
# =======================================================================================================================================
if 'n_wide' not in st.session_state:
    st.session_state.n_wide = {'n_wide': [8, 30, 4]}

if "cat_univar_params" not in st.session_state:
    st.session_state.cat_univar_params = {"proportions": False, "n_wide": (8, 30, 4), "super_title": "Univariate Categorical Variables - Reject Good-Of-Fit for Uniform"}

if "catcat_bivar_params" not in st.session_state:
    st.session_state.catcat_bivar_params = {"n_wide": (8, 30, 4), "stacked_bars_when_max_bars_is_exceeded": True, "sorted": False, "super_title": "Categoric-To-Categoric Bivariates - Reject Null"}

if "numnum_bivar_params" not in st.session_state:
    st.session_state.numnum_bivar_params = {"plot_type": "joint", "linreg": False, "plot_type_kwargs": None, "linreg_kwargs": None, "super_title": "Numeric Bivariates With Significant Correlation(s)"}

if "numcat_bivar_params" not in st.session_state:
    st.session_state.numcat_bivar_params = {"plot_type": "boxen", "n_wide": (8, 30, 4), "super_title": "Numeric-to-Categoric Bivariates  - Reject Null"}

if "super_subcat_pairs_params" not in st.session_state:
    st.session_state.super_subcat_pairs_params = {"row_height": 3, "cols_per_row": 2, "y_tick_fontsize": 12, "super_title": "Supercategory-Subcategory - One Categoric Variable Partitions Another"}

if "num_univar_params" not in st.session_state:
    st.session_state.num_univar_params = {"kde": None, "proportions": False, "n_wide": (8, 30, 4), "super_title": "Univariate Numerical Variables - Reject Normal Distribution", "force_significant_bin_edges": None, "minimize_significant_bins": None, "include_multivariate": True}

if "kruskal_assumption_check_params" not in st.session_state:
    st.session_state.kruskal_assumption_check_params = {"levene_alpha": 0.01, "ks_alpha": 0.01, "return_pseudo": True, "pseudo_test_max_global_ties_ratio": 0.7, "full_pseudo": False, "dropna": True, "n_jobs": 4, "guesstimate": {"rej_max_pct_in_group": 0.2, "max_num_outlier_all_reject": 3, "max_pct_reject_total": 0.2}}

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

# PLOTTING OPTION LISTS
# =====================================================================================================================================
# =====================================================================================================================================
if 'valid_group_param_indexes' not in st.session_state:
    st.session_state.valid_group_param_indexes = None

if 'univar_bivar_target_params_and_pairs' not in st.session_state:
    st.session_state.univar_bivar_target_params_and_pairs = None

if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 10

if 'group_plotable' not in st.session_state:
    st.session_state.group_plotable = []

# CALL_BACKS
# =======================================================================================================================================
# =======================================================================================================================================  
def set_page():
    st.session_state.page = st.session_state.navigation_control





if st.session_state.page == "Data Upload & Processing":
    

    # Sidebar for threshold adjustments
    with st.sidebar:

        # Page navigation #
        st.header("Navigation")        
        st.session_state.page = st.radio("Navigate", 
                                         ["Data Upload & Processing", "Group Visualizations", "Target Visualizations", "Feedback"], 
                                         index=["Data Upload & Processing", "Group Visualizations", "Target Visualizations", "Feedback"].index(st.session_state.page),
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
        curr_isolate = st.session_state.supercat_subcat_params["isolate_super_subs"]
        isolate_options = [curr_isolate, not curr_isolate]
        selected_isolate = st.selectbox("Isolate Partitioning Super-Subcategory Counterparts",isolate_options,index=0)
        if selected_isolate != st.session_state.supercat_subcat_params['isolate_super_subs']:
            st.session_state.supercat_subcat_params.update({'isolate_super_subs':selected_isolate })

        
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

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], help="File size limit: 10MB")

# ======================================================================================================================================
# ======================================================================================================================================
# ======================================================================================================================================

    if uploaded_file is not None:

        #   multipliers to help name a dtype
        st.session_state.min_pct_non_null_to_propose_a_dtype                        = 0.99
        st.session_state.max_pct_unique_for_numtype_cattype_threshold               = 0.005 # over for num <= for cat
        st.session_state.max_date_cols_used                                         = 5
        st.session_state.max_unique_pct_of_total_ie_identifier_variable_ID          = 0.2 # percentage of all observations that are unique to a columns, such as ID number or unique locations
        st.session_state.propose_drop_frac_nan_for_num                              = 0.95
        st.session_state.max_frac_droppable_nan_cat                                 = 0.1 # nans over this frac will be converted to str(NAN)


        # Check file size (10MB limit for fly.io)
        file_size = len(uploaded_file.getvalue())
        if file_size > 10 * 1024 * 1024:
            st.error("File size exceeds 10MB limit. Please upload a smaller file.")
        else:

            try:

                if 2 < st.session_state.step <= 7:
                    # display a sample
                    n_sample_rows = 3
                    st.markdown(f'{n_sample_rows} sample rows of values')
                    st.dataframe(st.session_state.data.sample(min(n_sample_rows, len(st.session_state.data))))
                if st.session_state == 7:
                    st.markdown(f'Hypothesis and Correlation Results')
                    st.dataframe(st.session_state.result_df)


                # Read CSV to DataFrame
                # Only re-read CSV when a new file is uploaded; preserve session state data across reruns
                if st.session_state.uploaded_file_name != uploaded_file.name:

                    # reset dynamic ploting variables
                    st.session_state.valid_group_param_indexes = None
                    st.session_state.univar_bivar_target_params_and_pairs = None
                    st.session_state.group_plotable = []

                    st.session_state.step = 1



                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.data = pd.read_csv(uploaded_file)   
                    if len(st.session_state.data.shape)<2:
                        st.session_state.data = st.session_state.data.to_frame()
                    # capture original len and shape
                    st.session_state.original_len = len(st.session_state.data)
                    st.session_state.original_shape = st.session_state.data.shape

                    st.session_state.step = 2

                    # display a sample
                    st.markdown('3 sample rows of existing values')
                    st.dataframe(st.session_state.data.sample(min(3, len(st.session_state.data))))

                if  st.session_state.step==2:
                
                    # Drop nulls
                    # DROPNA NOT REQUIRED. MODULE WILL HANDLE NAN INTERNALLY
                    #st.session_state.data = st.session_state.data.dropna()

                    new_len = len(st.session_state.data)

                    # store cols and meta
                    cols_and_meta = {}
                    # Robust datatype detection
                    date_columns = []
                    native_date_cols = []
                    for col in st.session_state.data.columns:
                        proposed_dtype = None
                        col_dtype = st.session_state.data[col].dtype
                        niq = st.session_state.data[col].nunique()
                        nan = st.session_state.data[col].isna().sum()

                    
                        if str(col_dtype).startswith('datetime') or str(col_dtype).startswith('timedelta'):
                            date_columns.append(col)
                            native_date_cols.append(col)
                            proposed_dtype = 'datetime'
                        # Try to detect dates/datetimes
                        elif str(col_dtype) in ('object','category'):
                            test_as_dt = pd.to_datetime(st.session_state.data[col].copy(), errors='coerce').notna().sum()

                            # if st.session_state.min_pct_non_null_to_propose_a_dtype% convert to datetime it can be datetime
                            if (test_as_dt >= (st.session_state.min_pct_non_null_to_propose_a_dtype * new_len)):
                                date_columns.append(col)
                                proposed_dtype = 'datetime'

                            else:
                                # attempt to parse as numeric
                                test_as_num = pd.to_numeric(st.session_state.data[col].copy(), errors='coerce').notna().sum()

                                # if >st.session_state.min_pct_non_null_to_propose_a_dtype% are valid values and over st.session_state.max_pct_unique_for_numtype_cattype_threshold are unique, it can be int or float
                                if (test_as_num>(new_len*st.session_state.min_pct_non_null_to_propose_a_dtype)) and (niq>(new_len*st.session_state.max_pct_unique_for_numtype_cattype_threshold)):
                                    if (pd.to_numeric(st.session_state.data[col].copy(), errors='coerce').dropna() % 1 != 0).any():
                                        proposed_dtype = 'float64'
                                    else:
                                        proposed_dtype = 'int64'
                                else:
                                    proposed_dtype = 'category'

                        elif str(col_dtype).startswith('int') or str(col_dtype).startswith('float') or str(col_dtype).startswith('UInt'):
                            # if not over st.session_state.max_pct_unique_for_numtype_cattype_threshold are unique, it will be categoric
                            if (niq>(new_len*st.session_state.max_pct_unique_for_numtype_cattype_threshold)):
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
                                            False]

                    index = [
                            'original_dtype', 
                            'possible_change',
                            'n_unique', 
                            'accept_change', 
                            'n_nan', 
                            'drop'
                            ]
                    st.session_state.datatypes_df               = pd.DataFrame(cols_and_meta, index=index)

                    st.session_state.step = 3


                if st.session_state.step == 3:


                    # Restore detection results from session state on reruns
                    new_len                     = len(st.session_state.data)


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
                                                (st.session_state.indexed_col_meta_df['n_nan']>(new_len*st.session_state.max_frac_droppable_nan_cat)))
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
                    st.session_state.indexed_col_meta_df['pct_unique']    = st.session_state.indexed_col_meta_df['n_unique']/new_len
                    # pct nan
                    st.session_state.indexed_col_meta_df['pct_nan']       = st.session_state.indexed_col_meta_df['n_nan']/new_len
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
                    over_unique_id_threshold       = st.session_state.indexed_col_meta_df['pct_unique'] > st.session_state.max_unique_pct_of_total_ie_identifier_variable_ID
                    excessiv_nan_for_numeric       = st.session_state.indexed_col_meta_df['pct_nan']    > st.session_state.propose_drop_frac_nan_for_num         
                    # capture it now before rows are dropped and dtypes changed etcetera
                    st.session_state.indexed_col_meta_df['nan_to_NAN_if_cat']    = st.session_state.indexed_col_meta_df['pct_nan']    > st.session_state.max_frac_droppable_nan_cat



                    #explain thresholds to user
                    st.subheader('Data Manipulation')
                    st.markdown(f"Threshold 'pct_nan' to drop a column - %{round(100*st.session_state.propose_drop_frac_nan_for_num,2)}")
                    st.markdown(f"Threshold 'pct_nan' to convert NaN to string: 'NAN' - %{round(100*st.session_state.max_frac_droppable_nan_cat,2)}")
                    st.markdown(f"Threshold 'pct_unique' to drop an identifier column - %{round(100*st.session_state.max_unique_pct_of_total_ie_identifier_variable_ID,2)}")
                    st.markdown(f"Threshold 'pct_unique' to consider a type change - %{round(100*st.session_state.max_pct_unique_for_numtype_cattype_threshold,3)}")


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
                                                        st.session_state.indexed_col_meta_df['make_change'].copy()
                                                        )][['original_dtype', 'possible_change', 
                                                            'n_unique','pct_unique',  
                                                            'n_nan','pct_nan', 'make_change', 'drop']]
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
                            st.info("All Columns Have Been Dropped. There is No Data to Process.")

                            st.session_state.early_stop = True
                            st.session_state.step = 1



                        if not st.session_state.early_stop:
                            st.rerun()
                    #st.rerun('fragment')
                        
                        

                if st.session_state.step == 5:

         
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
                        # capture date type cols
                        is_date = (
                                    (   st.session_state.indexed_col_meta_df['dtype_as_string'].str.startswith('date')    ) 
                                    | 
                                    (   st.session_state.indexed_col_meta_df['dtype_as_string'].str.startswith('time')   )
                                    )
                        date_columns = list(st.session_state.indexed_col_meta_df.loc[is_date].index)

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

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Please upload a CSV file to begin analysis.")

    if st.session_state.step==6:
        fit = st.button("Fit the Data")

        if fit:
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
            
  


            result_df = st.session_state.AD.column_relationships_df(st.session_state.data.columns).reset_index(drop=False)
            st.session_state.result_df = result_df[['Target', 'Distribution', 'FeatureColum(s)', 'Test(s)']].rename(columns={'Test(s)':'Test:Assumptions_Met'}).set_index(['Target','Distribution'])

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
                                         ["Data Upload & Processing", "Group Visualizations", "Target Visualizations", "Feedback"], 
                                         index=["Data Upload & Processing", "Group Visualizations", "Target Visualizations", "Feedback"].index(st.session_state.page),
                                         key = 'navigation_control',
                                         on_change = set_page)

        st.header("Plot Layout Controls")
        
        # Number of plot axes per row
        n_axes_options = [5, 6, 7]
        selected_n_axes = st.selectbox("Number of plot axes per row", n_axes_options, index=n_axes_options.index(st.session_state.n_wide['n_wide'][0]) if st.session_state.n_wide['n_wide'][0] in n_axes_options else 0)
        
        # Ideal max bars per row
        max_bars_options = [20, 30, 40, 50]
        selected_max_bars = st.selectbox("Ideal max bars per row", max_bars_options, index=max_bars_options.index(st.session_state.n_wide['n_wide'][1]) if st.session_state.n_wide['n_wide'][1] in max_bars_options else 1)
        
        # Row height
        height_options = [3, 4, 5, 6]
        selected_height = st.selectbox("Row height (inches)", height_options, index=height_options.index(st.session_state.n_wide['n_wide'][2]) if st.session_state.n_wide['n_wide'][2] in height_options else 2)
        
        # Update n_wide
        st.session_state.n_wide['n_wide'] = [selected_n_axes, selected_max_bars, selected_height]   

        # Update all relevant params
        st.session_state.num_univar_params.update(st.session_state.n_wide)
        st.session_state.cat_univar_params.update(st.session_state.n_wide)
        st.session_state.catcat_bivar_params.update(st.session_state.n_wide)
        # N/A st.session_state.numnum_bivar_params.update(st.session_state.n_wide)
        st.session_state.numcat_bivar_params.update(st.session_state.n_wide)
        
        st.header("Numerical Univariate Parameters")
        
        # force_significant_bin_edges
        #force_options = ["None", "True"]
        #current_force = "True" if st.session_state.num_univar_params.get('force_significant_bin_edges') else "None"
        #selected_force = st.selectbox("Force significant bin edges", force_options, index=force_options.index(current_force))
        st.session_state.num_univar_params['force_significant_bin_edges'] = True #if selected_force == "True" else None
        
        # minimize_significant_bins
        minimize_options = ["False", "True"]
        current_minimize = "True" if st.session_state.num_univar_params.get('minimize_significant_bins') else "False"
        selected_minimize = st.selectbox("Minimize significant bins", minimize_options, index=minimize_options.index(current_minimize))
        st.session_state.num_univar_params['minimize_significant_bins'] = True if selected_minimize == "True" else None
        
        # include_multivariate
        # multivariate_options = ["True", "False"]
        # current_multivariate = "True" if st.session_state.num_univar_params.get('include_multivariate', True) else "False"
        # selected_multivariate = st.radio("Include multivariate in bin computations", multivariate_options, index=multivariate_options.index(current_multivariate))
        # st.session_state.num_univar_params['include_multivariate'] = selected_multivariate == "True"

        st.header("Numeric-Categoric Bivariate Parameters")
        
        # plot_type for numcat_bivar_params
        plot_type_options = ["box", "boxen", "violin"]
        current_plot_type = st.session_state.numcat_bivar_params.get('plot_type', 'boxen')
        selected_plot_type = st.selectbox("Plot type for numeric-categoric plots", plot_type_options, index=plot_type_options.index(current_plot_type) if current_plot_type in plot_type_options else 1)
        st.session_state.numcat_bivar_params['plot_type'] = selected_plot_type
        
        st.header("Numeric-Numeric Bivariate Parameters")
        
        # plot_type for numnum_bivar_params
        numnum_plot_type_options = ["joint", "scatter"]
        current_numnum_plot_type = st.session_state.numnum_bivar_params.get('plot_type', 'joint')
        selected_numnum_plot_type = st.selectbox("Plot type for numeric-numeric plots", numnum_plot_type_options, index=numnum_plot_type_options.index(current_numnum_plot_type) if current_numnum_plot_type in numnum_plot_type_options else 0)
        st.session_state.numnum_bivar_params['plot_type'] = selected_numnum_plot_type
        
        # linreg for numnum_bivar_params
        linreg_options = ["False", "True"]
        current_linreg = "True" if st.session_state.numnum_bivar_params.get('linreg', False) else "False"
        selected_linreg = st.selectbox("Include linear regression line", linreg_options, index=linreg_options.index(current_linreg))
        st.session_state.numnum_bivar_params['linreg'] = selected_linreg == "True"
        
        st.header("Supercategory-Subcategory Parameters")
        
        # row_height for super_subcat_pairs_params
        row_height_options = [2, 3, 4, 5]
        current_row_height = st.session_state.super_subcat_pairs_params.get('row_height', 3)
        selected_row_height = st.selectbox("Row height (inches) for supersubcat plots", row_height_options, index=row_height_options.index(current_row_height) if current_row_height in row_height_options else 1)
        st.session_state.super_subcat_pairs_params['row_height'] = selected_row_height
        
        # cols_per_row for super_subcat_pairs_params
        cols_per_row_options = [1, 2, 3, 4]
        current_cols_per_row = st.session_state.super_subcat_pairs_params.get('cols_per_row', 2)
        selected_cols_per_row = st.selectbox("Columns per row for subcategory plots", cols_per_row_options, index=cols_per_row_options.index(current_cols_per_row) if current_cols_per_row in cols_per_row_options else 1)
        st.session_state.super_subcat_pairs_params['cols_per_row'] = selected_cols_per_row
        
        # y_tick_fontsize for super_subcat_pairs_params
        y_tick_fontsize_options = [8, 10, 12, 14, 16]
        current_y_tick_fontsize = st.session_state.super_subcat_pairs_params.get('y_tick_fontsize', 12)
        selected_y_tick_fontsize = st.selectbox("Y-axis tick font size", y_tick_fontsize_options, index=y_tick_fontsize_options.index(current_y_tick_fontsize) if current_y_tick_fontsize in y_tick_fontsize_options else 2)
        st.session_state.super_subcat_pairs_params['y_tick_fontsize'] = selected_y_tick_fontsize

  
    if st.session_state.step < 7:
        st.info("The Data Hasn't Been Fit.")
    else:
        # PLOT GROUPS
        # ---------------------------------------------------------------------------------------------------
        if st.session_state.page == "Group Visualizations":

            plot_overview_type_index_position = [
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


            var_target_params_and_pairs = [
                        {'num_univar':list(st.session_state.AD.reject_null_normal),
                            'num_univar_params':{**st.session_state.num_univar_params,'super_title':'Numerical Non-Normal'}} ,
                        {'num_univar':list(st.session_state.AD.fail_to_reject_null_normal),
                            'num_univar_params':{**st.session_state.num_univar_params,'super_title':'Numerical Normal'}}  , 
                        {'cat_univar':list(st.session_state.AD.reject_null_good_of_fit),
                            'cat_univar_params':{**st.session_state.cat_univar_params,'super_title':'Categorical Non-Uniform'}} ,  
                        {'cat_univar':list(st.session_state.AD.fail_to_reject_null_good_of_fit),
                            'cat_univar_params':{**st.session_state.cat_univar_params,'super_title':'Categorical Uniform'}} , 
                        {'numnum_bivar':st.session_state.AD.above_threshold_corr_numnum,
                            'numnum_bivar_params':{**st.session_state.numnum_bivar_params, 'super_title':'Numeric-Numeric With Correlation'}}  , 
                        {'numnum_bivar':st.session_state.AD.below_threshold_corr_numnum,
                            'numnum_bivar_params':{**st.session_state.numnum_bivar_params,'super_title':'Numeric-Numeric Without Correlation'}} , 
                        {'numcat_bivar':st.session_state.AD.reject_null_numcat,
                            'numcat_bivar_params':{**st.session_state.numcat_bivar_params, 'super_title':'Numeric-Categoric Reject Null'}}  , 
                        {'numcat_bivar': st.session_state.AD.fail_to_reject_null_numcat,
                            'numcat_bivar_params':{**st.session_state.numcat_bivar_params,'super_title':'Numeric-Categoric Fail to Reject Null'}}  , 
                        {'catcat_bivar':st.session_state.AD.reject_null_catcat,
                            'catcat_bivar_params':{**st.session_state.catcat_bivar_params,'super_title':'Categoric-Categoric Reject Null'}}  , 
                        {'catcat_bivar':st.session_state.AD.fail_to_reject_null_catcat,
                            'catcat_bivar_params':{**st.session_state.catcat_bivar_params,'super_title':'Categoric-Categoric Fail to Reject Null'}}  , 
                        {'super_subcat_pairs':st.session_state.AD.supercategory_subcategory_pairs,
                        'super_subcat_pairs_params':{**st.session_state.super_subcat_pairs_params,'super_title':'Categoric Partitioned by Another Categoric'}} ,                                             
                        {'catcat_bivar':st.session_state.AD.assumptions_not_met['catcat'],
                        'catcat_bivar_params':{**st.session_state.catcat_bivar_params,'super_title':'Assumptions Not Met -- Categoric-Categoric'}},
                        {'numnum_bivar':st.session_state.AD.assumptions_not_met['numnum'],
                        'numnum_bivar_params':{**st.session_state.numnum_bivar_params,'super_title':'Assumptions Not Met -- Numeric-Numeric'}},
                        {'numcat_bivar': st.session_state.AD.assumptions_not_met['numcat'],
                        'numcat_bivar_params':{**st.session_state.numcat_bivar_params,'super_title':'Assumptions Not Met -- Numeric-Categoric'}},
                        {'num_univar':list(st.session_state.AD.assumptions_not_met['num']),
                        'num_univar_params':{**st.session_state.num_univar_params,'super_title':'Assumptions Not Met -- Numerical'}},
                        {'cat_univar':list(st.session_state.AD.assumptions_not_met['cat']),
                        'cat_univar_params':{**st.session_state.cat_univar_params,'super_title':'Assumptions Not Met -- Categorical'}}
                        ]
            
            plot_list_keys = [
                              'num_univar', 'num_univar', 
                              'cat_univar', 'cat_univar',
                              'numnum_bivar', 'numnum_bivar',
                              'numcat_bivar', 'numcat_bivar',
                              'catcat_bivar', 'catcat_bivar',
                              'super_subcat_pairs',
                              'catcat_bivar', 'numnum_bivar', 'numcat_bivar', 
                              'num_univar', 'cat_univar'
                              ]

            if (st.session_state.valid_group_param_indexes is None):  # COULD ADD or (st.session_state.univar_bivar_target_params_and_pairs is None): but it is needless
                valid_indexes = [] 
                new_params_list = []
                for i in range(len( plot_overview_type_index_position)):
                    # fill a list with dict params to iterate over
                    curr_param = []
                    if var_target_params_and_pairs[i][plot_list_keys[i]]:
                        valid_indexes.append(i)
                        if len(var_target_params_and_pairs[i][plot_list_keys[i]])>st.session_state.chunk_size:
                            second_key = None
                            # seek and capture the second key, it is the non match to the first
                            for key in var_target_params_and_pairs[i].keys():
                                if key != plot_list_keys[i]:
                                    second_key=key
                                    break
                            start = 0
                            stop  = st.session_state.chunk_size
                            # create an iterable list at the index position
                            while start < len(var_target_params_and_pairs[i][plot_list_keys[i]]):
                                chunk = var_target_params_and_pairs[i][plot_list_keys[i]][start:stop]
                                start = stop
                                stop = stop + st.session_state.chunk_size
                                appendable = {plot_list_keys[i]:chunk,second_key:var_target_params_and_pairs[i][second_key]}
                                curr_param.append(appendable)
                            #suffix =  f": Fewer Than {len(curr_param)*st.session_state.chunk_size} Plots"
                            #plot_overview_type_index_position[i]+=suffix
                            new_params_list.append(curr_param)
                        else:
                            # else enclose/append it in a list: [ original_list ], so it can be iterated 1 time
                            curr_param.append(var_target_params_and_pairs[i])
                            #suffix =  f": Fewer Than {len(curr_param)*st.session_state.chunk_size} Plots"
                            #plot_overview_type_index_position[i]+=suffix
                            new_params_list.append(curr_param)
                    else:
                        # it's a placeholder
                        new_params_list.append(curr_param)
                        st.info(f"{plot_overview_type_index_position[i]} Not in the Data.")
                st.session_state.univar_bivar_target_params_and_pairs = new_params_list
                st.session_state.valid_group_param_indexes =  valid_indexes

            if not st.session_state.group_plotable:
                st.session_state.group_plotable = [plot_overview_type_index_position[i] for i in st.session_state.valid_group_param_indexes]
            if ( 
                (st.session_state.group_plotable is not None) 
                and 
                (st.session_state.univar_bivar_target_params_and_pairs is not None) 
                and 
                (st.session_state.univar_bivar_target_params_and_pairs is not None)
                ):
                try:
                    plot_selection = st.segment_control("Select a Group to Plot", 
                                                    st.session_state.group_plotable,
                                                    selection_mode="single",
                                                    default=None)
                except:
                    plot_selection = st.selectbox("Select a Group to Plot", 
                                                    st.session_state.group_plotable,
                                                    index=None)
                if plot_selection:


                    default_params =  {
                                        'cat_univar':False,   
                                        'num_univar':False, 
                                        'catcat_bivar':False,
                                        'numnum_bivar':False,
                                        'numcat_bivar':False,
                                        'super_subcat_pairs':False,
                                        'cat_univar_params':st.session_state.cat_univar_params,
                                        'catcat_bivar_params': st.session_state.catcat_bivar_params,
                                        'numnum_bivar_params': st.session_state.numnum_bivar_params,
                                        'numcat_bivar_params':  st.session_state.numcat_bivar_params,
                                        'super_subcat_pairs_params': st.session_state.super_subcat_pairs_params,
                                        'num_univar_params': st.session_state.num_univar_params
                                        }    

                    iterable_params_list = st.session_state.univar_bivar_target_params_and_pairs[plot_overview_type_index_position.index(plot_selection)]
                    # iterate and plot
                    for params in iterable_params_list:
                        # update input parameters  
                        curr_params = default_params.copy()         
                        curr_params.update(params)
                        #  plot funciton
                        st.session_state.AD.produce_all_plots(st.session_state.data,
                                                    **curr_params,
                                                    streamlit_=True)
                
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

            

            available_ploting_options = [
                                        'Univariate - Numeric Non-Normal and Categorical Non-Uniform',
                                        'Numeric-Numeric With Correlation',
                                        'Numeric-Categoric Reject Null', 
                                        'Categoric-Categoric Reject Null',
                                        'Categoric Partitioned by Another Categoric'
                                        ]
            try:
                plot_type_selection = st.segment_control("Select a Plot Type", 
                                                        available_ploting_options,
                                                        selection_mode="multi")
            except:
                plot_type_selection = st.multiselect("Select a Plot Type", 
                                                        available_ploting_options,
                                                        default=None)

            plot_selections = st.button("Plot Selections")
            if plot_selections:
                if ( not variable_selection) or (not plot_type_selection):
                    st.info("No Selections to Plot")
                else:

                    param_col_lists = [
                                        {'not_uniform_or_reject_normal':True,
                                         'num_univar_params':{**st.session_state.num_univar_params,'super_title':'Numerical Non-Normal'}
                                         ,'cat_univar_params':{**st.session_state.cat_univar_params,'super_title':'Categorical Non-Uniform'}} ,   
                                        {'reject_numnum':True,'numnum_bivar_params':{**st.session_state.numnum_bivar_params,'super_title':'Numeric-Numeric With Correlation'}}  ,  
                                        {'reject_numcat':True,'numcat_bivar_params':{**st.session_state.numcat_bivar_params,'super_title':'Numeric-Categoric Reject Null'}}  ,  
                                        {'reject_catcat':True,'catcat_bivar_params':{**st.session_state.catcat_bivar_params,'super_title':'Categoric-Categoric Reject Null'}}  ,  
                                        {'is_super_or_subcat':True,'super_subcat_pairs_params':{**st.session_state.super_subcat_pairs_params,'super_title':'Categoric Partitioned by Another Categoric'}}   
                                        ]
                    target_plot_default_params = {    
                                                    'reject_numcat': False,  
                                                    'reject_numnum': False,
                                                    'reject_catcat': False,
                                                    'is_super_or_subcat': False,
                                                    'not_uniform_or_reject_normal': False,  
                                                    'reject_multivariates': False,        
                                                    'auto_fit': True,   
                                                    'targets_share_plots': True ,  
                                                    'check_assumptions': True,
                                                    'dropna_gof': True,
                                                    'cat_univar_params':  st.session_state.cat_univar_params,
                                                    'catcat_bivar_params':  st.session_state.catcat_bivar_params,
                                                    'numnum_bivar_params':  st.session_state.numnum_bivar_params,
                                                    'numcat_bivar_params':  st.session_state.numcat_bivar_params,
                                                    'super_subcat_pairs_params':  st.session_state.super_subcat_pairs_params,
                                                    'num_univar_params':  st.session_state.num_univar_params
                                                    }
                    # use this to make sure values are present to plot
                    list_of_lists = [ list(st.session_state.AD.reject_null_normal)+list(st.session_state.AD.reject_null_good_of_fit),
                                      st.session_state.AD.above_threshold_corr_numnum,
                                      st.session_state.AD.reject_null_numcat,
                                      st.session_state.AD.reject_null_catcat,
                                      st.session_state.AD.supercategory_subcategory_pairs ] 
                    # only update parameters where variables are selected
                    index_list = []
                    for  i in range(len(available_ploting_options)):
                        if (available_ploting_options[i] in plot_type_selection):
                            if ( not list_of_lists[i] ):
                                st.warning(f"{available_ploting_options[i]} not present in the data")
                            else:
                                index_list.append(i)
                    for index in index_list:
                        target_plot_default_params.update(param_col_lists[index])

                    st.session_state.AD.visualize_by_targets(
                                            data=st.session_state.data,
                                            targets = list([variable_selection]),
                                            **target_plot_default_params ,
                                             streamlit_=True )


elif st.session_state.page == "Feedback":
    import datetime

    FEEDBACK_FILE = "data_analyzer_app_feedback.txt"

    with st.sidebar:
        st.header("Navigation")
        st.session_state.page = st.radio("Navigation",
                                         ["Data Upload & Processing", "Group Visualizations", "Target Visualizations", "Feedback"],
                                         index=["Data Upload & Processing", "Group Visualizations", "Target Visualizations", "Feedback"].index(st.session_state.page),
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

