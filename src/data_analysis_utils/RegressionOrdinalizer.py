
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class RegressionOrdinalizer:

    def __init__(self):
        self.column_ordinal_records = {
                                       'header':[],
                                       'best_center':[],
                                       'r2':[],
                                       'coefficients':[],
                                       'intercept':[],
                                       'mode_variability_instructions':[]
                                       }
        self.ordinalization_lookup_maps = {}
        

    def ordinalize(self,dataframe_x_y:pd.DataFrame,center:str):
        """
        takes a dataframe of x=categorical values and y=numeric,
        returns a new x column that is ordinalized based on y values and the center input, where center can be any of 'mean','median', or 'mode'
        """
        x, y = dataframe_x_y.columns[0],dataframe_x_y.columns[1]
        if center=='mode':
            stats  = dataframe_x_y.groupby(x,as_index=False)[y].agg(lambda s: s.mode().iloc[0]).rename(columns={y:center}).set_index(x)
        else:
            stats  = dataframe_x_y.groupby(x,as_index=True)[y].agg([center])
        try:
            dataframe_x_y[center] = dataframe_x_y[x].map(stats[center])
        except:
            dataframe_x_y=pd.merge(dataframe_x_y,stats,left_on=x,right_index=True,how='left',sort=False)
        return dataframe_x_y[[center]]
    
    
    # a function that 'truncates' decimals. It's used to make mode more meaningful when ordinalizing
    def decrease_decimal_variability(self,input_vec,decimal_factor,place):
        """
        takes an x col such as df[[x]] and rounds (floors) decimals down to the lower decimal_factor
        """
        return ((input_vec*place)//decimal_factor)*decimal_factor/place  
     
    def decrease_number_variablity(self,input_vec,number_factor):
        """
        takes an x col such as df[[x]] and floors down to the lower number_factor
        """
        return (input_vec//number_factor)*number_factor       

        
    def column_ordinalizer_choose_best_center(self,dataframe:pd.DataFrame,target_column:str,feature_columns:list|None=None,mode_trunc:tuple|list=None, override_centers:list|None=None):
        """
        chooses between mean, median, and mode
        takes a dataframe as input and if feature_columns is None, auto detects all columns that are 'object'
        mode_trunc should be instrucitons to truncate all values before choosing a mode, if None, mode is mode. it should be of ['decimal',flore_factor,place] or ['number',floor_factor]
        where truncate_factor is the factor to round down to such as 1.3 with factor 2 and place 10 becomes 1.2, 36 with factor 5 becomes 35, 1.222 with place 100 and factor 1 becomes 1.21
        override_centers can be used to pass a singel or two centers instead of the default ['mean','median','mode']. 
        """
  

        if feature_columns is None:
            feature_columns=dataframe.select_dtypes(include='object').columns
        if override_centers is None:
            centers=['mean','median','mode']
        else:
            centers=override_centers
        if mode_trunc is not None:
            if mode_trunc[0]=='decimal':
                mode_y=self.decrease_decimal_variability(dataframe[target_column].copy(),mode_trunc[1],mode_trunc[2])
            elif mode_trunc[0]=='number':
                mode_y=self.decrease_number_variablity(dataframe[target_column].copy(),mode_trunc[1])
            else:
                raise ValueError("mode_trunc should be of ['decimal',flore_factor, place] or ['number',floor_factor], where a factor is a whole number and place is 10 for 10th 100 for 100th etc.")

        for column in feature_columns:
            max_r2 = -np.inf
            best = {}

            for center in centers:
                x_df = dataframe[[column]].copy()

                if center == 'mode' and mode_trunc is not None:
                    x_df['y_tmp'] = mode_y
                else:
                    x_df['y_tmp'] = dataframe[target_column]
                X = self.ordinalize(x_df[[column, 'y_tmp']], center)
                y = dataframe[target_column]

                reg = LinearRegression().fit(X, y)
                r2 = reg.score(X, y)

                if r2 > max_r2:
                    max_r2 = r2
                    best = dict(
                        r2=r2,
                        center=center,
                        coef=float(reg.coef_[0]),
                        intercept=reg.intercept_
                    )

            self.column_ordinal_records['header'].append(column)
            self.column_ordinal_records['r2'].append(best['r2'])
            self.column_ordinal_records['best_center'].append(best['center'])
            self.column_ordinal_records['coefficients'].append(best['coef'])
            self.column_ordinal_records['intercept'].append(best['intercept'])
            self.column_ordinal_records['mode_variability_instructions'].append(
                mode_trunc if best['center'] == 'mode' else None
            )
            

    def ranking_ordinalizer(self, X_y_df, center:str|None=None, mode_trunc:tuple|list=None):
        """
        uses size and center to sort and ordinalize
        returns a lookup key data frame with df[[x,f"{x}_Ordinalized"]] where ordinalized is range(0,x.nunique())
        if center is None, this looks up the center from self.column_ordinal_records
        else center should be of 'mean', 'median', or 'mode'
        if center is 'mode' mode_trunc should be instrucitons to truncate all values before choosing a mode, if None, mode is mode. it should be of ['decimal',flore_factor,place] or ['number',floor_factor]
        where truncate_factor is the factor to round down to such as 1.3 with factor 2 and place 10 becomes 1.2, 36 with factor 5 becomes 35, 1.222 with place 100 and factor 1 becomes 1.21
        """
        x, y = X_y_df.columns[0], X_y_df.columns[1]
        X_y_df=X_y_df.copy()
        if center is None:
            try:
                index      = self.column_ordinal_records['header'].index(x)
                center     = self.column_ordinal_records['best_center'][index]
                mode_trunc = self.column_ordinal_records['mode_variability_instructions'][index]
            except Exception as e:
                print(f"Didn't successfully find X in model.column_ordinal_records. Error: {e}")

        if center=='mode':
            if mode_trunc is not None:
                if mode_trunc[0]=='decimal':
                    X_y_df[y]=self.decrease_decimal_variability(X_y_df[y],mode_trunc[1],mode_trunc[2])
                elif mode_trunc[0]=='number':
                    X_y_df[y]=self.decrease_number_variablity(X_y_df[y],mode_trunc[1])
            stats  = X_y_df.groupby(x,as_index=False)[y].agg(lambda s: s.mode().iloc[0]).rename(columns={y:center})
            sizes  = X_y_df.groupby(x,as_index=False)[y].size()
            stats  = pd.merge(stats,sizes, on=x)
        else:
            stats  = X_y_df.groupby(x,as_index=False)[y].agg([center,'size'])
        stats = stats.sort_values(by=[center,'size'],ascending=[True,True]).reset_index(drop=True)
        stats[f"{x}_Ordinalized"] = stats.index
        self.ordinalization_lookup_maps[f"{x}_Ordinalized"]=stats[[x,f"{x}_Ordinalized"]]
        return stats[[x,f"{x}_Ordinalized"]]
    
    
    def center_ordinalizer(self, X_y_df, center:str|None=None, mode_trunc:tuple|list=None):
        """
        returns a lookup key data frame with df[[x,f"{x}_Ordinalized"]] where ordinalized is range(0,x.nunique())
        if center is None, this looks up the center from self.column_ordinal_records
        else center should be of 'mean', 'median', or 'mode'
        if center is 'mode' mode_trunc should be instrucitons to truncate all values before choosing a mode, if None, mode is mode. it should be of ['decimal',flore_factor,place] or ['number',floor_factor]
        where truncate_factor is the factor to round down to such as 1.3 with factor 2 and place 10 becomes 1.2, 36 with factor 5 becomes 35, 1.222 with place 100 and factor 1 becomes 1.21
        """
        x, y = X_y_df.columns[0], X_y_df.columns[1]
        X_y_df=X_y_df.copy()
        if center is None:
            try:
                index      = self.column_ordinal_records['header'].index(x)
                center     = self.column_ordinal_records['best_center'][index]
                mode_trunc = self.column_ordinal_records['mode_variability_instructions'][index]
            except Exception as e:
                print(f"Didn't successfully find X in model.column_ordinal_records. Error: {e}")
                return None

        if center=='mode':
            if mode_trunc is not None:
                if mode_trunc[0]=='decimal':
                    X_y_df[y]=self.decrease_decimal_variability(X_y_df[y],mode_trunc[1],mode_trunc[2])
                elif mode_trunc[0]=='number':
                    X_y_df[y]=self.decrease_number_variablity(X_y_df[y],mode_trunc[1])
            stats  = X_y_df.groupby(x,as_index=False)[y].agg(lambda s: s.mode().iloc[0]).rename(columns={y:center})
        else:
            stats  = X_y_df.groupby(x,as_index=False)[y].agg([center])
        stats = stats.rename(columns={center:f"{x}_Ordinalized"})
        self.ordinalization_lookup_maps[f"{x}_Ordinalized"]=stats[[x,f"{x}_Ordinalized"]]
        return stats[[x,f"{x}_Ordinalized"]]
    

    def create_ordinalized_columns(self, dataframe:pd.DataFrame, target_column:str, feature_columns:list|None, center:str|None, rank:bool=False, mode_trunc:tuple|list=None):
        """
        where how should be one of 'mean', 'median', 'mode', or None
        """

        if feature_columns is None:
            feature_columns = dataframe.select_dtypes(include='object').columns

        if center is None:
            self.column_ordinalizer_choose_best_center(dataframe,target_column,feature_columns,mode_trunc, None)
            if rank==True:
                for column in feature_columns:
                    lookup=self.ranking_ordinalizer( dataframe[[column,target_column]].copy())
                    dataframe=pd.merge(dataframe,lookup[[column,f"{column}_Ordinalized"]],on=column,how='left')
            else:
                for column in feature_columns:
                    lookup=self.center_ordinalizer(dataframe[[column,target_column]].copy())
                    dataframe=pd.merge(dataframe,lookup[[column,f"{column}_Ordinalized"]],on=column,how='left')
            return dataframe

        else:
            if rank==True:
                for column in feature_columns:
                    lookup=self.ranking_ordinalizer( dataframe[[column,target_column]].copy() , center , mode_trunc)
                    dataframe=pd.merge(dataframe,lookup[[column,f"{column}_Ordinalized"]],on=column,how='left')
            else:
                for column in feature_columns:
                    lookup=self.center_ordinalizer(dataframe[[column,target_column]].copy() , center , mode_trunc)
                    dataframe=pd.merge(dataframe,lookup[[column,f"{column}_Ordinalized"]],on=column,how='left')
            return dataframe



        


            

