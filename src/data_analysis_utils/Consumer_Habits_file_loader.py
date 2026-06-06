
from .BinnerClass import Bin
import pathlib
import pandas as pd
bin=Bin()


def load_consumer_habits(filepath:str="data/shopping_behavior_updated.csv"):
    """
    """
    global bin

    behavior=pathlib.Path(filepath)
    df=pd.read_csv(behavior)
    df['Male']=df['Gender'].replace({'Male':1,'Female':0}).astype('object')
    df.drop(columns=['Customer ID','Gender'],inplace=True)
    mymap={'Yes':1,'No':0}
    for col in ['Subscription Status', 'Discount Applied','Promo Code Used']:
        df[col]=df[col].map(mymap).astype('object')

    def freq_factor(x):
        if x=='Every 3 Months': return 365/4
        if x=='Annually': return 365/1
        if x=='Quarterly': return 365/4
        if x=='Monthly': return 365/12
        if x=='Bi-Weekly': return 7/2
        if x=='Fortnightly': return 14
        if x=='Weekly': return 7
    df['Total Days of Patronage']=(df['Frequency of Purchases'].map(freq_factor)*df['Previous Purchases']).astype(int)

    #bin numeric columns
    bin.relational_binner(df,
                    numnum_meth_alpha_above=('pearson',0.6,True),    
                    numcat_meth_alpha_above=('kruskal',0.05,False),   
                    original_value_count_threshold=5,  
                    numeric_columns=None,     
                    categoric_columns=None,    
                    numeric_target=None,      
                    categoric_target=None) 
    for k, v in bin.numeric_target_column_minimums.items():
        v=max(v,5)
        df[f"{k}_Binned"]=bin.binner(df[k],v,rescale=True)
        df[f"{k}_Binned"]=df[f"{k}_Binned"].astype(float).round(2)
        df[f"{k}_Ordinalized"]=bin.binner(df[k],v,rescale=False)
        df[f"{k}_Ordinalized"]=df[f"{k}_Ordinalized"].astype('object')
    

    
    return df