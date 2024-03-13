"""Funciones que se utilizaran en el proyecto"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate 
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


#FUNCION 1#
def transformacion(df, variables):
    for i in variables:
        df[i] = df[i].astype('object')
    return df

# FUNCION NULOS#
def nulos(df):
    for i in df.columns:
        if df[i].isnull().sum()/len(df) < 0.1:
            df = df.dropna(subset = [i]) 
        else:
            df 
    return df

# FUNCION 3# 

def escala(df,variables):
    for i in variables:
        df[i] = df[i].replace({'1.0':'Muy insatisfecho', '2.0':'Insatisfecho', '3.0':'Satisfecho', '4.0':'Muy satisfecho'})
    return df

# FUNCION 4#
def sel_variables(modelos,X,y,threshold):
    
    var_names_ac=np.array([])
    for modelo in modelos:
   
        modelo.fit(X,y)
        sel = SelectFromModel(modelo, prefit=True,threshold=threshold)
        var_names= modelo.feature_names_in_[sel.get_support()]
        var_names_ac=np.append(var_names_ac, var_names)
        var_names_ac=np.unique(var_names_ac)
    
    return var_names_ac

# FUNCION 5#


def medir_modelos(modelos,scoring,X,y,cv):
    os = RandomOverSampler()
    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        pipeline = make_pipeline(os, modelo)
        scores=cross_val_score(modelo,X,y, scoring=scoring, cv=cv )
        pdscores=pd.DataFrame(scores)
        metric_modelos=pd.concat([metric_modelos,pdscores],axis=1)
    
    metric_modelos.columns=["gard_boost","decision_tree","random_forest","reg_logistic"]
    return metric_modelos   

# FUNCION 6#
def encode_data(df, list_le, list_dd,list_oe):    
    #Get dummies
    df_encoded=pd.get_dummies(df_encoded,columns=list_dd)
    
    # Ordinal Encoding
    oe = OrdinalEncoder()
    for col in list_oe:
        df_encoded[col] = oe.fit_transform(df_encoded[[col]])
    
    # Label Encoding
    le = LabelEncoder()
    for col in list_le:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    return df_encoded