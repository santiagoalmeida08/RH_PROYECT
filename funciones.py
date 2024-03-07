"""Funciones que se utilizaran en el proyecto"""
from sklearn.feature_selection import RFE

#FUNCION 1#
def transformacion(df, variables):
    for i in variables:
        df[i] = df[i].astype('object')
    return df

#FUNCION 2#
# FUNCION NULOS#
def nulos(df,variables_nulas):
    for i in variables_nulas:
        if df[i].isnull().sum()/len(df) < 0.1:
            df = df.dropna(subset = [i]) 
        else:
            df = df[i]
    return df

# FUNCION 3# 

def recursive_feature_selection(X,y,model,k): #model=modelo que me va a servir de estimador para seleccionar las variables
                                              # K = numero de variables a seleccionar
  rfe = RFE(model, n_features_to_select=k, step=1) # step=1 significa que se eliminara una variable en cada iteracion
  fit = rfe.fit(X, y) # ajustar el modelo
  b_var = fit.support_ # seleccionar las variables
  print("Num Features: %s" % (fit.n_features_)) # numero de variables seleccionadas
  print("Selected Features: %s" % (fit.support_)) # variables seleccionadas
  print("Feature Ranking: %s" % (fit.ranking_)) # ranking de las variables

  return b_var 
