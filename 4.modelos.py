## MODELOS ###

import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier 
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.utils import resample
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.impute import SimpleImputer

# Cargar el DataFrame

data_seleccion= 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/base_seleccion2.csv'
df = pd.read_csv(data_seleccion,sep=',')
df.info() 

df.isnull().sum() # Verificar valores nulos

df1 = df.copy()
df1 = df1.drop('EmployeeID', axis = 1) # Eliminar columna EmployeeID ya que no aporta información relevante
df1.info()

df1['JobLevel'] = df1['JobLevel'].astype('object')
df1['Attrition'] = df1['Attrition'].astype('object')
df1['Attrition'].value_counts()# Cambiar valores de Attrition a 1 y 0
#Cambiamos el formato de las variables a int64 para poder trabajar con ellas en un solo formato
for i in df1.columns:
    if df1[i].dtypes == "float64":
        df1[i] = df1[i].astype('int64')
    else:
        pass

df1.dtypes # Verificar cambios

list_cat = [df1.columns[i] for i in range(len(df1.columns)) if df1[df1.columns[i]].dtype == 'object']
list_oe = ['JobLevel']
list_le = [df1.columns[i] for i in range(len(df1.columns)) if df1[df1.columns[i]].dtype == 'object' and len(df1[df1.columns[i]].unique()) == 2]
list_dd = ['Department','Education','EducationField','JobRole','MaritalStatus','BusinessTravel']

# Codificación de variables 

def encode_data(df, list_le,list_oe, list_dd):
    df_encoded = df1.copy()
    
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

df_encoded = encode_data(df1, list_le,list_oe,list_dd)

df3 = df_encoded.copy()


df3.dtypes

for i in df3.columns:
    if df3[i].dtypes == "float64":
        df3[i] = df3[i].astype('int64')
    else:
        pass

#Normalizacion

v_num = []
for col in df3.columns:
    if df3[col].dtypes == "int64":
        v_num.append(col)

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
for col in v_num:
    df3[[col]] = scaler.fit_transform(df3[[col]])

df3.head()

X_esc = df3.drop('Attrition', axis = 1)
y = df3['Attrition']

# Selecccion de modelos #

model_gb = GradientBoostingClassifier()
model_arb = DecisionTreeClassifier() 
model_log = LogisticRegression( max_iter=1000, random_state=42)
model_rand = RandomForestClassifier()

modelos  = list([model_gb,model_arb,model_log,model_rand])

# Seleccion de variables con base a los modelos seleccionados
def sel_variables(modelos,X,y,threshold):
    
    var_names_ac=np.array([])
    for modelo in modelos:
   
        modelo.fit(X,y)
        sel = SelectFromModel(modelo, prefit=True,threshold=threshold)
        var_names= modelo.feature_names_in_[sel.get_support()]
        var_names_ac=np.append(var_names_ac, var_names)
        var_names_ac=np.unique(var_names_ac)
    
    return var_names_ac


var_names= sel_variables(modelos,X_esc,y,threshold="3.1*mean") 
var_names.shape

# Al utiizar numeros menores se aceptaban mas variables sin embargo el desempeño seguia siendo el mosmo por lo cual
# Se utilizo un treshold de 2.8 en el cual se traba con 8 variables las cuales aportan significia a los modelos


df_var_sel = df3[var_names]
df_var_sel.info()

# Division data en train y test kfold-croos-validation #

df4 = df_var_sel.copy()

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline



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

f1sco_df = medir_modelos(modelos,"f1",X_esc,y,15)  #se definen 10 iteraciones para tener mejor visión del desempeño en el boxplot
f1dco_var_sel = medir_modelos(modelos,"f1",df4,y,15)


f1s=pd.concat([f1sco_df,f1dco_var_sel],axis=1) 
f1s.columns=['rlog', 'dtree', 'rforest', 'gboosting',
       'rl_Sel', 'dt_sel', 'rf_sel', 'gb_Sel']


f1s.plot(kind='box') ### gráfico para modelos sel y todas las variables

f1s.mean() 
"""En los boxplots podemos observar que el mejor desempeño con las variables seleccionadas y con todas las variables lo tiene decision tree.
Ademas de esto una pequeña diferencia de desempeño (2%) entre el modelo con todas las variables y las variables seleccionadas; se va a trabajar con 
el modelo de variables seleccionadas, sacrificando ese 2% de desempeño pero mejorando la interpretabilidad del modelo y ahorrando recursos computacionales."""

#Matriz confusion para rl_Sel
model_log = LogisticRegression( max_iter=1000, random_state=42)
model_log.fit(df4,y)

y_pred = model_log.predict(df4)
cm = confusion_matrix(y, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=model_log.classes_)
cmd.plot()

#matriz confusion para dt_sel
model_arb = DecisionTreeClassifier(max_depth=4, random_state=42)
model_arb.fit(df4,y)

y_pred = model_arb.predict(df4)
cm = confusion_matrix(y, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=model_arb.classes_)
cmd.plot()

#matriz confusion para gb_Sel
model_gb = GradientBoostingClassifier()
model_gb.fit(df4,y)

y_pred = model_gb.predict(df4)
cm = confusion_matrix(y, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=model_gb.classes_)
cmd.plot()




# Ajuste de hiperparametros del modelo ganador #

from sklearn.model_selection import RandomizedSearchCV

#Definicion de parametros para regresion logistica

parameters = {
    'penalty': ['l1', 'l2'], # penalización l1 y l2 ridge y lasso
    'fit_intercept': [True, False], # si se ajusta la intersección
    'max_iter': [100, 500, 1000]
} # max_leaf_nodes es el numero maximo de nodos finales

# create an instance of the randomized search object
r1 = RandomizedSearchCV(LogisticRegression(), parameters, cv=5, n_iter=100, random_state=42, n_jobs=-1, scoring='accuracy') 

r1.fit(df4,y) #df4 es el dataframe con las variables seleccionadas

resultados = r1.cv_results_
r1.best_params_
pd_resultados=pd.DataFrame(resultados)
pd_resultados[["params","mean_test_score"]].sort_values(by="mean_test_score", ascending=False)

rf_final=r1.best_estimator_ ### Guardar el modelo con hyperparameter tunning

# Exportar modelo ganador #
import joblib

joblib.dump(rf_final, "salidas\\rf_final.pkl") # Modelo ganador con afinamiento de hipermarametros 
joblib.dump(list_cat, "salidas\\list_cat.pkl") ### para realizar imputacion
joblib.dump(list_oe, "salidas\\list_oe.pkl")  ### para ordinal encoding
joblib.dump(list_le, "salidas\\list_le.pkl")  ### para label encoding
joblib.dump(list_dd, "salidas\\list_dd.pkl")  ### para dummies
joblib.dump(var_names, "salidas\\var_names.pkl") ### para variables con que se entrena modelo
joblib.dump(scaler, "salidas\\scaler.pkl") ## para normalizar datos con MinMaxScaler

#### convertir resultado de evaluacion entrenamiento y evaluacion en data frame para 

eval=cross_validate(rf_final,df4,y,cv=5,scoring='accuracy',return_train_score=True) 

train_rf=pd.DataFrame(eval['train_score'])
test_rf=pd.DataFrame(eval['test_score'])
train_test_rf=pd.concat([train_rf, test_rf],axis=1)
train_test_rf.columns=['train_score','test_score']
train_test_rf

train_test_rf["test_score"].mean()

