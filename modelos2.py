
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

# Cargar el DataFrame


data_seleccion= 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/base_seleccion.csv'

df = pd.read_csv(data_seleccion,sep=',') ## BASE ORIGINAL ##

df.isnull().sum()
df = df.drop('EmployeeID', axis = 1)

df1 = df.copy()
df1.info()
list_cat = [df1.columns[i] for i in range(len(df1.columns)) if df1[df1.columns[i]].dtype == 'object']
list_oe = ['StockOptionLevel','EnvironmentSatisfaction','JobSatisfaction','WorkLifeBalance','PercentSalaryHike','JobLevel','JobInvolvement']
list_le = [df1.columns[i] for i in range(len(df1.columns)) if df1[df1.columns[i]].dtype == 'object' and len(df1[df1.columns[i]].unique()) == 2]
list_dd = ['Department','Education','EducationField','JobRole','MaritalStatus','NumCompaniesWorked','YearsSinceLastPromotion']

# DUMMIES #

def encode_data(df, list_oe, list_le, list_dd):
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

df_encoded = encode_data(df1, list_oe, list_le,list_dd)

df3 = df_encoded.copy()

"""
#LabelEncoder

df2 = df1.copy()   
le = LabelEncoder() # 0 = No y 1 = Yes

for i in df2.columns:
    if df2[i].dtype == 'object' and len(df2[i].unique()) == 2:  
        df2[i] = le.fit_transform(df2[i])
    else:
        df2


# Ordinal Encoding
ord =  ['StockOptionLevel','EnvironmentSatisfaction','JobSatisfaction','WorkLifeBalance','PercentSalaryHike','JobLevel','JobInvolvement']
oe = OrdinalEncoder()

for i in ord : 
    df2[i]=oe.fit_transform(df2[[i]])


#Get Dummies
df3 = pd.get_dummies(df2)
df3.columns"""

#Normalizacion

v_num = []
for col in df3.columns:
    if df3[col].dtypes == "int64":
        v_num.append(col)

scaler = MinMaxScaler()
for col in v_num:
    df3[[col]] = scaler.fit_transform(df[[col]])

df3.head()

X_esc = df3.drop('Attrition', axis = 1)
y = df3['Attrition']

# Selecccion de modelos #

# Gradient Boosting
# Arboles de desicion 
# Regresión Logistica
# Random Forest

model_gb = GradientBoostingClassifier()
model_arb = DecisionTreeClassifier(class_weight='balanced', max_depth=4, random_state=42)
model_log = LogisticRegression()
model_rand = RandomForestClassifier(n_estimators = 100,#o regresation
                               criterion = 'gini',#error
                               max_depth = 5,#cuantos arboles
                               max_leaf_nodes = 10,#profundidad
                               max_features = None,#nodos finales
                               oob_score = False,
                               n_jobs = -1,
                               random_state = 123)

modelos  = list([model_gb,model_arb,model_log,model_rand])

# Seleccion de variables con base a los modelos seleccionados
def sel_variables(modelos,X,y,threshold):
    
    var_names_ac=np.array([])
    for modelo in modelos:
        #modelo=modelos[i]
        modelo.fit(X,y)
        sel = SelectFromModel(modelo, prefit=True,threshold=threshold)
        var_names= modelo.feature_names_in_[sel.get_support()]
        var_names_ac=np.append(var_names_ac, var_names)
        var_names_ac=np.unique(var_names_ac)
    
    return var_names_ac


var_names= sel_variables(modelos,X_esc,y,threshold="2.8*mean") 
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

f1sco_df = medir_modelos(modelos,"f1",X_esc,y,10)  #se definen 10 iteraciones para tener mejor visión del desempeño en el boxplot
f1dco_var_sel = medir_modelos(modelos,"f1",df4,y,10)


f1s=pd.concat([f1sco_df,f1dco_var_sel],axis=1) 
f1s.columns=['rlog', 'dtree', 'rforest', 'gboosting',
       'rl_Sel', 'dt_sel', 'rf_sel', 'gb_Sel']


f1sco_df.plot(kind='box') #### gráfico para modelos todas las varibles
f1dco_var_sel.plot(kind='box') ### gráfico para modelo variables seleccionadas
f1s.plot(kind='box') ### gráfico para modelos sel y todas las variables

f1s.mean() 
"""En los boxplots podemos observar que el mejor desempeño con las variables seleccionadas y con todas las variables lo tiene decision tree.
Ademas de esto una pequeña diferencia de desempeño (2%) entre el modelo con todas las variables y las variables seleccionadas; se va a trabajar con 
el modelo de variables seleccionadas, sacrificando ese 2% de desempeño pero mejorando la interpretabilidad del modelo y ahorrando recursos computacionales."""


# Ajuste de hiperparametros del modelo ganador DecisionTreeClasifier #
from scipy.stats import uniform, poisson
from sklearn.model_selection import RandomizedSearchCV

# setup parameter space

parameters = {'criterion':['gini','entropy'],
              'max_depth': [3,5,10,15], # mex_depth es la profundidad del arbol
              'min_samples_split': [2,4,5,10], # min_samples_split es el numero minimo de muestras que se requieren para dividir un nodo
              'max_leaf_nodes': [5,10,15,20]} # max_leaf_nodes es el numero maximo de nodos finales

# create an instance of the randomized search object
r1 = RandomizedSearchCV(DecisionTreeClassifier(), parameters, cv=5, n_iter=100, random_state=42, n_jobs=-1, scoring='f1') 

r1.fit(df4,y)

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

eval=cross_validate(rf_final,df4,y,cv=5,scoring='f1',return_train_score=True) 

train_rf=pd.DataFrame(eval['train_score'])
test_rf=pd.DataFrame(eval['test_score'])
train_test_rf=pd.concat([train_rf, test_rf],axis=1)
train_test_rf.columns=['train_score','test_score']
train_test_rf

train_test_rf["test_score"].mean()
