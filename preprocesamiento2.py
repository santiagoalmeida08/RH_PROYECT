"""PREPROCESAMIENTO BASE employee"""

#1. Cargamos los paquetes necesaeios para la limpieza

import pandas as pd
import numpy as np
import datetime
import funciones as fn
#2. Cargamos las bases de datos requeridas

data_employee = 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/employee_survey_data.csv'

df_empl = pd.read_csv(data_employee,sep=';') ## BASE ORIGINAL ##

df_empl.info() 

"""se puede observar que el dataframe tiene una columna sobrante que replica la funcion del EmployeeID
por lo cual se procedera a eliminarla, ademas existen variables con datos nulos que posteriormente se analizara su situacion"""

df_empl2 = df_empl.drop('Unnamed: 0',axis=1)

df_empl2.info() # podemos observar que las unicas variables sin nulos son emplyeeid y datasurvey

# En caso de que se encuentren datos nulos con un bajo porcentaje de representatividad se procedera a eliminarlos#

df_empl2 = fn.nulos(df_empl2)
df_empl2.isnull().sum()

# TRANSFORMACION DE VARIABLES #

"""Se cambiara el formato de fecha a datetime, y el resto de las variables a categoricas, ya que 
al tratarse de encuestas, se puede reorganizar la categorizacion """

df_empl4 = df_empl2.copy()

df_empl4['DateSurvey'] = pd.to_datetime(df_empl4['DateSurvey'])#transformacion de la variable fecha a datetime
df_empl4['EmployeeID'] = df_empl4['EmployeeID'].astype('object')#transformacion de la variable employeeid a object

df_empl4.info()
df_empl4['EnvironmentSatisfaction'].value_counts()#se puede observar que las variables se han transformado correctamente
df_empl4['JobSatisfaction'].value_counts()
df_empl4['WorkLifeBalance'].value_counts()

df_empl5 = df_empl4.copy()
df_empl5 = df_empl5.rename(columns= {'DateSurvey':'fecha'})# se quiere que en todas las bases la variable fecha tenga el mismo nombre
df_empl5.info() # BASE FINAL EMPL #


"""#PREPROCESAMIENTO BASE retirement"""

# CARGA DE BASE DE DATOS #

retirements = 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/retirement_info.csv'

df_req = pd.read_csv(retirements, sep= ';')
df_req

# LIMPIEZA BASE DE DATOS #

df_req.info() 

''' Se deben borrar las columnas 1 y 2; ademas tratar los datos nulos de la variables 
y pasar retirementDate a formato fecha '''

df_req1 = df_req.drop(['Unnamed: 0.1','Unnamed: 0'], axis= 1)
df_req1

df_req1['retirementDate'] = pd.to_datetime(df_req1['retirementDate'], format='%d/%m/%Y')
df_req1['EmployeeID'] = df_req1['EmployeeID'].astype('object')
df_req1.info()

# Evaluación variables categoricas #
df_req1['Attrition'].value_counts() #variable referente a abandonar la empresa
df_req1['retirementType'].value_counts() # las personas se van de la empresa por resignacion o por despido
df_req1['resignationReason'].value_counts() # razones de resignacion

# Analisis de datos nulos
df_req1.isnull().sum() # se puede observar que la variable resignationReason tiene 70 datos nulos
                        #los cuales coinciden con los 70 empleados que fueron despedidos   


# Para tratar los nulos vamos a crear una nueva variable que nos permita explicar los valores nulos de resignationReason
#los cuales seran reemplazados por la categoria fired

df_req3 = df_req1.copy()

df_req3['resignationReason'] = df_req3['resignationReason'].fillna('Fired')
df_req3 = df_req3.drop('retirementType', axis=1)

df_req3 = df_req3.rename(columns= {'resignationReason':'retirement_reason'})
df_req3['retirement_reason'].value_counts()

df_ret4 = df_req3.copy()
df_ret4 = df_ret4.rename(columns= {'retirementDate':'fecha'})# se quiere que en todas las bases la variable fecha tenga el mismo nombre

df_ret4 # BASE FINAL RETIREMENTS #


"""#PREPROCESAMIENTO BASE general data """
#Carga base de datos 
data_general= 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/general_data.csv'
df_g1= pd.read_csv(data_general, sep=';')#base de datos
#Limpieza base de datos
df_g1.info()
"""La columna unnamed: 0 cuenta las filas iniciando desde el cero, esta columna no aporta ninguna información relevante por lo cual se va eliminar"""
df_g2= df_g1.drop('Unnamed: 0', axis=1)
df_g2
df_g2[df_g2.isnull().any(axis=1)]#analisis de datos nulos 
df_g2.isnull().sum()
"""Hay 56 datos nulos que pertenecen a las columnas de NumCompaniesWorked y TotalWorkingYears debido a que el numero de nulos es pequeño respecto a la cantidad 
de datos que contiene la base de datos y ademas es importante conocer la información de todos los empleados por lo cual por ahora no se van a eliminar las filas 
que contienen estos datos nulos """

df_g2 = fn.nulos(df_g2) # eliminamos nulos 

df_g2[df_g2.duplicated()]#filtra duplicados 
#no se encuentran datos duplicados
df_g3=df_g2.copy()
"""
#ANALIZAR VARIABLES Y CATEGORIAS 
df_g3['EmployeeCount'].value_counts() #SOLO TIENE UNA CATEGORIA POR LO CUAL SE TOMA LA DECISIÓN DE ELIMINAR ESTA VARIABLE
df_g3['Over18'].value_counts() #solo tiene una categoria por lo cual se toma la decision de eliminar esta variable
df_g3['StandardHours'].value_counts() # todos los valores son iguales, tambien se toma la decision de eliminar esta variable

df_g3['JobLevel'].value_counts()#esta variable se va a convertir a variable categorica con 5 categorias
df_g3['Education'].value_counts()#Se va a convertir a una variable categorica de 5 categorias
df_g3['NumCompaniesWorked'].value_counts()#Se va a convertir a una variable categorica agrupando los valores
df_g3['TrainingTimesLastYear'].value_counts()#Se va a convertir a una variable categorica agrupando los valores

#Transformacion de variables

#va3 = ['StockOptionLevel', 'JobLevel', 'Education', 'NumCompaniesWorked', 'TotalWorkingYears','EmployeeID']
va3=['EmployeeID']
df_g3 = fn.transformacion(df_g3, va3)


df_g3['NumCompaniesWorked'] = df_g3['NumCompaniesWorked'].replace({0:'Al menos 2 empresas', 1:'Al menos 2 empresas', 2:'Al menos 2 empresas', 
                                                                  3:'De 3 a 5 empresas', 4:'De 3 a 5 empresas', 5:'De 3 a 5 empresas', 
                                                                  6:'Mas de 5 empresas', 7:'Mas de 5 empresas', 8:'Mas de 5 empresas', 9:'Mas de 5 empresas'})


df_g3['TrainingTimesLastYear'] = df_g3['TrainingTimesLastYear'].replace({0:'Ningun entrenamiento', 
                                                                   1:'Al menos 3 semanas', 2:'Al menos 3 semanas', 3:'Al menos 3 semanas', 
                                                                   4:'De 4 a 5 semanas', 5:'De 4 a 5 semanas', 6:'De 4 a 5 semanas'})

df_g3['Education'] = df_g3['Education'].replace({1:'Escuela secundaria', 2:'Licenciatura', 3:'Maestria', 4:'Doctorado', 5:'Posdoctorado'})

df_g3['JobLevel'] = df_g3['JobLevel'].replace({1:'Nivel 1', 2:'Nivel 2', 3:'Nivel 3', 4:'Nivel 4', 5:'Nivel 4'}) # se entiende a joblevel como el nivel jerarquico del empleado


df_g4= df_g3.drop(['EmployeeCount','Over18', 'StandardHours'], axis=1)#eliminar variables que no son representativas para el analisis
df_g4= df_g4.rename(columns= {'InfoDate':'fecha'})# se quiere que en todas las bases la variable fecha tenga el mismo nombre

df_g4['NumCompaniesWorked'].value_counts()
df_g4['TrainingTimesLastYear'].value_counts()
df_g4['Education'].value_counts()
"""
df_g4 = df_g3.copy()
df_g4= df_g4.rename(columns= {'InfoDate':'fecha'})
#Como la fecha se encuentra en formato object vamos a convertirlo en formato fecha
df_g4['fecha']=pd.to_datetime(df_g4['fecha'])
df_g4.info()# base final


"""#PREPROCESAMIENTO BASE manager survey """ #ENCUESTA DE DESEMPEÑO
#Carga base de datos 
data_manager= 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/manager_survey.csv'
df_man1= pd.read_csv(data_manager, sep=';')#base de datos
#Limpieza base de datos
df_man1.info()
"""La columna unnamed: 0 cuenta las filas iniciando desde el cero, esta columna no aporta ninguna información relevante por lo cual se va eliminar"""
df_man2= df_man1.drop('Unnamed: 0', axis=1)
df_man2
df_man2.isnull().sum()#analisis de datos nulos 
"""La base no contiene datos nulos"""


df_man2[df_man2.duplicated()]#filtra duplicados 
#no se encuentran datos duplicados
df_man3=df_man2.copy()
df_man3
#Como la fecha se encuentra en formato object vamos a convertirlo en formato fecha
df_man3["SurveyDate"]=pd.to_datetime(df_man3['SurveyDate'], format="%d/%m/%Y")
df_man3['EmployeeID'] = df_man3['EmployeeID'].astype('object')
df_man3 = df_man3.rename(columns= {'SurveyDate':'fecha'})# se quiere que en todas las bases la variable fecha tenga el mismo nombre

#ANALISIS DE VARIABLES Y CATEGORIAS 

df_man3['JobInvolvement'].value_counts()
df_man3['PerformanceRating'].value_counts()

#df_man3['JobInvolvement'] = df_man3['JobInvolvement'].replace({1:'Bajo', 2:'Bajo', 3:'Medio', 4:'Alto'})
#df_man3['PerformanceRating'] = df_man3['PerformanceRating'].replace({3:'Bajo', 4:'Alto'})

df_man3.info() 

#Traer solo la información necesaria
#RETIRENMENT: En esta base de datos solo se necesita la información de aquellos trabjadores que salieron en 2016 por lo cual solo traeremos los datos referentes a este año 
ret_16 = df_ret4[df_ret4['fecha'].dt.year == 2016]#Base general data con los datos del 2016

# UNIR BASES DE DATOS ANTERIOMENTE DEPURADAS #
"""se van a unir las bases partiendo del employeeID y la fecha para evitar que se dupliquen los datos, en todas las bases se usaran los datos tanto del 2015 y del 2016
menos en la base de retirenment"""

df_g4.info()
base= df_g4.merge(df_empl5, how= 'left' , on=['EmployeeID', 'fecha']).merge(df_man3, how='left', on=['EmployeeID', 'fecha'])#base con la union de todas las tablas sin la tabla de retirement
#base.info()


base15 = base[base['fecha'].dt.year == 2015]#separar la tabla base solo con los datos del 2015
base16 = base[base['fecha'].dt.year == 2016]#separar la tabla base solo con los datos del 2016, esta base se usara para la predicción para el año 2017
basefinal= pd.merge(base15, ret_16, how= 'left', on= 'EmployeeID')#Union de la tabla con los datos del 2015 con la base retirement que contiene la variable respuesta

basefinal = basefinal.rename(columns= {'fecha_x':'fecha_info', 'fecha_y':'fecha_retiro' })#Renombrar las columnas de las fechas para mejor interpretabilidad

#basefinal.to_csv('data_hr_proyect/basefinal2.csv', index= False)

#base16.to_csv('data_hr_proyect/baseprediccion2.csv', index= False)


############### ANALISIS EXPLORATORIO DE DATOS ####################

# Quitamos las variables que no van a servir 
base = basefinal.drop(['fecha_info', 'fecha_retiro'], axis=1)

base.dtypes

# Variable Objetivo #
df_bfinal2 = base.copy()
    
df_bfinal2['Attrition'] = df_bfinal2['retirement_reason'].replace({'Salary':'yes', 'Others':'yes', 'Stress':'yes', 'Fired':'no'}).fillna('no')  #reemplazamos valores segun la categoria retirement_reason
df_bfinal2['Attrition'].value_counts()

import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='Attrition', data=df_bfinal2, palette='hls')
plt.show()


df_bfinal2['retirement_reason'] = df_bfinal2['retirement_reason'].fillna('no aplica') 
df_bfinal2 = fn.nulos(df_bfinal2)
df_bfinal2.isnull().sum()

#Variables numericas #
df_bfinal2.hist(bins=20, figsize=(20,15))

# Eliminacion variables #
df_bfinal3 = df_bfinal2.drop(['EmployeeCount','StandardHours','PerformanceRating'], axis=1) 

#Categoroizacion de variables #
df_bfinal3['Education'] = df_bfinal3['Education'].replace({'Escuela secundaria':1, 'Licenciatura':2, 'Maestria':3, 'Doctorado':4, 'Posdoctorado':5})
df_bfinal3['TrainingTimesLastYear'] = df_bfinal3['TrainingTimesLastYear'].replace({0:'Ningun entrenamiento', 
                                                                   1:'Al menos 3 semanas', 2:'Al menos 3 semanas', 3:'Al menos 3 semanas', 
                                                                   4:'De 4 a 5 semanas', 5:'De 4 a 5 semanas', 6:'De 4 a 5 semanas'})
df_bfinal3['TrainingTimesLastYear'].value_counts()
#Correlacion 
correlation = df_bfinal3.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, annot=True, cmap='coolwarm')

# analisis atipicos #
for i in df_bfinal3.columns:
    if df_bfinal3[i].dtype == 'int64':
        sns.boxplot(df_bfinal3[i])
        plt.show()

#Monthly income
#years at company
#years since last promotion
#years with current manager

df4 = df_bfinal3.copy()

# Tratamiento de atipicos #
for column in df4.columns:
    if df4[column].dtype != 'object':
        Q1 = df4[column].quantile(0.25)
        Q3 = df4[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df4[(df4[column] < lower_bound) | (df4[column] > upper_bound)]
        print(f"Variable: {column}, Number of outliers: {len(outliers)}")

def impute_outliers(df):
    # Imputar datos atipicos MonthlyIncome
    mean = df['MonthlyIncome'].mean()
    std = np.std(df['MonthlyIncome'])
    df['MonthlyIncome'] = np.where(df['MonthlyIncome'] > 3 * std, mean, df['MonthlyIncome'])

    #Impotar datos atipicos NumCompaniesWorked
    std = np.std(df['NumCompaniesWorked'])
    mode_num_companies = df['NumCompaniesWorked'].mode().values[0]
    df['NumCompaniesWorked'] = np.where(df['NumCompaniesWorked'] > 3* std, mode_num_companies, df['NumCompaniesWorked']) 

    # Imputing outliers in 'StockOptionLevel' with mode
    std = np.std(df['StockOptionLevel'])
    mode_stock_option = df['StockOptionLevel'].mode().values[0]
    df['StockOptionLevel'] = np.where(df['StockOptionLevel'] > 3*std, mode_stock_option, df['StockOptionLevel'])

    #Imputar TotalWorkingYears
    mean = df['TotalWorkingYears'].mean()
    std = np.std(df['TotalWorkingYears'])
    df['TotalWorkingYears'] = np.where(df['TotalWorkingYears'] > 3 * std, mean, df['TotalWorkingYears'])

    #Imputar YearsAtCompany
    mean = df['YearsAtCompany'].mean()
    std = np.std(df['YearsAtCompany'])
    df['YearsAtCompany'] = np.where(df['YearsAtCompany'] > 3 * std, mean, df['YearsAtCompany'])

    #Imputar YearsSinceLastPromotion
    mean = df['YearsSinceLastPromotion'].mean()
    std = np.std(df['YearsSinceLastPromotion'])
    df['YearsSinceLastPromotion'] = np.where(df['YearsSinceLastPromotion'] > 3 * std, mean, df['YearsSinceLastPromotion'])

    #Imputar YearsWithCurrManager
    mean = df['YearsWithCurrManager'].mean()
    std = np.std(df['YearsWithCurrManager'])
    df['YearsWithCurrManager'] = np.where(df['YearsWithCurrManager'] > 3 * std, mean, df['YearsWithCurrManager'])
    
    return df

df6 = impute_outliers(df4)

sns.boxplot(x='Attrition', y='MonthlyIncome', data=df6)

df5 = df4.copy()
df5 = df5.select_dtypes(include='object')
# Analisis Categoricos #

import plotly.express as px 
for column in df5.columns:
    if column != 'Attrition': # se excluye la variable objetivo ya que esta se analizara por separado
        base = df5.groupby([column])[['Attrition']].count().reset_index().rename(columns ={'Attrition':'count'})
        fig = px.pie(base, names=column, values='count', title= column)
        xaxis_title = column
        yaxis_title = 'Cantidad'
        template = 'simple_white'
        fig.show()

# se eliminaran las variables con categorias poco representativas que son :
df6 = df4.copy() 
df6 = df6.drop(['EmployeeID','JobRole','Over18','retirement_reason'], axis=1)


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

# BASE DF6 #

df1 = df6.copy()

df1.EnvironmentSatisfaction = df1.EnvironmentSatisfaction.astype(int)
df1.JobSatisfaction  = df1.JobSatisfaction.astype(int)
df1.WorkLifeBalance = df1.WorkLifeBalance.astype(int)
df1.NumCompaniesWorked = df1.NumCompaniesWorked.astype(int)
df1.TotalWorkingYears = df1.TotalWorkingYears.astype(int)



list_cat = [df1.columns[i] for i in range(len(df1.columns)) if df1[df1.columns[i]].dtype == 'object']
#list_oe = ['Education']
list_le = [df1.columns[i] for i in range(len(df1.columns)) if df1[df1.columns[i]].dtype == 'object' and len(df1[df1.columns[i]].unique()) == 2]
#list_dd = ['Department','Education','EducationField','JobRole','MaritalStatus','NumCompaniesWorked','YearsSinceLastPromotion']
list_dd = [df1.columns[i] for i in range(len(df1.columns)) if df1[df1.columns[i]].dtype == 'object' and len(df1[df1.columns[i]].unique()) > 2]

# DUMMIES #
df_encoded=pd.get_dummies(df1,columns=list_dd)
def encode_data(df, list_le, list_dd):
    df_encoded = df1.copy()
    
    #Get dummies
    df_encoded=pd.get_dummies(df_encoded,columns=list_dd)
    
    #Ordinal Encoding
    #oe = OrdinalEncoder()
    #for col in list_oe:
     #   df_encoded[col] = oe.fit_transform(df_encoded[[col]])
    
    # Label Encoding
    le = LabelEncoder()
    for col in list_le:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    return df_encoded

df_encoded = encode_data(df1, list_le,list_dd)

df3 = df_encoded.copy()
df3


for i in df3.columns:
    if df3[i].dtypes == "float64":
        df3[i] = df3[i].astype("int64")

df3.dtypes

v_num = []
for col in df3.columns:
    if df3[col].dtypes == "int64":
        v_num.append(col)

scaler = MinMaxScaler()
for col in v_num:
    df3[[col]] = scaler.fit_transform(df3[[col]])

df3

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
#from sklearn.pipeline import make_pipeline



def medir_modelos(modelos,scoring,X,y,cv):
    os = RandomOverSampler()
    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        #pipeline = make_pipeline(os, modelo)
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


parameters = {'criterion':['gini','entropy'],
              'max_depth': [3,5,10,15], # mex_depth es la profundidad del arbol
              'min_samples_split': [2,4,5,10], # min_samples_split es el numero minimo de muestras que se requieren para dividir un nodo
              'max_leaf_nodes': [5,10,15,20]} # max_leaf_nodes es el numero maximo de nodos finales

# create an instance of the randomized search object
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

r1 = RandomizedSearchCV(DecisionTreeClassifier(), parameters, cv=5, n_iter=100, random_state=42, n_jobs=-1, scoring='f1') 

r1.fit(df4,y)

resultados = r1.cv_results_
r1.best_params_
pd_resultados=pd.DataFrame(resultados)
pd_resultados[["params","mean_test_score"]].sort_values(by="mean_test_score", ascending=False)

rf_final=r1.best_estimator_ ### Guardar el modelo con hyperparameter tunning


eval=cross_validate(rf_final,df4,y,cv=5,scoring='f1',return_train_score=True) 

train_rf=pd.DataFrame(eval['train_score'])
test_rf=pd.DataFrame(eval['test_score'])
train_test_rf=pd.concat([train_rf, test_rf],axis=1)
train_test_rf.columns=['train_score','test_score']
train_test_rf

train_test_rf["test_score"].mean()
