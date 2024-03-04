## Paquetes requeridos
import pandas as pd
import numpy as np
import country_converter as coco
import pycountry_convert as pc
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import funciones as fn

#Dataframes
basefinal = 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/basefinal.csv'
df_bfinal = pd.read_csv(basefinal, sep=',')

df_bfinal.info() #la base final cuenta con 29 variables con 4410 datos en cada una

base16 = 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/baseprediccion.csv' #base para predicciones
df_pred = pd.read_csv(base16, sep = ',')
df_pred.info() #la base de prediccion cuenta con 29 variables con 4410 datos en cada una4

# Estucturacion variable objetivo (Attrition)
df_bfinal['retirement_reason'].value_counts() # 650 empleados han abandonado de la empresa o han sido despedidos
df_bfinal['retirement_reason'].isnull().sum() # 3760 empleados continuan trabajando en la empresa

'''Se estructura la variable objetivo Attrition, la cual se define como la salida de un empleado de la empresa:
    si el empleado ha abandonado por razones de salario,estres u otros. se le asigna 'Yes'
    si el empleado continua trabajando(valor nulo) en la empresa o ha sido despedido(fired) se le asigna 'No' '''
    
df_bfinal2 = df_bfinal.copy()
    
df_bfinal2['Attrition'] = df_bfinal2['retirement_reason'].replace({'Salary':'yes', 'Others':'yes', 'Stress':'yes', 'Fired':'no'}) #reemplazamos valores segun la categoria retirement_reason
df_bfinal2['Attrition'].value_counts()
df_bfinal2.isnull().sum()

df_bfinal3 =df_bfinal2.copy()

df_bfinal3.isnull().sum() 
df_bfinal3['Attrition'] = df_bfinal3['Attrition'].fillna('no') #rellenamos valores faltantes con 'no'
df_bfinal3['Attrition'].isnull().sum() #no hay valores faltantes

# Priorización de exploración de variables NUMERICAS'

###VER CLASE EN LA QUE SE HABLABA DE LAS FECHAS  DE PREDICCION Y ENTRENAMIENTO

#Transformacion de variables
"""Antes de realizar el modelo para priorizar las variables a explorar, vamos a realizar trannsformaciones de ciertas variables
para que sean mas entendibles y para que el modelo pueda entenderlas mejor"""
"""variables a convertir a categoricas:
Education, EnvironmentSatisfaction, JobInvolvement, JobSatisfaction, WorkLifeBalance,JobLevel"""

variables = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'WorkLifeBalance', 'JobLevel']
df_fin = df_bfinal3.copy()

df_fin = fn.transformaciontransformacion(df_fin, variables)

df_fin.info() # base con variable  objetivo estructurada

# Exploracion de variables que tengan nulos #
#se eliminaran las variables con nulos ya que los modelos a realizar para la seleccion de variables no aceptan nulos

df_fin.isnull().sum() 

v_nulas = ['NumCompaniesWorked', 'TotalWorkingYears', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance','mean_survery']

df_no_null = fn.nulos(df_fin, v_nulas)
df_no_null.isnull().sum()

df_no_null['Attrition'].value_counts()

# Imputar valores nulos #
# se imputaran los valores de la variable retirement_reason con 'no' ya que estos operarios suguen trabajando en la empresa

df_no_null['retirement_reason'] = df_no_null['retirement_reason'].fillna('NA')

"""DEFINIR MODELO""" #ANTES HAY QUE ELIMINAR LOS NULOS

# MODELO WRAPPER para seleccion de variables a explorar detalladamente #
## Backward selection

df_no_null['Attrition'] = df_no_null['Attrition'].replace({'yes':1, 'no':0}) #reemplazamos valores de la variable objetivo


b_wrpper = df_no_null.copy() #copiamos la base sin nulos
b_wrpper = b_wrpper.drop(['fecha_retiro','fecha_info'], axis = 1)

#Volver variables dummies
b_wrpper_d = pd.get_dummies(b_wrpper)
b_wrpper_d.head()


b_wrpper_int = b_wrpper.select_dtypes(include = ["number"]) # filtrar solo variables númericas
b_wrpper_int = b_wrpper_int.drop('Attrition', axis = 1) #quitamos la variable objetivo
y = b_wrpper_d['Attrition']#definimos variable objetivo

# Normalizacion variables numericas

b_wrpper_norm = b_wrpper_int.copy(deep = True)  # crear una copia del DataFrame
scaler = MinMaxScaler() # asignar el tipo de normalización
sv = scaler.fit_transform(b_wrpper_norm.iloc[:,:]) # normalizar los datos
b_wrpper_norm.iloc[:,:] = sv # asignar los nuevos datos
b_wrpper_norm.head()

#Estimador para seleccion de variables (modelo de regresion logistica)#

model = LogisticRegression() 

b_var = fn.recursive_feature_selection(b_wrpper_norm, y, model, 7) # seleccionar 10 variables

# Nuevo conjunto de datos a explorar
b_varp = b_wrpper_int.iloc[:,b_var]
b_varp.head(10)

###------------------------------------------------------------------------------------#########

# Create a Random Forest classifier
rf = RandomForestClassifier()

# Fit the model on the selected variables and the target variable
df_class = df_no_null.copy()
df_class['Attrition'] = df_class['Attrition'].replace({'yes':1, 'no':0})
df_class = df_class.drop(['fecha_retiro','fecha_info','EmployeeID'], axis = 1)

df_class_d = pd.get_dummies(df_class)

df_class_d2 = df_class_d.drop('Attrition', axis = 1)

y = df_class_d['Attrition']

# Normalizar df_class_d2
df_class_norm = df_class_d2.copy(deep = True)  # crear una copia del DataFrame
scaler = MinMaxScaler() # asignar el tipo de normalización
sv = scaler.fit_transform(df_class_norm.iloc[:,:]) # normalizar los datos
df_class_norm.iloc[:,:] = sv # asignar los nuevos datos
df_class_norm.head()

rf.fit(df_class_norm, y)

# Get the feature importances
importances = rf.feature_importances_

pd.DataFrame(importances, index = df_class_norm.columns, columns = ['importance']).sort_values('importance', ascending = False).head(14)


### EXPLORACION DE DATOS CON LAS VARIABLES SELECCIONADAS ###

"""Segun el modelo de regresion logistica y el modelo de random forest, las variables mas importantes para predecir la variable objetivo, son :
    - Age
    - TotalWorkingYears
    - YearsWithCurrManager
    - NumCompaniesWorked
    - PercentSalaryHike
    - mean_survery
    - retirmen_reason""" # 10 numericas