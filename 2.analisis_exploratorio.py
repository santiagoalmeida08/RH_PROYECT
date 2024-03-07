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

'''Se estructurá la variable objetivo Attrition, la cual se define como la salida de un empleado de la empresa:
    si el empleado ha abandonado por razones de salario,estres u otros. se le asigna 'Yes'
    si el empleado continua trabajando(valor nulo) en la empresa o ha sido despedido(fired) se le asigna 'No' '''
    
df_bfinal2 = df_bfinal.copy()
    
df_bfinal2['Attrition'] = df_bfinal2['retirement_reason'].replace({'Salary':'yes', 'Others':'yes', 'Stress':'yes', 'Fired':'no'}).fillna('no')  #reemplazamos valores segun la categoria retirement_reason
df_bfinal2['Attrition'].value_counts()
df_bfinal2.isnull().sum()

#Transformacion de variables

"""variables a convertir a categoricas:
Education, EnvironmentSatisfaction, JobInvolvement, JobSatisfaction, WorkLifeBalance,JobLevel"""

variables = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'WorkLifeBalance', 'JobLevel']
df_fin = df_bfinal2.copy()

df_fin = fn.transformacion(df_fin, variables) # se transforman las variables a categoricas

# Exploracion de variables que tengan nulos #

df_fin.isnull().sum() 

v_nulas = ['NumCompaniesWorked', 'TotalWorkingYears', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance','mean_survery']

df_no_null = fn.nulos(df_fin, v_nulas) # se eliminan valores nulos que representen menos del 10% de la base
df_no_null.isnull().sum()

# Analisis de variables con mas del 10% de nulos #

df_no_null['retirement_reason'].value_counts() # empleados que abandonaron la empresa
df_no_null['retirement_reason'] = df_no_null['retirement_reason'].fillna('NA') # se imputaran valores nulos con 'NA' lo cual implica que el empleado sigue trabajando en la empresa

df_no_null2 = df_no_null.copy()
df_no_null2 = df_no_null2.drop(['fecha_retiro','PercentSalaryHike'], axis=1) # se elimina la variable fecha_retiro ya que solo fue util para union de bases
df_no_null2.info() # la base final cuenta con 27 variables y 4410 datos en cada una

# Exploración Variables Numercias #
df_expl = df_no_null2.copy()
df_expl = df_expl.select_dtypes(include=np.number) 
df_expl.columns

# ANALISIS ATIPICOS #
"""eliminar variables como employeeid, Percentajesalary . Disminuir el tamaño de letra a los boxplot
o reagrupar las variables para que se vean mejor"""

bxp1 = df_expl.iloc[:,:7] 
bxp2 = df_expl.iloc[:,7:13]
#bxp3 = df_expl.iloc[:]

plt.figure(figsize=(10, 6))
sns.boxplot(data=bxp1)
plt.title('Boxplot 1')
plt.xlabel('Variables')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=bxp2)
plt.title('Boxplot 2')
plt.xlabel('Variables')
plt.show()

# Analisis Variable Objetivo #

plt.figure(figsize=(8, 6))
sns.countplot(data=df_expl, x='Attrition')
plt.title('Attrition Count')
plt.xlabel('Attrition')
plt.ylabel('Count')
plt.show()

# DIAGRAMAS VARIABLES UNIVARIADO #

# Histogramas #

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('Histogramas Variables')
sns.histplot(df_expl['Age'], kde=True, ax=axes[0, 0])
sns.histplot(df_expl['TotalWorkingYears'], kde=True, ax=axes[0, 1])
sns.histplot(df_expl['YearsWithCurrManager'], kde=True, ax=axes[0, 2])
sns.histplot(df_expl['NumCompaniesWorked'], kde=True, ax=axes[1, 0])
#sns.histplot(df_expl['PercentSalaryHike'], kde=True, ax=axes[1, 1])
sns.histplot(df_expl['mean_survery'], kde=True, ax=axes[1, 2])
sns.histplot(df_expl['TrainingTimesLastYear'], kde=True, ax=axes[2, 0])
sns.histplot(df_expl['YearsSinceLastPromotion'], kde=True, ax=axes[2, 1])
sns.histplot(df_expl['YearsAtCompany'], kde=True, ax=axes[2, 2])
plt.show()

# Analisis de correlacion entre variables numericas

correlation = df_expl.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()