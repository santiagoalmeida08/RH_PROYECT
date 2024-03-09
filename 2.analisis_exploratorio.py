
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
from sklearn.decomposition import PCA

#Dataframes
basefinal = 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/basefinal.csv'
df_bfinal = pd.read_csv(basefinal, sep=',')

df_bfinal.info() #la base final cuenta con 29 variables con 4410 datos en cada una

base16 = 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/baseprediccion.csv' #base para predicciones
df_pred = pd.read_csv(base16, sep = ',')
df_pred.info() #la base de prediccion cuenta con 29 variables con 4410 datos en cada una4

# Estucturacion variable objetivo (Attrition); es la variable objetivo ya que se busca predecir si un empleado abandonará la empresa o no

'''Se estructurá la variable objetivo Attrition, la cual se define como la salida de un empleado de la empresa:
    si el empleado ha abandonado por razones de salario,estres u otros. se le asigna 'Yes'
    si el empleado continua trabajando(valor nulo) en la empresa o ha sido despedido(fired) se le asigna 'No' '''
    
df_bfinal2 = df_bfinal.copy()
    
df_bfinal2['Attrition'] = df_bfinal2['retirement_reason'].replace({'Salary':'yes', 'Others':'yes', 'Stress':'yes', 'Fired':'no'}).fillna('no')  #reemplazamos valores segun la categoria retirement_reason
df_bfinal2['Attrition'].value_counts()
df_bfinal2.isnull().sum()

# Exploracion de variables que tengan nulos 

df_fin = df_bfinal2.copy()
df_no_null = fn.nulos(df_fin) # se eliminan valores nulos que representen menos del 10% de la base
df_no_null.isnull().sum()

# Analisis de variables con mas del 10% de nulos #

df_no_null['retirement_reason'].value_counts() # empleados que abandonaron la empresa
df_no_null['retirement_reason'] = df_no_null['retirement_reason'].fillna('NA') # se imputaran valores nulos con 'NA' lo cual implica que el empleado sigue trabajando en la empresa

df_no_null2 = df_no_null.copy()
df_no_null2 = df_no_null2.drop(['fecha_retiro'], axis=1) # se elimina la variable fecha_retiro ya que solo fue util para union de bases
df_no_null2.info() # la base final cuenta con 27 variables y 4410 datos en cada una

# Exploración Variables Numercias #

df_expl_num = df_no_null2.copy()
df_expl_num = df_expl_num.select_dtypes(include=np.number).drop(['EmployeeID'], axis=1) # seleccionamos variables numericas y eliminamos EmployeeID ya que esta no aporta informacion relevante para la exploracion
df_expl_num.info()

df_expl_num.describe()
"""En esta tabla se pueden observar varios datos importantes de cada variable como lo son :
    - El 75% de los empleados tienen menos de 43 años lo cual nos indica una poblacion joven en la empresa
    - El 50% de los empleados de la compañia han trabajaodo menos de 5 años en la empresa, esto nos podria indicar una cantidad siginificativa de empleados nuevos
    - Se observa que en la compañia no se promueve de cargo con mucha frecuencia, esto se ve reflejado en la variable YearsSinceLastPromotion en la cual el 75% de los empleados no han sido promovido en los ultimos 3 años"""

df_expl_num.columns
#

bxp1 = df_expl_num.iloc[:,:2] # partimos la base en 3 partes para poder visualizar los boxplot de manera mas clara
bxp2 = df_expl_num.iloc[:,4:6]
bxp3 = df_expl_num.iloc[:,6:10]


plt.figure(figsize=(10, 6))
sns.boxplot(data=bxp1)
plt.title('Boxplot 1')
plt.xlabel('Variables')
plt.show()
""" 1. Boxplot variable edad: se puede observar que la mayoria de los empleados tienen entre 30 y 40 años, admeas no se observan valores atipicos
 
    2. Boxplot variable Distancia : se puede observar que el 75% de los empleados viven a menos de 10 km de la empresa, sin embargo el 15% restante vive a mas de 20 km de la empresa, lo cual podria ser un factor de desercion laboral"""

# Analizaremos las variables stockoptionlevel, monthlyincome y percentSalaryHike aparte debido a que su escala es diferente a las demas variables
plt.figure(figsize=(10, 6))
sns.boxplot(data= df_expl_num[['StockOptionLevel']])
plt.title('Boxplot 1')
plt.xlabel('Variables')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data= df_expl_num[['PercentSalaryHike']])
plt.title('Boxplot 1')
plt.xlabel('Variables')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=bxp2)
plt.title('Boxplot 2')
plt.xlabel('Variables')
plt.show()
""" 1. Boxplot variable total de años trabajando : el 75% de los empleados a trabajado menos de 15 años; esto se ve reflejadp con lo analizado en la edad
       con lo cual puede exisir una alta correlacion entre estas dos variables 
"""

plt.figure(figsize=(10, 6))
sns.boxplot(data=bxp3)
plt.title('Boxplot 3')
plt.xlabel('Variables')
plt.show()
""" Analisis de las variables relacionadas con el tiempo en la empresa: en los boxplots se puede observar que la mayoria de los empleados
    lleva poco tiempo en la empresa por lo cual aun no se han promovido. Tanto en la variable años en la compañia como en la variable referente a 
    la promocion hay datos atipicos correspondientes a empleados que llevan mas de 20 años en la empresa y aparentemente no han sido promovidos en un gran periodo de tiempo. 
    Así entonces puede que la baja frecuencia de promociones sea un importante factor de desercion laboral"""
    
plt.figure(figsize=(10, 6))
sns.boxplot(data= df_expl_num['MonthlyIncome']) 
plt.title('Boxplot de Salario Mensual')
plt.xlabel('Variables')
plt.show()

""" Boxplot de salario mensual: se puede observar que la mayoria de los empleados tienen un salario bajo, ademas se observan valores atipicos
    correspondientes a empleados con salarios muy altos que probablemente sean los empleados que llevan mas tiempo en la empresa, esto podria ser un factor de desercion laboral ya que los empleados con salarios bajos
    podrian sentirse desmotivados"""
    
# DIAGRAMAS VARIABLES UNIVARIADO #

# Histogramas #

for column in df_expl_num.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df_expl_num, x=column, kde=True)
    plt.title(f'Histograma de {column}')
    plt.xlabel(column)
    plt.ylabel('Frecuancia')
    plt.show()
    
"""La variable Age tiene una distribucion normal, con lo cual se puede inferir que la mayoria de empleados tienen entre 30 y 40 años"""
"""La variable MonthlyIncome tiene una distribucion asimetrica a la derecha, con lo cual se puede inferir que la mayoria de empleados tienen un salario bajo"""
"""La variable YearsAtCompany tiene una distribucion asimetrica a la derecha, con lo cual se puede inferir que la mayoria de empleados llevan pocos años en la empresa"""
"""La variable YearsInCurrentRole tiene una distribucion asimetrica a la derecha, con lo cual se puede inferir que la mayoria de empleados llevan poco tiempo en el cargo actual"""
"""La variable YearsSinceLastPromotion tiene una distribucion asimetrica a la derecha, con lo cual se puede inferir que la mayoria de empleados llevan poco tiempo desde su ultima promocion"""
"""La variable YearsWithCurrManager tiene una distribucion asimetrica a la derecha, con lo cual se puede inferir que la mayoria de empleados llevan poco tiempo con el actual gerente"""
"""La variable TotalWorkingYears tiene una distribucion asimetrica a la derecha, con lo cual se puede inferir que la mayoria de empleados llevan pocos años trabajando"""
"""La variable YearsSinceLastPromotion tiene una distribucion asimetrica a la derecha, con lo cual se puede inferir que la mayoria de empleados llevan poco tiempo desde su ultima promocion"""
"""La variable YearsWithCurrManager tiene una distribucion asimetrica a la derecha, con lo cual se puede inferir que la mayoria de empleados llevan poco tiempo con el actual gerente"""

# Analisis de correlacion entre variables numericas #

correlation = df_expl_num.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()