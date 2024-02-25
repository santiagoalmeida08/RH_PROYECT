"""PREPROCESAMIENTO BASE employee"""

#1. Cargamos los paquetes necesaeios para la limpieza

import pandas as pd
import numpy as np

#2. Cargamos las bases de datos requeridas

data_employee = 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/employee_survey_data.csv'

df_empl = pd.read_csv(data_employee,sep=';') ## BASE ORIGINAL ##

df_empl.info() 
"""se puede observar que el dataframe tiene una columna sobrante que replica la funcion del EmployeeID
por lo cual se procedera a eliminarla, ademas existen variables con datos nulos que posteriormente se analizara su situacion"""

df_empl2 = df_empl.drop('Unnamed: 0',axis=1)

df_empl2.info()

##FILTRANMOS DATOS DONDE HAYAN NULOS#

df_empl2[df_empl2.isnull().any(axis=1)]

166/8820 
"""los datos nulos que hay en el dataframe representan un 1.8 por ciento de la totalidad de los datos
por lo cual consideramos que no son significativos"""

## ELIMINAMOS LOS DATOS NULOS ##

df_empl3 = df_empl2.dropna()

"""No se considera analizar datos duplicados ni atipicos ya que al aplicarse esta encuesta al varios empleados
en una misma fecha se repiten las fechas como tal y tambien puede que en algun punto se repitan las calificaciones."""


df_empl3[df_empl3['EmployeeID'] == 445] ## EN AMBOS AÑOS SE OBTIENEN LAS MISMAS PUNTUACIONES

## TRANSFORMACION DE VARIABLES Y FORMATO DE LAS MISMAS ##

df_empl3.info() 
"""Hay que cambiar el formato de la fecha y el del Employee ID"""

df_empl4 = df_empl3.copy()
df_empl4['EmployeeID'] = df_empl3['EmployeeID'].astype('object')
df_empl4['DateSurvey'] = pd.to_datetime(df_empl3['DateSurvey'],format= '%d/%m/%Y') #formato ????

df_empl4


"""#PREPROCESAMIENTO BASE requierments"""


"""#PREPROCESAMIENTO BASE 3"""


"""#PREPROCESAMIENTO BASE 4"""
