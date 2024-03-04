"""PREPROCESAMIENTO BASE employee"""

#1. Cargamos los paquetes necesaeios para la limpieza

import pandas as pd
import numpy as np
import datetime
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

df_empl5 = df_empl4.copy()

"""crearemos una nueva variable en la cual se reflejara el promedio de la encuesta realizada
a cada uno de los trabajadores"""

df_empl5.info()

df_empl5['mean_survery'] = ((df_empl5['EnvironmentSatisfaction']+df_empl5['JobSatisfaction']+df_empl5['WorkLifeBalance']) /3).round(1)
#Como la fecha se encuentra en formato object vamos a convertirlo en formato fecha
df_empl5["DateSurvey"]=pd.to_datetime(df_empl5['DateSurvey'], format="%d/%m/%Y")
df_empl5= df_empl5.rename(columns= {'DateSurvey':'fecha'})# se quiere que en todas las bases la variable fecha tenga el mismo nombre

df_empl5 # BASE FINAL EMPL #


"""#PREPROCESAMIENTO BASE retirement"""

# CARGA DE BASE DE DATOS #

retirements = 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/retirement_info.csv'

df_req = pd.read_csv(retirements, sep= ';')
df_req

# LIMPIEZA BASE DE DATOS #

df_req.info() 

''' Se deben borrar las columnas 1 y 2; ademas tratar los datos nulos de la variable resignationReason  
y pasar retirementDate a formato fecha '''

df_req1 = df_req.drop(['Unnamed: 0.1','Unnamed: 0'], axis= 1)
df_req1

df_req1['retirementDate'] = pd.to_datetime(df_req1['retirementDate'], format='%d/%m/%Y')
df_req1['EmployeeID'] = df_req1['EmployeeID'].astype('object')
df_req1.info()

#Evaluación variables categoricas#
df_req1['Attrition'].value_counts() 
df_req1['retirementType'].value_counts()
df_req1['resignationReason'].value_counts()



"""-Convertiremosla variable retirementType a binaria """

df_req2 = df_req1.copy()

df_req2['retirementType'].value_counts()
df_req2.info()

df_req2

#VAMOS A CREAR UNA NUEVA VARIABLE EN DONDE SE RESUMA EL TIPO DE DESPIDO; SUS CATEGORIAS SERAN#
#FIRED,SALAY,STRESS,OTHERS

df_req3 = df_req2.copy()

df_req3[(df_req3['retirementType']=='Fired') & (df_req3['resignationReason'] == 'Stress')]
"""En este caso los valores nulos se ven explicados porque los 70 empleados que fueron despedidos coinciden con los campos nulos
de la variable resignationReason;pues al ser una variable derivada de la categoria retirmentType no va a tener datos en dicha variable"""


"""Vamos a juntar las categorias fired,salary,stress y others en una nueva variable llamada retirement_reason """
df_req3['resignationReason'] = df_req3['resignationReason'].fillna('Fired')
df_req3 = df_req3.drop('retirementType', axis=1)

df_req3 = df_req3.rename(columns= {'resignationReason':'retirement_reason'})
#
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

df_g2[df_g2.duplicated()]#filtra duplicados 
#no se encuentran datos duplicados
df_g3=df_g2.copy()
df_g3
#Como la fecha se encuentra en formato object vamos a convertirlo en formato fecha
df_g3["InfoDate"]=pd.to_datetime(df_g3['InfoDate'], format="%d/%m/%Y")
df_g3.info()


#ANALIZAR VARIABLES Y CATEGORIAS 
df_g3['EmployeeCount'].value_counts() #SOLO TIENE UNA CATEGORIA POR LO CUAL SE TOMA LA DECISIÓN DE ELIMINAR ESTA VARIABLE
df_g3['Over18'].value_counts() #solo tiene una categoria por lo cual se toma la decision de eliminar esta variable
df_g3['StandardHours'].value_counts() # todos los valores son iguales, tambien se toma la decision de eliminar esta variable
df_g3['StockOptionLevel'].value_counts() #Esta variable se va a convertir a variable categorica con 4 categorias
df_g3['JobLevel'].value_counts()#esta variable se va a convertir a variable categorica con 5 categorias
df_g3['Education'].value_counts()#Se va a convertir a una variable categorica de 5 categorias

df_g4= df_g3.drop(['EmployeeCount','Over18', 'StandardHours'], axis=1)#eliminar variables que no son representativas para el analisis
df_g4=df_g4.astype({'EmployeeID':object,'StockOptionLevel': object,"JobLevel": object, "Education": object})
df_g4= df_g4.rename(columns= {'InfoDate':'fecha'})# se quiere que en todas las bases la variable fecha tenga el mismo nombre

df_g4.info()




"""#PREPROCESAMIENTO BASE manager survey """ 
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
df_man3.info() 
df_man3['JobInvolvement'].value_counts()
df_man3['PerformanceRating'].value_counts()


#df_1.merge(df_2, on="id", how="left").merge(df_3, on="id", how="left") UNIR VARIAS TABLAS

#Traer solo la información necesaria
#RETIRENMENT: En esta base de datos solo se necesita la información de aquellos trabjadores que salieron en 2016 por lo cual solo traeremos los datos referentes a este año 
ret_16 = df_ret4[df_ret4['fecha'].dt.year == 2016]#Base general data con los datos del 2016

# UNIR BASES DE DATOS ANTERIOMENTE DEPURADAS #
"""se van a unir las bases partiendo del employeeID y la fecha para evitar que se dupliquen los datos, en todas las bases se usaran los datos tanto del 2015 y del 2016
menos en la base de retirenment"""

#df_merged = pd.merge(df_g4, df_empl5, how= 'left', on=['EmployeeID', 'fecha'])
#df_1.merge(df_2, on="id", how="left").merge(df_3, on="id", how="left") UNIR VARIAS TABLAS


base= df_g4.merge(df_empl5, how= 'left' , on=['EmployeeID', 'fecha']).merge(df_man3, how='left', on=['EmployeeID', 'fecha'])#base con la union de todas las tablas sin la tabla de retirement
base.info()


base15 = base[base['fecha'].dt.year == 2015]#separar la tabla base solo con los datos del 2015
base16 = base[base['fecha'].dt.year == 2016]#separar la tabla base solo con los datos del 2016, esta base se usara para la predicción para el año 2017
basefinal= pd.merge(base15, ret_16, how= 'left', on= 'EmployeeID')#Union de la tabla con los datos del 2015 con la base retirement que contiene la variable respuesta

basefinal = basefinal.rename(columns= {'fecha_x':'fecha_info', 'fecha_y':'fecha_retiro' })#Renombrar las columnas de las fechas para mejor interpretabilidad

basefinal.to_csv('data_hr_proyect/basefinal.csv', index= False)

base16.to_csv('data_hr_proyect/baseprediccion.csv', index= False)