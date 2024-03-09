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

df_empl2 = fn.nulos(df_empl2, ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance'])
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

"""Podemos observar que todas las variables tienen la misma escala de 1 a 4,
por lo cual podemos definir una recategorizacion asi : """

dict = { 1.0:'Muy insatisfecho', 2.0:'Insatisfecho', 3.0:'Satisfecho', 4.0:'Muy satisfecho'}
variables = ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']
df_empl5 = df_empl4.copy()

for i in variables:
    df_empl5[i] = df_empl5[i].replace(dict) #transformacion de las variables a categoricas

df_empl5= df_empl5.rename(columns= {'DateSurvey':'fecha'})# se quiere que en todas las bases la variable fecha tenga el mismo nombre

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

df_g2 = fn.nulos(df_g2, ['NumCompaniesWorked', 'TotalWorkingYears']) # eliminamos nulos 

df_g2[df_g2.duplicated()]#filtra duplicados 
#no se encuentran datos duplicados
df_g3=df_g2.copy()

#ANALIZAR VARIABLES Y CATEGORIAS 
df_g3['EmployeeCount'].value_counts() #SOLO TIENE UNA CATEGORIA POR LO CUAL SE TOMA LA DECISIÓN DE ELIMINAR ESTA VARIABLE
df_g3['Over18'].value_counts() #solo tiene una categoria por lo cual se toma la decision de eliminar esta variable
df_g3['StandardHours'].value_counts() # todos los valores son iguales, tambien se toma la decision de eliminar esta variable

df_g3['JobLevel'].value_counts()#esta variable se va a convertir a variable categorica con 5 categorias
df_g3['Education'].value_counts()#Se va a convertir a una variable categorica de 5 categorias
df_g3['NumCompaniesWorked'].value_counts()#Se va a convertir a una variable categorica agrupando los valores
df_g3['TrainingTimesLastYear'].value_counts()#Se va a convertir a una variable categorica agrupando los valores

#Transformacion de variables

va3 = ['StockOptionLevel', 'JobLevel', 'Education', 'NumCompaniesWorked', 'TotalWorkingYears','EmployeeID']
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

df_man3['JobInvolvement'] = df_man3['JobInvolvement'].replace({1:'Bajo', 2:'Bajo', 3:'Medio', 4:'Alto'})
df_man3['PerformanceRating'] = df_man3['PerformanceRating'].replace({3:'Bajo', 4:'Alto'})

df_man3.info() 

#Traer solo la información necesaria
#RETIRENMENT: En esta base de datos solo se necesita la información de aquellos trabjadores que salieron en 2016 por lo cual solo traeremos los datos referentes a este año 
ret_16 = df_ret4[df_ret4['fecha'].dt.year == 2016]#Base general data con los datos del 2016

# UNIR BASES DE DATOS ANTERIOMENTE DEPURADAS #
"""se van a unir las bases partiendo del employeeID y la fecha para evitar que se dupliquen los datos, en todas las bases se usaran los datos tanto del 2015 y del 2016
menos en la base de retirenment"""


base= df_g4.merge(df_empl5, how= 'left' , on=['EmployeeID', 'fecha']).merge(df_man3, how='left', on=['EmployeeID', 'fecha'])#base con la union de todas las tablas sin la tabla de retirement
#base.info()


base15 = base[base['fecha'].dt.year == 2015]#separar la tabla base solo con los datos del 2015
base16 = base[base['fecha'].dt.year == 2016]#separar la tabla base solo con los datos del 2016, esta base se usara para la predicción para el año 2017
basefinal= pd.merge(base15, ret_16, how= 'left', on= 'EmployeeID')#Union de la tabla con los datos del 2015 con la base retirement que contiene la variable respuesta

basefinal = basefinal.rename(columns= {'fecha_x':'fecha_info', 'fecha_y':'fecha_retiro' })#Renombrar las columnas de las fechas para mejor interpretabilidad

basefinal.to_csv('data_hr_proyect/basefinal.csv', index= False)

base16.to_csv('data_hr_proyect/baseprediccion.csv', index= False)
