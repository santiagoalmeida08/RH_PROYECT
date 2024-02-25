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

df_empl5 = df_empl4.copy()

"""crearemos una nueva variable en la cual se reflejara el promedio de la encuesta realizada
a cada uno de los trabajadores"""

df_empl5.info()

df_empl5['mean_survery'] = ((df_empl5['EnvironmentSatisfaction']+df_empl5['JobSatisfaction']+df_empl5['WorkLifeBalance']) /3).round(1)

df_empl5

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
df_req1.info()


#Evaluación variables categoricas#
df_req1['Attrition'].value_counts() 
df_req1['retirementType'].value_counts()
df_req1['resignationReason'].value_counts()

""" -La variable Attrition no presenta variabilidad por lo cual se la eliminará
    -Convertiremosla variable retirementType a binaria """

df_req2 = df_req1.copy()

df_req2 = df_req2.drop(['Attrition'], axis=1)

#df_req2['retirementType'] = df_req2['retirementType'].replace({'Resignation': 0, 'Fired':1}) #Causa : 0.Resignación;  1.Despido
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

df_ret4 = df_req3.copy()

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