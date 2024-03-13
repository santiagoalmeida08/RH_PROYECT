
"""En este apartado se analizaran y depuararan los datos nulos de las bases, sus duplicados y se procedera a realizar los respectivos cambios para
    unirlas correctamente, ademas se exportaran las bases depuradas para su posterior uso en el analisis de datos y la prediccion de retiros para el año 2017"""

#1. Preprocesamiento de la base de datos de encuestas de empleados
#2. Preprocesamiento de la base de datos de retiros
#3. Preprocesamiento de la base de datos general de empleados
#4. Preprocesamiento de la base de datos de encuestas de jefes
#5. Union bases
#6. Exportación de bases de datos

# Importación de paquetes 

import pandas as pd
import numpy as np
import datetime
import funciones as fn

# Dataframes necesarios 

data_employee = 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/employee_survey_data.csv' #Contiene encuestas de datisfaccion de los empleados
retirements = 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/retirement_info.csv' # Contiene informacion de empleados que se retiraron por algun motivo de la empresa
data_general= 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/general_data.csv' # Contiene informacion general de los empleados
data_manager= 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/manager_survey.csv' #Contiene informacion de encuestas realizadas a los jefes de los empleados

# 1. Preprocesamiento de la base de datos de encuestas de empleados

df_empl = pd.read_csv(data_employee,sep=';')
df_empl.info()  

df_empl2 = df_empl.drop('Unnamed: 0',axis=1) #Se elimina la columna Unnamed ya que no aporta informacion relevante

df_empl2.isnull().sum() # Existen nulos en 3 variables, se procedera a tratarlos

# Tratamiento nulos : Se realiza con la funcion 'nulos' la cual elimina los datos faltantes en caso de que estos representen menos del 10% de la base de datos

df_empl2 = fn.nulos(df_empl2)
df_empl2.isnull().sum()

df_empl2.dtypes 
# Se cambiara el formato de fecha a datetime y employeeid a object ya que de esta forma podemos manipular la informacion de mejor manera

df_empl4 = df_empl2.copy()
df_empl4['DateSurvey'] = pd.to_datetime(df_empl4['DateSurvey'])#transformacion de la variable fecha a datetime
df_empl4['EmployeeID'] = df_empl4['EmployeeID'].astype('object')#transformacion de la variable employeeid a object
df_empl4.info() # verificar cambios

# Renombrar columna fecha para que sea igual en todas las bases al unirlas
df_empl5 = df_empl4.copy()
df_empl5= df_empl5.rename(columns= {'DateSurvey':'fecha'})# Cambiar nombre de la columna DataSurvey a fecha

df_empl5.info() # Base preprocesada de encuestas de empleados

# 2. Preprocesamiento de la base de datos de retiros

df_req = pd.read_csv(retirements, sep= ';')
df_req.info() # Visualizacion previa de la base de datos

df_req1 = df_req.drop(['Unnamed: 0.1','Unnamed: 0'], axis= 1) # Se eliminan las columnas que no aportan informacion relevante ya que hacen referencia al index


df_req1.isnull().sum() #Existen nulos en razon de resignacion. ¿Que hacer con estos nulos?

# Exploramos como estan compuestas las variables de la base de datos
df_req1['Attrition'].value_counts() # Se refiere a los empleados que se retiraron de la empresa
df_req1['retirementType'].value_counts() # las personas se van de la empresa por resignacion o por despido
df_req1['resignationReason'].value_counts() # Causa de que el empleado se resigne y sea propenso a retirarse de la empresa

df_req1[['resignationReason','retirementType']].sample(20) # Sacamos una muestra para mirar el comportamiento de las variables

#Segun el analisis de la muestra se observa que los atipicos en la variable resignationReason son causados por los empleados que fueron despedidos
#No nos interesa analizar los despidos por lo cual se procedera a eliminar estos datos
df_req1 = df_req1.dropna()


# Procedemos a cambiar el formato de las variables fecha y employeeid, tambien a renombrar la variable retirementDate a fecha para que sea igual en todas las bases
df_ret4 = df_req1.copy() 
df_ret4['retirementDate'] = pd.to_datetime(df_ret4['retirementDate'], format='%d/%m/%Y')
df_ret4['EmployeeID'] = df_ret4['EmployeeID'].astype('object')
df_ret4 = df_ret4.rename(columns= {'retirementDate':'fecha'})
df_ret4.info()

df_ret4  # Base preprocesada de retiros 

# 3. Preprocesamiento de la base de datos general de empleados

df_g1= pd.read_csv(data_general, sep=';')#base de datos
df_g1.info() # Visualizacion previa de la base de datos

df_g2= df_g1.drop('Unnamed: 0', axis=1)#La columna unnamed: 0 cuenta las filas iniciando desde el cero, esta columna no aporta ninguna información relevante por lo cual se va eliminar

df_g2.isnull().sum() # Deteccion de datos nulos en la base; al ser pocos se procedera a eliminarlos

df_g2 = fn.nulos(df_g2) # eliminamos nulos 

df_g2[df_g2.duplicated()]#filtra duplicados; 

# Cambio en los formatos fecha y employeeid, tambien se renombra la variable InfoDate a fecha para que sea igual en todas las bases
df_g4=df_g2.copy()

df_g4['InfoDate']=pd.to_datetime(df_g4['InfoDate'], format='%d/%m/%Y') #transformacion de la variable fecha a datetime
df_g4= df_g4.rename(columns= {'InfoDate':'fecha'})# se quiere que en todas las bases la variable fecha tenga el mismo nombre
df_g4['EmployeeID'] = df_g4['EmployeeID'].astype('object')

df_g4.info()# base final

# 4. Preprocesamiento de la base de datos de encuestas de jefes

df_man1= pd.read_csv(data_manager, sep=';')#base de datos
df_man1.info() # Visualizacion previa de la base de datos
df_man2= df_man1.drop('Unnamed: 0', axis=1) #Se elimina la columna referente al index

#Nulos y duplicados 
df_man2.isnull().sum()#analisis de datos nulos; no se encuentran datos nulos
df_man2[df_man2.duplicated()]#filtra duplicados ; no se encuentran datos duplicados

# Cambio en los formatos fecha y employeeid, tambien se renombra la variable SurveyDate a fecha para que sea igual en todas las bases
df_man3=df_man2.copy()

df_man3["SurveyDate"]=pd.to_datetime(df_man3['SurveyDate'], format="%d/%m/%Y")
df_man3['EmployeeID'] = df_man3['EmployeeID'].astype('object')
df_man3 = df_man3.rename(columns= {'SurveyDate':'fecha'})# se quiere que en todas las bases la variable fecha tenga el mismo nombre

df_man3.info() # Base preprocesada de encuestas de jefes


# Consideraciones para la union de bases de datos

#RETIRENMENT: En esta base de datos solo se necesita la información de aquellos trabjadores que salieron en 2016 por lo cual solo traeremos los datos referentes a este año 
ret_16 = df_ret4[df_ret4['fecha'].dt.year == 2016]#Base general data con los datos del 2016

#5. Union bases 
# df_g4, df_empl5, df_man3, ret_16
"""Se van a unir las bases partiendo del employeeID y la fecha para evitar que se dupliquen los datos, en todas las bases se usaran los datos tanto del 2015 y del 2016
menos en la base de retirenment"""

base= df_g4.merge(df_empl5, how= 'left' , on=['EmployeeID', 'fecha']).merge(df_man3, how='left', on=['EmployeeID', 'fecha'])#base con la union de todas las tablas sin la tabla de retirement

base15 = base[base['fecha'].dt.year == 2015]#separar la tabla base solo con los datos del 2015
base16 = base[base['fecha'].dt.year == 2016]#separar la tabla base solo con los datos del 2016, esta base se usara para la predicción para el año 2017
basefinal= pd.merge(base15, ret_16, how= 'left', on= 'EmployeeID')#Union de la tabla con los datos del 2015 con la base retirement que contiene la variable respuesta

basefinal = basefinal.rename(columns= {'fecha_x':'fecha_info', 'fecha_y':'fecha_retiro' })#Renombrar las columnas de las fechas para mejor interpretabilidad

#6. Exportación de bases de datos

basefinal.to_csv('data_hr_proyect/basefinal.csv', index= False)
base16.to_csv('data_hr_proyect/baseprediccion.csv', index= False)