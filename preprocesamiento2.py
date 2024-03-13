
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
df_empl2.isnull().sum() # se puede observar que las variables tienen un porcentaje bajo de datos nulos

df_empl2 = fn.nulos(df_empl2)
df_empl2.isnull().sum()

# TRANSFORMACION DE VARIABLES #

"""Se cambiara el formato de fecha a datetime, y el resto de las variables a categoricas, ya que 
al tratarse de encuestas, se puede reorganizar la categorizacion """

df_empl4 = df_empl2.copy()

df_empl4['DateSurvey'] = pd.to_datetime(df_empl4['DateSurvey'])#transformacion de la variable fecha a datetime
df_empl4['EmployeeID'] = df_empl4['EmployeeID'].astype('object')#transformacion de la variable employeeid a object

df_empl4.info()


"""Podemos observar que todas las variables tienen la misma escala de 1 a 4,
por lo cual podemos definir una recategorizacion asi : """

#dict = { 1.0:'Muy insatisfecho', 2.0:'Insatisfecho', 3.0:'Satisfecho', 4.0:'Muy satisfecho'}
#variables = ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']
df_empl5 = df_empl4.copy()

#for i in variables:
 #   df_empl5[i] = df_empl5[i].replace(dict) #transformacion de las variables a categoricas

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

df_req1['Attrition'].value_counts() #variable objetivo
df_req1['retirementType'].value_counts() # las personas se van de la empresa por resignacion o por despido
df_req1['resignationReason'].value_counts() # razones de resignacion

# Analisis de datos nulos
df_req1.isnull().sum() # se puede observar que la variable resignationReason tiene 70 datos nulos
                        #los cuales coinciden con los 70 empleados que fueron despedidos   


# Para tratar los nulos vamos a crear una nueva variable que nos permita explicar los valores nulos de resignationReason
#los cuales seran reemplazados por la categoria fired

df_req3 = df_req1.copy()
"""
df_req3['resignationReason'] = df_req3['resignationReason'].fillna('Fired')
df_req3 = df_req3.drop('retirementType', axis=1)

df_req3 = df_req3.rename(columns= {'resignationReason':'retirement_reason'})
df_req3['retirement_reason'].value_counts()"""

# Al analizar los nulos podemos observar que se tienen 70 nulos en la variable resignationReason los cuales coinciden con los 70 empleados que fueron despedidos
# se eliminaran estos datos ya que no nos interesa analizar los despidos

df_req3 = df_req3.dropna()
df_req3.info()

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


df_g3['JobLevel'].value_counts()#esta variable se va a convertir a object
df_g3['Education'].value_counts()#Se va a convertir a object y categorizar


#Transformacion de variables

va3 = ['JobLevel','Education','EmployeeID']
df_g3 = fn.transformacion(df_g3, va3)



df_g3['NumCompaniesWorked'] = df_g3['NumCompaniesWorked'].replace({0:'Al menos 2 empresas', 1:'Al menos 2 empresas', 2:'Al menos 2 empresas', 
                                                                  3:'De 3 a 5 empresas', 4:'De 3 a 5 empresas', 5:'De 3 a 5 empresas', 
                                                                  6:'Mas de 5 empresas', 7:'Mas de 5 empresas', 8:'Mas de 5 empresas', 9:'Mas de 5 empresas'})


df_g3['TrainingTimesLastYear'] = df_g3['TrainingTimesLastYear'].replace({0:'Ningun entrenamiento', 
                                                                   1:'Al menos 3 semanas', 2:'Al menos 3 semanas', 3:'Al menos 3 semanas', 
                                                                   4:'De 4 a 5 semanas', 5:'De 4 a 5 semanas', 6:'De 4 a 5 semanas'})
                                                    
                                                                   

df_g3['Education'] = df_g3['Education'].replace({1:'Escuela secundaria', 2:'Licenciatura', 3:'Maestria', 4:'Doctorado', 5:'Posdoctorado'})

#df_g3['JobLevel'] = df_g3['JobLevel'].replace({1:'Nivel 1', 2:'Nivel 2', 3:'Nivel 3', 4:'Nivel 4', 5:'Nivel 4'}) # se entiende a joblevel como el nivel jerarquico del empleado


df_g4= df_g3.drop(['EmployeeCount','Over18', 'StandardHours'], axis=1)#eliminar variables que no son representativas para el analisis"""


df_g4 = df_g3.copy()

df_g4= df_g4.rename(columns= {'InfoDate':'fecha'})# se quiere que en todas las bases la variable fecha tenga el mismo nombre

#Como la fecha se encuentra en formato object vamos a convertirlo en formato fecha
df_g4['fecha']=pd.to_datetime(df_g4['fecha'])
df_g4['EmployeeID'] = df_g4['EmployeeID'].astype('object')
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
"""
#ANALISIS DE VARIABLES Y CATEGORIAS 
df_man3['JobInvolvement'].value_counts()
df_man3['PerformanceRating'].value_counts()

df_man3['JobInvolvement'] = df_man3['JobInvolvement'].astype('object')
#df_man3['PerformanceRating'] = df_man3['PerformanceRating'].replace({3:'Bajo', 4:'Alto'})"""

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

#basefinal.to_csv('data_hr_proyect/basefinal.csv', index= False)

#base16.to_csv('data_hr_proyect/baseprediccion.csv', index= False)

















############### ANALISIS EXPLORATORIO DE DATOS ####################

# Quitamos las variables que no van a servir 
base_expl = basefinal.drop(['fecha_info', 'fecha_retiro','EmployeeID'], axis=1)


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
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
import funciones as fn
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2

#Dataframes

df_bfinal = base_expl.copy()
df_bfinal.info()

# Estucturacion variable objetivo (Attrition); es la variable objetivo ya que se busca predecir si un empleado abandonará la empresa o no

'''Se estructurá la variable objetivo Attrition, la cual se define como la salida de un empleado de la empresa:
    si el empleado ha abandonado por razones de salario,estres u otros. se le asigna 'Yes'
    si el empleado continua trabajando(valor nulo) en la empresa se le asigna 'No' '''
    
df_bfinal2 = df_bfinal.copy()
df_bfinal2.columns
    
df_bfinal2['Attrition'] = df_bfinal2['resignationReason'].replace({'Salary':'yes', 'Others':'yes', 'Stress':'yes'}).fillna('no')  #reemplazamos valores segun la categoria retirement_reason
df_bfinal2['Attrition'].value_counts()
df_bfinal2.isnull().sum()

# Exploracion de variables que tengan nulos 

df_fin = df_bfinal2.copy()
df_no_null = fn.nulos(df_fin) # se eliminan valores nulos que representen menos del 10% de la base
df_no_null.isnull().sum()

# Analisis de variables con mas del 10% de nulos #

df_no_null['retirementType'].value_counts() # al tener una sola categoria la variable sera eliminada ya que es poco representativa
df_no_null['resignationReason'].value_counts() #los nulos corresponden a los empleados que no han renunciado por lo tanto se eliminara la variable



df_no_null2 = df_no_null.copy()
df_no_null2 = df_no_null2.drop(['retirementType','resignationReason'], axis=1) # se elimina la variable fecha_retiro ya que solo fue util para union de bases
df_no_null2.isnull().sum()



# Exploración Variables Numercias #

df_expl_num = df_no_null2.copy()
df_expl_num = df_expl_num.select_dtypes(include=np.number) # seleccionamos variables numericas 
df_expl_num.info()

# Histogramas 

# Analizaremos los histogramas con el objetivo de identificar comportamientos no deseados en variables #
df_expl_num.hist(figsize=(15, 15), bins=20)

# Se observa un comportamiento normal en la distribucion de la variable edad, sin embargo variables como
# el salario, los años de trabajo total, años en la compañia y años con el gerente actual presentan sesgos en su distribucion

# Se observa variables que solo tienen un valor, por lo cual no aportan informacion relevante para el modelo, tambien se puede notar
# que la variable performance rating tiene solo 2 valores y uno de ellos es mucho mas representativo que el otro, por lo cual se eliminará

# Ademas las variables como education,job level, y las variables referentes a encuestas de satisfaccion seran tratadas como categoricas
# para tener un analisis mas detallado de estas variables

df_expl_num = df_expl_num.drop(['EmployeeCount','StandardHours','PerformanceRating','Education','JobSatisfaction','EnvironmentSatisfaction','WorkLifeBalance','JobLevel'], axis=1)
df_no_null2 = df_no_null2.drop(['EmployeeCount','StandardHours','PerformanceRating'], axis=1)# eliminamos las variables que no aportan informacion relevante

# recategorizamos la variable education en base df_no_null2
df_no_null2['Education'] = df_no_null2['Education'].replace({1:'Escuela secundaria', 2:'Licenciatura', 3:'Maestria', 4:'Doctorado', 5:'Posdoctorado'})

#pasaps a categoricas las variables de encuestas de satisfacción y joblevel  en base df_no_null2
df_no_null2['JobLevel'] = df_no_null2['JobLevel'].astype('object')
df_no_null2['EnvironmentSatisfaction'] = df_no_null2['EnvironmentSatisfaction'].astype('object')
df_no_null2['JobSatisfaction'] = df_no_null2['JobSatisfaction'].astype('object')
df_no_null2['WorkLifeBalance'] = df_no_null2['WorkLifeBalance'].astype('object')


#Analisis descriptivo de variables numericas #

df_expl_num.describe()
"""En esta tabla se pueden observar varios datos importantes de cada variable que nos llevan a tener mejor vision de la empresa. como lo son :
    - El 75% de los empleados tienen menos de 43 años lo cual nos indica una poblacion joven en la empresa
    - El 50% de los empleados de la compañia han trabajaodo menos de 5 años en la empresa, esto nos podria indicar una cantidad siginificativa de empleados nuevos
    - El salario anual promedio de los empleados es de 65000 dolares; sin embargo el 50% de los empleados tienen un salario menor a 50000 dolares lo cual nos indica que la mayoria de los empleados tienen salarios bajos"""

df_expl_num.columns 

# Analisis de correlacion entre variables numericas #
correlation = df_expl_num.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()



#
# X_new ahora contiene solo las características que tienen una varianza mayor que el umbral
"""
Todas las variables que hacen referencia a los años de trabajo tienen una alta correlacion entre ellas;
esto podria ser un problema para el modelo de regresion logistica, ya que podria haber multicolinealidad; 
la solucion propuesta es realizar un modelo de seleccion de variables para conservar solo la variable mas representativa"""

#Eliminar variables correlacionadas referentes  a los 

df_no_null2 = df_no_null2.drop(['YearsSinceLastPromotion','YearsWithCurrManager'], axis=1)
df_expl_num = df_expl_num.drop(['YearsSinceLastPromotion','YearsWithCurrManager'], axis=1)

# Analisis Bivariado #

df_biv_num = df_expl_num.copy()
df_biv_num['Attrition'] = df_no_null2['Attrition']
df_biv_num.info()

# Boxplot de variables numericas vs variable objetivo #

for column in df_biv_num.columns:   
    if column != 'Attrition':
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_biv_num, x='Attrition', y=column)
        plt.title(f'Boxplot de {column} vs Attrition')
        plt.xlabel('Attrition')
        plt.ylabel(column)
        plt.show()
    else:
        pass # no se hace nada


""" En los boxplots se puede observar lo siguientes aspectos relevantes : 
    - Los empleados que abandonaron la empresa tienen menos edad que los que siguen trabajando
    - En la variable referente al salario anual se puede observar que hubo desercion de empleados con un salario bastante alto, lo cual
        nos llevaria a pensar que incluso estos empleados que probablemente lleven mucho tiempo en la empresa desiden abandonarla, siendo asi que la causa del abandono no sea el salario
    - Parte de los empleados que abandonaron el trabajo llevaban en promedio menos de 10 años en la empresa;
     sin embargo encontramos el mismo comportamiento que en salario; lo cual nos quiere decir que hay una razon mas alla de la antiguedad que lleve el trabajador en la empresa
    - Observando el boxplot de años con el mismo gente se nota un patron importante el cual hace referencia a que los empleados que llevan trabajando
      mucho tiempo bajo el mismo jefe no han abandonado la empresa, lo cual puede indicar que el jefe puede influir en la desicion de abandonar la empresa"""

df_biv_num.info()

# Atipicos #
variables  =  ['Age','TrainingTimesLastYear','NumCompaniesWorked', 'YearsAtCompany']
# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(12, 8))

# Crear los boxplots
ax.boxplot(df_biv_num[variables].values, labels=variables)

# Configurar el título y las etiquetas
ax.set_title('Boxplots Atipicos')
ax.set_ylabel('Valores')

# Rotar las etiquetas del eje x para una mejor visualización
plt.xticks(rotation=45)

# Mostrar la imagen
plt.show()


#No se trataran atipicos en monthly income ya que como se vio en los boxplots de analisis bivariado,
# los atipicos son muchos y al ser una variable continua puede que se corten mas datos de los que se deberian
#la imputacion no nos parece una buena opcion ya que no se puede imputar con la media o la mediana ya que se perderia la representatividad de la variable

# Tratamiento de atípicos #

# Se trataran los atipicos de years at company 

df_biv_num['YearsAtCompany'].describe()# se puede obervar que el 75% de los empleados llevan mas de 10 años en la empresa

df_biv_num[df_biv_num['YearsAtCompany'] > 32].count() 

# Tratamiento atipicos monthly income


#realizamos un tanteo de el numero de atipicos que se encuentran en la variable years at company, se concluye que 
# solo se realizara una imputacion de valores extremos en la variable years at company

# Imputacion de valores extremos en la variable years at company

df_no_null2 = df_no_null2[df_no_null2['YearsAtCompany'] < 32.0]
#df_no_null2 = df_no_null2.drop('MonthlyIncome', axis=1)
#Exploración variables categoricas

df_expl_cat = df_no_null2.copy()
df_no_null2.columns

df_expl_cat = df_expl_cat.select_dtypes(include='object')
df_expl_cat.columns
#Analisis de las variables 
    
for column in df_expl_cat.columns:
    if column != 'Attrition': # se excluye la variable objetivo ya que esta se analizara por separado
        base = df_expl_cat.groupby([column])[['Attrition']].count().reset_index().rename(columns ={'Attrition':'count'})
        fig = px.pie(base, names=column, values='count', title= column)
        xaxis_title = column
        yaxis_title = 'Cantidad'
        template = 'simple_white'
        fig.show()

# se eliminan variables con categorias poco representativas

df_expl_cat = df_expl_cat.drop('Over18',axis = 1)
df_no_null2 = df_no_null2.drop('Over18',axis = 1)

#Analisis Bivariado de variables categoricas #

for column in df_expl_cat.columns:
    # Crear el gráfico de conteo
    sns.countplot(x='Attrition', hue=column, data=df_expl_cat)

    # Añadir títulos y etiquetas (opcional)
    plt.title( f'Distribución de {column} vs de la variable respuesta Attrition')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')

    # Mostrar el gráfico
    plt.show()



# Vamos a cargar a la carpeta data una base con las variables seleccionadas para utilizarla en la seleccion de modelos #
df_no_null2.info()
df5 = df_no_null2.copy()
df5.info()
























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
from sklearn.impute import SimpleImputer ### para imputación
# Cargar el DataFrame

#data_seleccion= 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/base_seleccion.csv'

#df = pd.read_csv(data_seleccion,sep=',') ## BASE ORIGINAL ##

df = df5.copy()

df.dtypes  # NO se tienen : BussinessTravel,TotalWorkingYears,TrainingTimesLastYear, PerformanceRating

df.isnull().sum()

df = df5.drop(['BusinessTravel','TotalWorkingYears'], axis = 1)
df1 = df.copy()
df1.info()

df1['JobLevel'] = df1['JobLevel'].astype('object')  

df1['NumCompaniesWorked'] = df1['NumCompaniesWorked'].astype('int64')
#df1['TotalWorkingYears'] = df1['TotalWorkingYears'].astype('int64')
df1['PercentSalaryHike'] = df1['PercentSalaryHike'].astype('int64')
df1['WorkLifeBalance'] = df1['WorkLifeBalance'].astype('int64')
df1['JobSatisfaction'] = df1['JobSatisfaction'].astype('int64')
df1['EnvironmentSatisfaction'] = df1['EnvironmentSatisfaction'].astype('int64')

list_cat = [df1.columns[i] for i in range(len(df1.columns)) if df1[df1.columns[i]].dtype == 'object']
list_oe = ['JobLevel']
list_le = [df1.columns[i] for i in range(len(df1.columns)) if df1[df1.columns[i]].dtype == 'object' and len(df1[df1.columns[i]].unique()) == 2]
list_dd = ['Department','Education','EducationField','JobRole','MaritalStatus']


df1.dtypes

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


df3.dtypes

for i in df3.columns:
    if df3[i].dtypes == "float64":
        df3[i] = df3[i].astype('int64')
    else:
        pass

#Normalizacion

v_num = []
for col in df3.columns:
    if df3[col].dtypes == "int64":
        v_num.append(col)

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
for col in v_num:
    df3[[col]] = scaler.fit_transform(df3[[col]])

df3.head()

X_esc = df3.drop('Attrition', axis = 1)
y = df3['Attrition']

# Selecccion de modelos #

model_gb = GradientBoostingClassifier()
model_arb = DecisionTreeClassifier() 
model_log = LogisticRegression( max_iter=1000, random_state=42)
model_rand = RandomForestClassifier()

modelos  = list([model_gb,model_arb,model_log,model_rand])

# Seleccion de variables con base a los modelos seleccionados
def sel_variables(modelos,X,y,threshold):
    
    var_names_ac=np.array([])
    for modelo in modelos:
   
        modelo.fit(X,y)
        sel = SelectFromModel(modelo, prefit=True,threshold=threshold)
        var_names= modelo.feature_names_in_[sel.get_support()]
        var_names_ac=np.append(var_names_ac, var_names)
        var_names_ac=np.unique(var_names_ac)
    
    return var_names_ac


var_names= sel_variables(modelos,X_esc,y,threshold="2.7*mean") 
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

f1sco_df = medir_modelos(modelos,"f1",X_esc,y,15)  #se definen 10 iteraciones para tener mejor visión del desempeño en el boxplot
f1dco_var_sel = medir_modelos(modelos,"f1",df4,y,15)


f1s=pd.concat([f1sco_df,f1dco_var_sel],axis=1) 
f1s.columns=['rlog', 'dtree', 'rforest', 'gboosting',
       'rl_Sel', 'dt_sel', 'rf_sel', 'gb_Sel']


f1s.plot(kind='box') ### gráfico para modelos sel y todas las variables

f1s.mean() 
"""En los boxplots podemos observar que el mejor desempeño con las variables seleccionadas y con todas las variables lo tiene decision tree.
Ademas de esto una pequeña diferencia de desempeño (2%) entre el modelo con todas las variables y las variables seleccionadas; se va a trabajar con 
el modelo de variables seleccionadas, sacrificando ese 2% de desempeño pero mejorando la interpretabilidad del modelo y ahorrando recursos computacionales."""

#Matriz confusion para rl_Sel
model_log = LogisticRegression( max_iter=1000, random_state=42)
model_log.fit(df4,y)

y_pred = model_log.predict(df4)
cm = confusion_matrix(y, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=model_log.classes_)
cmd.plot()

#matriz confusion para dt_sel
model_arb = DecisionTreeClassifier(max_depth=4, random_state=42)
model_arb.fit(df4,y)

y_pred = model_arb.predict(df4)
cm = confusion_matrix(y, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=model_arb.classes_)
cmd.plot()

#matriz confusion para gb_Sel
model_gb = GradientBoostingClassifier()
model_gb.fit(df4,y)

y_pred = model_gb.predict(df4)
cm = confusion_matrix(y, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=model_gb.classes_)
cmd.plot()




# Ajuste de hiperparametros del modelo ganador #

from sklearn.model_selection import RandomizedSearchCV

#Definicion de parametros para regresion logistica

parameters = {
    'penalty': ['l1', 'l2'], # penalización l1 y l2 ridge y lasso
    'fit_intercept': [True, False], # si se ajusta la intersección
    'max_iter': [100, 500, 1000]
} # max_leaf_nodes es el numero maximo de nodos finales

# create an instance of the randomized search object
r1 = RandomizedSearchCV(LogisticRegression(), parameters, cv=5, n_iter=100, random_state=42, n_jobs=-1, scoring='accuracy') 

r1.fit(df4,y) #df4 es el dataframe con las variables seleccionadas

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

eval=cross_validate(rf_final,df4,y,cv=5,scoring='accuracy',return_train_score=True) 

train_rf=pd.DataFrame(eval['train_score'])
test_rf=pd.DataFrame(eval['test_score'])
train_test_rf=pd.concat([train_rf, test_rf],axis=1)
train_test_rf.columns=['train_score','test_score']
train_test_rf

train_test_rf["test_score"].mean()
