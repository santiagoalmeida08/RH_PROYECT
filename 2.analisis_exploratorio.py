############### ANALISIS EXPLORATORIO DE DATOS ####################

#1.  Estucturacion y analsis de la variable objetivo (Attrition)
#2.  Exploracion de variables que tengan nulos
#3.  Exploración Variables Numercias
#4.  Exploración Variables Categoricas
#5.  Exploración base para modelos

# Paquetes requeridos

#paquetes para manipulacion de datos
import pandas as pd
import numpy as np 

#paquetes para visualizacion de datos
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

#Archivo de funciones
import funciones as fn


# Importar base de datos
baseexploracion = 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/basefinal.csv'
base_expl = pd.read_csv(baseexploracion, sep = ',')
df_bfinal = base_expl.copy()
df_bfinal.info() # Vista previa tipo de datos
df_bfinal.columns # Vista previa columnas

df_bfinal = df_bfinal.drop(['fecha_retiro','fecha_info'], axis=1) # se elimina la variable fecha_retiro e info ya que no aporta informacion relevante para el analisis

# 1. Estucturacion y analsis de la variable objetivo (Attrition); es la variable objetivo ya que se busca predecir si un empleado abandonará la empresa o no

'''Se estructurá la variable objetivo Attrition, la cual se define como la salida de un empleado de la empresa:
    si el empleado ha abandonado por razones de salario,estres u otros. se le asigna 'Yes'
    si el empleado continua trabajando(valor nulo) en la empresa se le asigna 'No' '''
    
df_bfinal2 = df_bfinal.copy()
df_bfinal2.columns # verificamos las columnas

#reemplazamos valores segun la categoria resignationreason, reemplazamos los valores nulos por 'no' ya que los nulos representan que el empleado no ha renunciado    
df_bfinal2['Attrition'] = df_bfinal2['resignationReason'].replace({'Salary':'yes', 'Others':'yes', 'Stress':'yes'}).fillna('no') 
df_bfinal2['Attrition'] = df_bfinal2['Attrition'].replace({'yes':1, 'no':0}) # reemplazamos los valores yes y no por 1 y 0 respectivamente para ahorrar tiempo en el modelo
df_bfinal2['Attrition'].value_counts() # verificamos que se haya realizado correctamente el reemplazo
df_bfinal2.isnull().sum()

# Analisis de la variable objetivo #
#Entre los años 2015 y 2016 aproximadamente el 84% de los empleados no han abandonado la empresa, mientras que el 16% si lo ha hecho
#Observamos que la mayoria de los trabajadores no han abandonado la empresa, lo cual nos indica que la base esta desbalanceada
#por lo cual se debera tener en cuenta para el modelo

sns.countplot(x='Attrition', data=df_bfinal2)
plt.title('Distribución de la variable objetivo Attrition')
plt.xlabel('Attrition')
plt.ylabel('Frecuencia')
plt.show()


# 2.  Exploracion de variables que tengan nulos 

df_fin = df_bfinal2.copy()
df_no_null = fn.nulos(df_fin) # se eliminan valores nulos que representen menos del 10% de la base
df_no_null.isnull().sum()

# Analisis de variables con mas del 10% de nulos #

df_no_null['retirementType'].value_counts() # al tener una sola categoria la variable sera eliminada ya que es poco representativa

#los nulos corresponden a los empleados que no han renunciado por lo tanto se eliminara la variable ya que esta categoria quita relevancia a las demas porque es la que mas se repite
df_no_null['resignationReason'].value_counts() 

df_no_null2 = df_no_null.copy()
df_no_null2 = df_no_null2.drop(['retirementType','resignationReason'], axis=1) # se elimina la variable fecha_retiro ya que solo fue util para union de bases
df_no_null2.isnull().sum()

# 3.  Exploración Variables Numercias #

df_expl_num = df_no_null2.copy()
df_expl_num = df_expl_num.select_dtypes(include= np.number) # seleccionamos variables numericas 
df_expl_num.info()

# Histogramas 

# Analizaremos los histogramas con el objetivo de identificar comportamientos no deseados en variables #
df_expl_num.hist(figsize=(15, 15), bins=20)

# Se observa un comportamiento normal en la distribucion de la variable edad, sin embargo variables como
# el salario, los años de trabajo total, años en la compañia y años con el gerente actual presentan sesgos en su distribucion

# Se observa variables que solo tienen un valor, por lo cual no aportan informacion relevante para el modelo, tambien se puede notar
# que la variable performance rating tiene solo 2 valores y uno de ellos es mucho mas representativo que el otro, por lo cual se eliminará
"""
# Ademas las variables como education,job level, y las variables referentes a encuestas de satisfaccion seran tratadas como categoricas
# para tener un analisis mas detallado de estas variables"""

df_expl_num = df_expl_num.drop(['EmployeeCount','StandardHours','PerformanceRating','Education','JobSatisfaction','EnvironmentSatisfaction','WorkLifeBalance','JobLevel','EmployeeID'], axis=1) #Adicional a las variables df_no_null2 se eliminan las variables referentes a encuestas ya que seran tratadas como categoricas para un mejor analisis
df_no_null2 = df_no_null2.drop(['EmployeeCount','StandardHours','PerformanceRating'], axis=1)# eliminamos las variables que no aportan informacion relevante

# recategorizamos la variable education en base df_no_null2
df_no_null2['Education'] = df_no_null2['Education'].replace({1:'Escuela secundaria', 2:'Licenciatura', 3:'Maestria', 4:'Doctorado', 5:'Posdoctorado'})
df_no_null2['Education'].value_counts()

#Analisis descriptivo de variables numericas #

df_expl_num.describe()
"""En esta tabla se pueden observar varios datos importantes de cada variable que nos llevan a tener mejor vision de la empresa. como lo son :
    - El 75% de los empleados tienen menos de 43 años lo cual nos indica una poblacion joven en la empresa
    - El 50% de los empleados de la compañia han trabajaodo menos de 5 años en la empresa, esto nos podria indicar una cantidad siginificativa de empleados nuevos
    - El salario anual promedio de los empleados es de 65000 dolares; sin embargo el 50% de los empleados tienen un salario menor a 50000 dolares lo cual nos indica que la mayoria de los empleados tienen salarios bajos
    - La empresa no frecuenta hacer ascensos ya que el 75% de los empleados no han tenido un ascenso en los ultimos 5 años, esto podria incentivar desercion de empleados"""

df_expl_num.columns 

# Analisis de correlacion entre variables numericas #
correlation = df_expl_num.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

""""Con la correlacion podemos notar que la variable objetivo tiene relacion lineal principalmente con edad, años de trabajo total,años en la compañia y año bajo el mismo gerente 
    sin embargo las variables referentes a años de trabajo estan fuertemente correlacionadas; lo cual nos puede generar multicolinealidad en el modelo y afectar su rendimiento. Para esto
    eliminaremos las variables que menos correlacion tengan con la variable objetivo"""

# Eliminacion de variables correlacionadas #
# A pesar de que TotalWorkingYears es la variable con mayor relacion con la variable objetivo, la eliminamos ya que adicionalmente prsenta correlacion
# con la variable Age. Las demas variables se eliminan por tener una correlacion menor con la variable objetivo que con la que se conserva que es yearswithcurrmanager
df_expl_num2 = df_expl_num.copy()
df_expl_num2 = df_expl_num.drop(['YearsSinceLastPromotion','TotalWorkingYears','YearsAtCompany'], axis=1) 

correlation = df_expl_num2.corr() # verificamos la correlacion
plt.figure(figsize=(15, 15))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


#Eliminar variables correlacionadas de la base principal
df_no_null2 = df_no_null2.drop(['YearsSinceLastPromotion','TotalWorkingYears','YearsAtCompany'], axis=1)


# Analisis Bivariado #

df_biv_num = df_expl_num2.copy()
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
variables  =  ['Age','TrainingTimesLastYear','NumCompaniesWorked', 'YearsWithCurrManager']
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
#los atipicos son muchos y al ser una variable continua puede que se corten mas datos de los que se deberian
#la imputacion no nos parece una buena opcion ya que no se puede imputar con la media o la mediana ya que se perderia la representatividad de la variable

# Tratamiento de atípicos #

# Se trataran los atipicos de years with current manager ya que es la variable que mas relacion tiene con la variable objetivo

df_biv_num['YearsWithCurrManager'].describe()# se puede obervar que el 75% de los empleados llevan mas de 10 años en la empresa
df_biv_num['YearsWithCurrManager'].value_counts()
df_biv_num[df_biv_num['YearsWithCurrManager'] > 15].count() # se tienen 21 empleados los cuales llevan mas de 15 años con el mismo gerente

# Tratamiento atipicos monthly income


#realizamos un tanteo de el numero de atipicos que se encuentran en la variable YWCM y, se concluye que 
# solo se realizara una imputacion de valores extremos en la variable

# Imputacion de valores extremos en la variable years at company

df_no_null2 = df_no_null2[df_no_null2['YearsWithCurrManager'] < 15]


#Exploración variables categoricas

df_expl_cat = df_no_null2.copy()
df_no_null2.columns

df_expl_cat = df_expl_cat.select_dtypes(include='object')
df_expl_cat['Attrition'] = df_no_null2['Attrition'].astype('object') # se transforma la variable objetivo a categorica para un mejor analisis
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

# Exportar base de datos para modelos #
df_no_null2.info()
df5 = df_no_null2.copy()
df5.info()

base_seleccion2 = df5.copy()

base_seleccion2.to_csv('data_hr_proyect/base_seleccion2.csv', index= False)






