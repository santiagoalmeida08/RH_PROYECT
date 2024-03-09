
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
df_no_null2.info() 

# Exploración Variables Numercias #

df_expl_num = df_no_null2.copy()
df_expl_num = df_expl_num.select_dtypes(include=np.number).drop(['EmployeeID'], axis=1) # seleccionamos variables numericas y eliminamos EmployeeID ya que esta no aporta informacion relevante para la exploracion
df_expl_num.info()

# Histogramas #
#Analizaremos los histogramas con el objetivo de identificar comportamientos no deseados en variables #
for column in df_expl_num.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df_expl_num, x=column, kde=True)
    plt.title(f'Histograma de {column}')
    plt.xlabel(column)
    plt.ylabel('Frecuancia')
    plt.show()
    
"""Se pudo identificar variables que podrian ser convertidas a categoricas como lo son: 
    - StockOptionLevel
    - PercentSalaryHike
    - YearsSinceLastPromotion"""

vars = ['StockOptionLevel', 'PercentSalaryHike', 'YearsSinceLastPromotion']
df_no_null2 = fn.transformacion(df_no_null2, vars) # se transforman las variables a categoricas

df_no_null2['StockOptionLevel'] = df_no_null2['StockOptionLevel'].replace({0.0 : 'Ninguno', 1.0:'Bajo',2.0:'Medio', 3.0:'Alto'})   #  se convierten las variables a categoricas
df_no_null2['PercentSalaryHike'] = df_no_null2['PercentSalaryHike'].replace({11.0 : 'Bajo', 12.0:'Bajo',13.0:'Bajo',
                                                                             14.0:'Medio', 15.0:'Medio', 16.0:'Medio',
                                                                             17.0:'Medio', 18.0:'Medio', 19.0:'Medio',
                                                                             20.0:'Alto', 21.0:'Alto', 22.0:'Alto', 
                                                                             23.0:'Alto', 24.0:'Alto', 25.0:'Alto'})

df_no_null2['YearsSinceLastPromotion'].value_counts()

df_no_null2['YearsSinceLastPromotion'] = df_no_null2['YearsSinceLastPromotion'].replace({0:'Nunca',
                                                              1: 'Entre 1 y 4 años',2: 'Entre 1 y 4 años', 3: 'Entre 1 y 4 años', 4: 'Entre 1 y 4 años',
                                                              5 : 'Entre 5 y 9 años', 6: 'Entre 5 y 9 años', 7: 'Entre 5 y 9 años', 8: 'Entre 5 y 9 años', 9: 'Entre 5 y 9 años',
                                                              10: 'Entre 10 y 15 años', 11: 'Entre 10 y 15 años', 12: 'Entre 10 y 15 años', 13: 'Entre 10 y 15 años', 14: 'Entre 10 y 15 años', 15: 'Entre 10 y 15 años'})


#Analisis descriptivo de variables numericas #
df_expl_num = df_expl_num.drop(['StockOptionLevel', 'PercentSalaryHike', 'YearsSinceLastPromotion'], axis=1) # eliminamos las variables que se transformaron a categoricas
df_expl_num.describe()
"""En esta tabla se pueden observar varios datos importantes de cada variable como lo son :
    - El 75% de los empleados tienen menos de 43 años lo cual nos indica una poblacion joven en la empresa
    - El 50% de los empleados de la compañia han trabajaodo menos de 5 años en la empresa, esto nos podria indicar una cantidad siginificativa de empleados nuevos
    - El salario anual promedio de los empleados es de 65000 dolares; sin embargo el 50% de los empleados tienen un salario menor a 50000 dolares lo cual nos indica que la mayoria de los empleados tienen salarios bajos"""

df_expl_num.columns

bxp1 = df_expl_num.iloc[:,:2] # partimos la base en 3 partes para poder visualizar los boxplot de manera mas clara

bxp2 = df_expl_num.iloc[:,3:6]


plt.figure(figsize=(10, 6))
sns.boxplot(data=bxp1)
plt.title('Boxplot Edad y Distancia')
plt.xlabel('Variables')
plt.show()
""" 1. Boxplot variable edad: se puede observar que la mayoria de los empleados tienen entre 30 y 40 años, admeas no se observan valores atipicos
 
    2. Boxplot variable Distancia : se puede observar que el 75% de los empleados viven a menos de 10 km de la empresa, sin embargo el 15% restante vive a mas de 20 km de la empresa, lo cual podria ser un factor de desercion laboral"""

# Analizaremos las variable monthlyincome aparte debido a que su escala es diferente a las demas variables
plt.figure(figsize=(10, 6))
sns.boxplot(data= df_expl_num['MonthlyIncome']) 
plt.title('Boxplot de Salario Anual')
plt.xlabel('Variables')
plt.show()

""" Boxplot de salario anual: se puede observar que la mayoria de los empleados tienen un salario bajo, ademas se observan valores atipicos
    correspondientes a empleados con salarios muy altos que probablemente sean los empleados que llevan mas tiempo en la empresa, esto podria ser un factor de desercion laboral ya que los empleados con salarios bajos
    podrian sentirse desmotivados"""


plt.figure(figsize=(10, 6))
sns.boxplot(data=bxp2)
plt.title('Boxplot de Variables relacionadas con el tiempo en la empresa')
plt.xlabel('Variables')
plt.show()
""" Analisis de las variables relacionadas con el tiempo en la empresa: el 75% de los empleados a trabajado menos de 15 años; esto se ve reflejadp con lo analizado en la edad
    con lo cual puede exisir una alta correlacion entre estas dos variables .Tanto en la variable años en la compañia como en la variable referente a 
    la promocion hay datos atipicos correspondientes a empleados que llevan mas de 20 años en la empresa y aparentemente no han sido promovidos en un gran periodo de tiempo. """
    
 
# Analisis de correlacion entre variables numericas #

correlation = df_expl_num.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

"""Todas las variables que hacen referencia a los años de trabajo tienen una alta correlacion entre ellas;
esto podria ser un problema para el modelo de regresion logistica, ya que podria haber multicolinealidad; 
una solucion seria seleccionar las variables mas importantes para el modelo o aplicar un modelo de reduccion de dimensionalidad como PCA para reducir
la cantidad de variables y evitar la multicolinealidad"""

# Analisis Bivariado #

df_biv_num = df_expl_num.copy()
df_biv_num['Attrition'] = df_no_null2['Attrition']
df_biv_num.info()

# Boxplot de variables numericas vs variable objetivo #

for column in df_biv_num.columns:   
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_biv_num, x='Attrition', y=column)
    plt.title(f'Boxplot de {column} vs Attrition')
    plt.xlabel('Attrition')
    plt.ylabel(column)
    plt.show()
    

""" En los boxplots se puede observar lo siguientes aspectos relevantes : 
    - Los empleados que abandonaron la empresa tienen menos edad que los que siguen trabajando
    - Parte de los empleados que abandonaron el trabajo llevaban en promedio menos de 10 años en la empresa;
      lo mismo sucede con la variable relacionada a total de años trabajados y años en la compañia; los empleados que 
      abandonaron la empresa llevaban en todos los aspectos menos tiempo en la empresa que los que siguen trabajando
    -Los empleados con salarios bajos tienden a abandonar la empresa"""

df_pred.columns

# Exploracion de variables categoricas #
"""for i in df_no_null2[['StockOptionLevel', 'PercentSalaryHike', 'YearsSinceLastPromotion']]:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_no_null2, x=i, hue='Attrition')
    plt.title(f'{i} vs Attrition')
    plt.xlabel(i)
    plt.ylabel('Count')
    plt.show()"""
    

"""En los graficos de barras se puede observar lo siguiente:
    - Los empleados con stock options bajo o ninguno  tienden a abandonar la empresa
    - Los empleados con aumentos salariales Medios tienden a abandonar la empresa (no es lo que se espera)
    - Los empleados que no han sido promovidos en un largo periodo de tiempo tienden a abandonar la empresa"""
    

#Exploración variables categoricas
df_expl_cat = df_no_null2.copy()
df_no_null2.columns

df_expl_cat = df_expl_cat.select_dtypes(include='object').drop(['fecha_info'], axis=1)


#Analisis de las variables 

for column in df_expl_cat.columns:
    base = df_expl_cat.groupby([column])[['Attrition']].count().reset_index().rename(columns ={'Attrition':'count'})
    fig = px.pie(base, names=column, values='count', title= column)
    xaxis_title = column
    yaxis_title = 'Cantidad'
    template = 'simple_white'
    fig.show()
## REEESTRUCTURAR JOBROL

"""Para la variable businesstravel se observa que la mayoria de datos se encuentran en la categoria viajar rara vez pero aun asi no es un peso lo suficientemente grande para descartar la variable"""
"""Para la variable department es importante analizar si en realidad es una variable representativa ya que el departamento de recursos humanos solo representa el 4,33% del total de los datos
por lo cual esta variable puede descartarse y tomarse como no representativa"""
"""En education al tener un mayor numero de categorias es mas comun que existan categorias con poca representación por lo cual esto no sera motivo para descartar la variable"""
"""Para la variable education field  hay muy poca representación de recursos humanos pero esto puede deberse a la poca cantidad de personas que hay en este departamento tal como se observo
en la variable department, debido a esto y a que se tienen mas categorias que podrian ser de interes la variable no se va a descartar"""
"""La variable gender esta distribuida mas equitativamente hay una diferencia del solo 10% entre las 2 categorias"""
"""En cuanto a las categorias de la variable job level todas tienen una representatividad importante por lo cual la variable se sigue teniendo en cuenta"""
"""La variable job role cuenta con 9 categorias de cuales aproximadamente 4 son muy poco representativas por lo cual puede ser una variable que no aporte mucho al modelo y ademas 
dificulte su interpretación a pesar de esto aun no se va tomar la decisión de eliminar la variable"""
"""En la variable marital status se observan 3 categorias, todas significativas"""
"""Para la variable de numcompaniesworked no se puede tomar aun una decision de si es una variable representativa o no ya que a pesar de que tiene una categoria con un bajo porcentaje aun puede seguir
aportando al modelo"""
""""""
"""En la variable TrainingTimesLastYear  se observa que hay una variable con un muy bajo porcentaje y otra con muy alto porcentaje por lo cual esta variable no es significativa"""

"""Para el caso de la variable EnvironmentSatisfaction se tienen 4 variables cada una con una representatividad importante por lo cual se decide seguir trabajando con la variable"""
"""En la variable job satisfaction se puede observar que tiene un comportamiento muy similar al de la variable EnvironmentSatisfaction por lo cual se toma la decisión se trabjar solo con una
de estas variables y en este caso sera la de job satisfaction debido a que esta mas enfocada a la satisfaccion con el  puesto o el cargo y esto podria influir en mayor medida en la decisión de 
renunciar al empleo """
"""En lo relacionado con la variable work life balance se observa que la mayoria de datos se distribuyen entre las categorias de satisfecho e insatisfecho por lo cual consideramos que esta variable si podria 
ser representativa dentro de modelo"""
"""Para la variable job involvement hay una categoria que cuenta con mas de la mitad de los datos pero de igual forma aun no se descartara esta variable"""
"""A pesar de que la variable performance rating tiene mas del 80% de los datos en la categoria de un rendimiento bajo no se va descartar aun esta variable porque podria ser importante 
para el desarrollo del modelo"""


#Grafico retirement reason
base = df_expl_cat.groupby(['retirement_reason'])[['Attrition']].count().reset_index().rename(columns ={'Attrition':'count'})
fig = px.pie(base, names='retirement_reason', values='count', title= '<b>Retirement Reason<b>')
xaxis_title = 'retirement_reason'
yaxis_title = 'Cantidad'
template = 'simple_white'
fig.show()

"""Para retirement reason se puede observar que la mayoria  datos se encuentran en la categoria NA ya que son aquellas personas que no han renunciado por lo cual no se tiene una razon de despido"""


#Relación de variables categoricas

df_expl_cat.boxplot("Attrition","Gender",figsize=(15,15),grid=False)


from scipy.stats import chi2_contingency
import pandas as pd
import seaborn as sns

for column in df_expl_cat.columns:
    # Crear una tabla de contingencia
    tabla_contingencia = pd.crosstab(df_expl_cat[column], df_expl_cat['Attrition'])

    # Realizar la prueba chi-cuadrado
    chi2, p, dof, expected = chi2_contingency(tabla_contingencia)

    print(f"Chi-cuadrado: {chi2}, P-valor: {p}, {column}")
    """Si p-valor ≤ α, rechazas la hipótesis nula. Esto indica que hay suficiente evidencia para afirmar que existe una asociación significativa entre las variables categóricas.
Si p-valor > α, no rechazas la hipótesis nula. Esto sugiere que no hay suficiente evidencia para afirmar que existe una asociación entre las variables, y cualquier diferencia 
observada podría atribuirse al azar."""

for column in df_expl_cat.columns:
    # Crear el gráfico de conteo
    sns.countplot(x='Attrition', hue=column, data=df_expl_cat)

    # Añadir títulos y etiquetas (opcional)
    plt.title( f'Distribución de {column} vs de la variable respuesta Attrition')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')

    # Mostrar el gráfico
    plt.show()

#INTERPRETACIÓN
"""No se observa una relacion significativa entre la variable businesstravel y la variable respuesta """
"""Para la variable department tampoco logra verse una relación entre esta y la variable respuesta """
"""En education tampoco hay variación entre las personas que si renuncian y las que no, en realidad la variacion que se observa es por la diferencia en la cantidad de datos"""
"""Para la variable education field  hay muy poca variación al compararla con las categorias de las variables respuesta"""
"""La variable gender tampoco presenta una variación, al igual que en las variables anteriores las varaciones que observan es por la cantidad de datos"""
"""En cuanto a las categorias de la variable job level cuando estas se comparan con la variable respuesta no se ve una relacion significativa"""
"""La variable job role cuenta con 9 categorias por lo cual es mas dificil identificar una relacion entre las variables, pero de igual forma se puede afirmar que esta variable
no influye en la variable respuesta"""
"""En la variable marital status si se puede observar cierta influencia en la variable respuesta teniendo que las personas que mas renuncian son las que estan solteras"""
"""El numero de compañias en el que ha trabajado tampoco  refleja influencia en la variable respuesta"""
"""En la variable TrainingTimesLastYear  no se observa relacion con la variable respuesta y de igual forma las variaciones se deben a la cantidad de datos"""

"""Para el caso de la variable EnvironmentSatisfaction si se observa que podria haber cierte relacion con la variable respuesta, renunciando mas las personas que se encuentran en la categoria de 
muy insatisfecho tal como era de esperarse"""
"""En la variable job satisfaction se puede observar que si hay una influencia y que las personas que mas renuncian son las que se encuentran en las categorias de satisfecho y muy satisfecho """
"""En lo relacionado con la variable work life balance no se observa relacion con la variable respuesta"""
"""Para la variable job involvement no se observa relacion alguna con la variable respuesta"""
"""Para la variable performance rating se observa que no hay variacion pero de igual forma las personas que mas renuncian son aquellas que tienen un rendimiento bajo
aunque esto tambien puede deberse a que estas representan la mayor parte de los datos"""
"""Tal como se esperaba para la variable retirement reason solo se tienen 3 categorias para las personas que si se retiraron organizadas de menor a mayor se encuentra el estres, el salario y otras razones"""



"""VARIABLES IMPORTANTES SEGUN EL ANALISIS EXPLORATORIO
- Age
-MonthlyIncome
-TotalWorkingYears
-YearsAtCompany

-Education
-EducationField
-JobLevel
-JobRole
-NumCompaniesWorked"""
