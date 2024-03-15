# Modelos #

"""En este apartado se realizara la transformación y selección de variables para realizar el entrenamiendo de los modelos"""

#1. Codificación de variables
#2. Escalado de variables
#3. Selección de variables
#4. Selección de algoritmo ganador
#5. Afinamiento de hiperparametros
#6. Exportar modelo ganador

#Paquetes para manejo de datos
import pandas as pd 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
import funciones as fn

# Importar el dataframe deputado en el analisis exploratorio

data_seleccion= 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/base_seleccion2.csv'
df = pd.read_csv(data_seleccion,sep=',')
df.isnull().sum() # Verificar valores nulos
df.dtypes

df1 = df.copy()
df1 = df1.drop('EmployeeID', axis = 1) # Eliminar columna EmployeeID ya que no aporta información relevante
df1.info()#Visualizar tipo de datos y verificar variables 

df1['JobLevel'] = df1['JobLevel'].astype('object') #transformar JobLevel a variable categorica
df1['Attrition'].value_counts()# Cambiar valores de Attrition a 1 y 0

#Cambiamos el formato de las variables a int64 para poder trabajar con ellas en un solo formato
for i in df1.columns:
    if df1[i].dtypes == "float64":
        df1[i] = df1[i].astype('int64')
    else:
        pass

df1.dtypes # Verificar cambios

#1. Codificación de variables
#Guardamos las variables categoricas en una lista para poder realizar la codificación de las variables acorde a sus variables 
list_cat = [df1.columns[i] for i in range(len(df1.columns)) if df1[df1.columns[i]].dtype == 'object']
list_oe = ['JobLevel']
list_le = [df1.columns[i] for i in range(len(df1.columns)) if df1[df1.columns[i]].dtype == 'object' and len(df1[df1.columns[i]].unique()) == 2]
list_dd = ['Department','Education','EducationField','JobRole','MaritalStatus','BusinessTravel']

# Codificación de variables 
df_encoded = fn.encode_data(df1, list_le,list_dd,list_oe)
df3 = df_encoded.copy()

y = df3['Attrition'] # variable objetivo
df3 = df3.drop(['Attrition'], axis = 1) # Eliminar columna EmployeeID ya que no aporta información relevante

#2. Escalado de variables

scaler = RobustScaler() # se usa robust scaler en el escalado para evitar la influencia de outliers
x = scaler.fit_transform(df3) # se escalan las variables predictoras
X_esc = pd.DataFrame(x, columns = df3.columns)
X_esc.columns

# Algorirmos a seleccionar #
model_gb = GradientBoostingClassifier()
model_arb = DecisionTreeClassifier() 
model_log = LogisticRegression( max_iter=1000, random_state=42)
model_rand = RandomForestClassifier()
modelos  = list([model_gb,model_arb,model_log,model_rand])

#3. Seleccion de variables con base a los modelos seleccionados

var_names= fn.sel_variables(modelos,X_esc,y,threshold="3.1*mean") 
var_names.shape
# Al utiizar en el treshhold numeros menores se aceptaban mas variables sin embargo el desempeño del modelo con todas las variables era muy similar al desempeño con variables seleccionadas
# Debido a lo anterior se utilizo un treshold de 3.1 en el cual se trabaja con 4 variables las cuales aportan significia a los modelos

df_var_sel = df3[var_names]
df_var_sel.info()

#4. Seleccion de algoritmo ganador #

df4 = df_var_sel.copy()
f1sco_df = fn.medir_modelos(modelos,"f1",X_esc,y,15)  #se definen 15 iteraciones para tener mejor visión del desempeño en el boxplot
f1dco_var_sel = fn.medir_modelos(modelos,"f1",df4,y,15) # utilizamos el f1 score como metrica ya que esta metrica le da igual importancia a falsos positivos y falsos negativos, buscamos observar cual es la precisión general del modelo al hacer la clasificación

f1s=pd.concat([f1sco_df,f1dco_var_sel],axis=1) 
f1s.columns=['rlog', 'dtree', 'rforest', 'gboosting',
       'rl_Sel', 'dt_sel', 'rf_sel', 'gb_Sel']
f1s.plot(kind='box') # Boxplot de f1 score para cada modelo con todas las variables y con las variables seleccionadas
f1s.mean()  # Media de rendimiendo para cada variable 

#En el boxplot de rendimiento observamos que los modelos con mayor rendimiento son descision tree clasifier y gardient boosting, con un rendimiento medio de
#0.67 y 0.72 respectivamente. Buscando una mayor interpretabilidad de los datos y un mejor rendimiento computacional se selecciona el modelo de decision tree classifier

#Matriz confusion para dt_sel
model_arb = DecisionTreeClassifier(max_depth=4, random_state=42)
model_arb.fit(df4,y)
y_pred = model_arb.predict(df4)
cm = confusion_matrix(y, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=model_arb.classes_)
cmd.plot()

#Definicion de parametros para DecisionTreeClassifier
parameters = {
    'criterion': ['gini', 'entropy'], # utilizaremos gini y entropy para medir la calidad de la division de los nodos
    'max_depth': [5, 10, 15], # decidimos probar con 5, 10 y 15 niveles de profundidad debido a 
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    } #max_feauters se refiere a la cantidad de variables que se toman en cuenta para hacer la mejor division

# create an instance of the randomized search object
r1 = RandomizedSearchCV(DecisionTreeClassifier(), parameters, cv=5, n_iter=100, random_state=42, n_jobs=-1, scoring='accuracy') 

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

eval=cross_validate(rf_final,df4,y,cv=5,scoring='recall',return_train_score=True) #evaluacion de modelo final con cross validation
#Como metrica de evaluacion final utilizaremos el recall; debido a que a nos preocupa el numero de falsos negativos, es decir, el numero 
#de empleados que se van y que el modelo no logra predecir

train_rf=pd.DataFrame(eval['train_score'])
test_rf=pd.DataFrame(eval['test_score'])
train_test_rf=pd.concat([train_rf, test_rf],axis=1)
train_test_rf.columns=['train_score','test_score']
train_test_rf

train_test_rf["test_score"].mean()
#El recall promedio del modelo es de 0.68, lo cual indica que el modelo es capaz de predecir correctamente el 68% de los empleados que se van
# Sin embarigo, como se observa en la tabla train_test_rf, el modelo en algunos casos tiene amplias diferencias entre el rendimiento en el set de entrenamiento y el set de prueba, lo cual indica que el modelo puede estar sobreajustando.

#haz una matriz de confusion para el modelo final con la data de cross validation
y_pred = cross_val_predict(rf_final,df4,y,cv=5)
cm = confusion_matrix(y, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=rf_final.classes_)
cmd.plot()

#El modelo tiene un rendimiento aceptable, sin embargo, tal como se esperaba el modelo tiene un alto numero de falsos negativos, es decir, el modelo predice que 266 empleados no se van, cuando en realidad si se van.


