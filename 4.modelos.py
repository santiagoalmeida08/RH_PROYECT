
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

# Cargar el DataFrame

data_seleccion= 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/base_seleccion.csv'

df = pd.read_csv(data_seleccion,sep=',') ## BASE ORIGINAL ##

df.dtypes

df.isnull().sum()
df = df.drop('EmployeeID', axis = 1)

df1 = df.copy()
df1.info()

df1['JobLevel'] = df1['JobLevel'].astype('object')  

df1['NumCompaniesWorked'] = df1['NumCompaniesWorked'].astype('int64')
df1['PercentSalaryHike'] = df1['PercentSalaryHike'].astype('int64')
df1['WorkLifeBalance'] = df1['WorkLifeBalance'].astype('int64')
df1['JobSatifaction'] = df1['JobSatisfaction'].astype('int64')
df1['EnviromentSatisfaction'] = df1['EnvironmentSatisfaction'].astype('int64')

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

#Normalizacion

v_num = []
for col in df3.columns:
    if df3[col].dtypes == "int64":
        v_num.append(col)

scaler = MinMaxScaler()
for col in v_num:
    df3[[col]] = scaler.fit_transform(df3[[col]])

df3.head()

X_esc = df3.drop('Attrition', axis = 1)
y = df3['Attrition']

# Selecccion de modelos #

model_gb = GradientBoostingClassifier()
model_arb = DecisionTreeClassifier(max_depth=4, random_state=42) 
model_log = LogisticRegression( max_iter=1000, random_state=42)
model_rand = RandomForestClassifier(n_estimators = 100,#o regresation
                               criterion = 'gini',#error
                               max_depth = 5,#cuantos arboles
                               max_leaf_nodes = 10,#profundidad
                               max_features = None,#nodos finales
                               oob_score = False,
                               n_jobs = -1,
                               random_state = 123)

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


var_names= sel_variables(modelos,X_esc,y,threshold="2.8*mean") 
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

f1sco_df = medir_modelos(modelos,"accuracy",X_esc,y,10)  #se definen 10 iteraciones para tener mejor visión del desempeño en el boxplot
f1dco_var_sel = medir_modelos(modelos,"accuracy",df4,y,10)


f1s=pd.concat([f1sco_df,f1dco_var_sel],axis=1) 
f1s.columns=['rlog', 'dtree', 'rforest', 'gboosting',
       'rl_Sel', 'dt_sel', 'rf_sel', 'gb_Sel']


f1sco_df.plot(kind='box') #### gráfico para modelos todas las varibles
f1dco_var_sel.plot(kind='box') ### gráfico para modelo variables seleccionadas
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























"""

# PRUEBA DEL DESEMPEÑO DE CLASIFICADOR CON HIPERPARAMETROS AJUSTADOS #

# Particion de datos para entrenamiento y evaluación
X_train, X_test, y_train, y_test = train_test_split(X_esc,y,shuffle=True , test_size = 0.3, random_state = 42)

os =  RandomOverSampler()
x_train_res, y_train_res = os.fit_resample(X_train, y_train)

print ("Distribution before resampling {}".format(Counter(y_train)))
print ("Distribution labels after resampling {}".format(Counter(y_train_res)))

rf_final.fit(x_train_res,y_train_res)

from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
#plot_tree(rf_final, filled=True, feature_names = x.columns, class_names = ['No','Yes'])
plt.show()

y_pred = model_arb.predict(X_test)

#Matriz de confusión #
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=model_arb.classes_)
cmd.plot()

print(metrics.classification_report(y_test, model_arb.predict(X_test)))



x = df4.drop('Attrition', axis = 1)
y = df4['Attrition']

X_train, X_test, y_train, y_test = train_test_split(x,y,shuffle=True , test_size = 0.3, random_state = 42) 



# Modelo Gradient Boosting #

model_gb = GradientBoostingClassifier()

model_gb.fit(x_train_res, y_train_res)
y_pred_gb = model_gb.predict(X_test)


# Matriz de confusión y su display

cm = confusion_matrix(y_test, y_pred_gb)
cmd = ConfusionMatrixDisplay(cm, display_labels=model_gb.classes_)
cmd.plot()
plt.show()

# Metricas de desempeño del modelo Gradient Boosting #
print(metrics.classification_report(y_test, model_gb.predict(X_test)))

# Modelo Arboles de desicion #

model_arb = DecisionTreeClassifier(class_weight='balanced', max_depth=4, random_state=42)

model_arb.fit(x_train_res, y_train_res)

# visualizar el árbol

from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
plt.figure(figsize=(20,10))
plot_tree(model_arb, filled=True, feature_names = x.columns, class_names = ['No','Yes'])
plt.show()

y_pred = model_arb.predict(X_test)

#Matriz de confusión #
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=model_arb.classes_)
cmd.plot()

print(metrics.classification_report(y_test, model_arb.predict(X_test)))




#MODELO DE REGRESIÓN LOGISTICA

model_log = LogisticRegression() # definir el modelo de regresión losgistica
model_log.fit(x_train_res,y_train_res) # entrenar el modelo
y_pred_train = model_log.predict(x_train_res) # guardar la predicción para train
y_pred_test = model_log.predict(X_test) # guardar la predicción para test

#Metricas de desempeño modelo de regresión logistica
# Matriz de confusión:
cm = confusion_matrix(y_test, y_pred_test, labels=model_log.classes_) # guardar las clases para la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_log.classes_)
disp.plot();
print(cm)


TP=cm[0,0]
FP=cm[1,0]
FN=cm[0,1]
TN=cm[1,1]

print(f"Accuracy test: {accuracy_score(y_test, y_pred_test)}")
print(f'Precicion: {TP/(TP+FP)}')
print(f'Recall (Sensibilidad)): {TP/(TP+FN)}')
print(f'F1-score:', f1_score(y_test, y_pred_test, average='binary'))
print(f'Especificidad: {TN/(FP+TN)}')


#MODELO RANDOM FOREST
model = RandomForestClassifier(n_estimators = 100,#o regresation
                               criterion = 'gini',#error
                               max_depth = 5,#cuantos arboles
                               max_leaf_nodes = 10,#profundidad
                               max_features = None,#nodos finales
                               oob_score = False,
                               n_jobs = -1,
                               random_state = 123)
model.fit(x_train_res, y_train_res)

# Matriz de confusión para el modelo Random Forest
cm_rf = confusion_matrix(y_test, model.predict(X_test))
cmd_rf = ConfusionMatrixDisplay(cm_rf, display_labels=model.classes_)
cmd_rf.plot()
plt.show()

print(metrics.classification_report(y_test, model.predict(X_test)))

"""