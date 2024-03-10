# Gradient Boosting
# Arboles de desicion 
import pandas as pd 
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
#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier 


# Cargar el DataFrame

data_seleccion= 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/base_seleccion.csv'

df = pd.read_csv(data_seleccion,sep=',') ## BASE ORIGINAL ##

df.isnull().sum()

"""Para el modelo de Gradient Boosting se seleccionan las siguientes variables :
    -Age
    -MonthlyIncome
    -YearsWithCurrManager
    -StockOptionLevel
    -MaritalStatus
    -EnvironmentSatisfaction"""
    
df1 = df[['Age','MonthlyIncome','YearsWithCurrManager','StockOptionLevel','MaritalStatus','EnvironmentSatisfaction','Attrition']]
df1.head()

# DUMMIES #

#LabelEncoder
df2 = df1.copy()   
le = LabelEncoder() 

for i in df2.columns:
    if df2[i].dtype == 'object' and len(df2[i].unique()) == 2:
        df2[i] = le.fit_transform(df2[i])
    else:
        df2
        
#Get Dummies
df3 = pd.get_dummies(df2)
df3.head()

#Normalizacion

v_num = []
for col in df3.columns:
    if df3[col].dtypes == "int64":
        v_num.append(col)

scaler = MinMaxScaler()
for col in v_num:
    df3[[col]] = scaler.fit_transform(df[[col]])

df3.head()
    
# Division data en train y test #

df4 = df3.copy()
x = df4.drop('Attrition', axis = 1)
y = df4['Attrition']

X_train, X_test, y_train, y_test = train_test_split(x,y,shuffle=True , test_size = 0.3, random_state = 42) 

# Modelo Gradient Boosting #

model_gb = GradientBoostingClassifier()

model_gb.fit(X_train, y_train)
y_pred_gb = model_gb.predict(X_test)

# Matriz de confusión y su display
cm = confusion_matrix(y_test, y_pred_gb)
cmd = ConfusionMatrixDisplay(cm, display_labels=model_gb.classes_)
cmd.plot()
plt.show()

# Accuracy
accuracy = accuracy_score(y_test, y_pred_gb)
print("Accuracy:", accuracy)


# Modelo Arboles de desicion #

model_arb = DecisionTreeClassifier(class_weight='balanced')

model_arb.fit(X_train, y_train)

# visualizar el árbol

from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
plot_tree(model_arb, filled=True, feature_names = x.columns)
plt.show()

#METRICAS TRAIN
y_true = model_arb.predict(X_train)
print(metrics.classification_report(y_train, y_true, digits = 3))   

#METRICAS TEST
y_true = model_arb.predict(X_test)
print(metrics.classification_report(y_test, y_true, digits = 3))

#Matriz de confusión #
cm = confusion_matrix(y_test, y_true)
cmd = ConfusionMatrixDisplay(cm, display_labels=model_arb.classes_)
cmd.plot()

#MODELO DE REGRESIÓN LOGISTICA

model_log = LogisticRegression() # definir el modelo de regresión losgistica
model_log.fit(X_train,y_train) # entrenar el modelo
y_pred_train = model_log.predict(X_train) # guardar la predicción para train
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
model.fit(X_train, y_train)
print(metrics.classification_report(y_test, model.predict(X_test)))