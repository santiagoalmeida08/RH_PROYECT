
# Seleccion de variables para el modelo #
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import funciones as fn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Cargar el DataFrame

data_seleccion= 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/base_seleccion.csv'

df = pd.read_csv(data_seleccion,sep=',') ## BASE ORIGINAL ##

df.isnull().sum()

# Create a Random Forest classifier
rf = RandomForestClassifier()

# Fit the model on the selected variables and the target variable
df_class = df.copy()
df_class = df_class.drop('retirement_reason', axis = 1)
df_class = df_class.drop(['fecha_info','EmployeeID'], axis = 1)


le = LabelEncoder() # sirve para volver 1 y 0 las categorias ; si y solo si son 2 categorias

for i in df_class.columns:
    if df_class[i].dtype == 'object' and len(df_class[i].unique()) == 2:
        df_class[i] = le.fit_transform(df_class[i])
    else:
        df_class
        
df_class.head()  
        
df_class_d = pd.get_dummies(df_class)

df_class_d2 = df_class_d.drop('Attrition', axis = 1)


y = df_class_d['Attrition']

# Normalizar df_class_d2
df_class_norm = df_class_d2.copy(deep = True)  # crear una copia del DataFrame
scaler = MinMaxScaler() # asignar el tipo de normalización
sv = scaler.fit_transform(df_class_norm.iloc[:,:]) # normalizar los datos
df_class_norm.iloc[:,:] = sv # asignar los nuevos datos
df_class_norm.head()

rf.fit(df_class_norm, y)

# Get the feature importances
importances = rf.feature_importances_

pd.DataFrame(importances, index = df_class_norm.columns, columns = ['importance']).sort_values('importance', ascending = False).head(20)


""" Seleccionaremos las variables acorde a al analisis exploratorio y a los resultados del modelo de seleccion.
    De esta forma se trabajara con las siguientes variables:
    
    Categoricas : 
    WorkLifeBalance, JobSatisfaction,EnviromentSatisfaction,YearsSionceLastPromotion,MaritialStatus
    NO se trabajaran con el resto de variables ya que estas presentan poca variablidad en sus categorias.
    
    Numericas : 
    MonthlyIncome, Age, DistanceFromHome, YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager, NumCompaniesWorked.
    NO se tendria en cuenta la variable totalworkingyears ya que esta correlacionada con age y yearsatcompany."""


   