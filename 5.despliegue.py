
# Importar librerias necesarias

import funciones as funciones  ###archivo de funciones propias
import pandas as pd ### para manejo de datos
import joblib
import openpyxl ## para exportar a excel
import numpy as np

# Importar elementos necesarios para despliegue
rf_final = joblib.load("salidas\\rf_final.pkl")
list_oe=joblib.load("salidas\\list_oe.pkl")
list_le=joblib.load("salidas\\list_le.pkl")
list_dd=joblib.load("salidas\\list_dd.pkl")
list_cat=joblib.load("salidas\\list_cat.pkl")
var_names=joblib.load("salidas\\var_names.pkl")
scaler=joblib.load("salidas\\scaler.pkl") 

# Cargar base de datos con la que se quiere hacer predicciones

df_pred= 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/baseprediccion.csv'
df = pd.read_csv(df_pred,sep=',')
df.isnull().sum() # Verificar valores nulos y variables
df.columns

df_t=funciones.preparar_datos(df) # Transformación de datos para realizar predicciones
df_t.columns

#Cargar modelo y predecir
rf_final = joblib.load("salidas\\rf_final.pkl")
predicciones=rf_final.predict(df_t)
pd_pred=pd.DataFrame(predicciones, columns=['Attrition_17'])

#Crear base con predicciones
perf_pred=pd.concat([df['EmployeeID'],df_t,pd_pred],axis=1)

perf_pred['Attrition_17'].value_counts() # Verificar valores nulos
   
####LLevar a BD para despliegue 
#perf_pred.loc[:,['EmployeeID', 'Attrition_17']].to_sql("perf_pred",conn,if_exists="replace") ## llevar predicciones a BD con ID Empleados

####ver_predicciones_bajas ###
#emp_pred_bajo=perf_pred.sort_values(by=["Attrition_17"],ascending=True).head(10)
    
#emp_pred_bajo.set_index('EmployeeID', inplace=True) 
#pred=emp_pred_bajo.T
    
  ### agregar coeficientes
#perf_pred.to_excel("salidas\\predicciones.xlsx")   #### exportar todas las  predicciones 

pred.to_excel("salidas\\prediccion.xlsx")   #### exportar predicciones mas bajas y variables explicativas
 ### exportar coeficientes para analizar predicciones
    


perf_pred.isnull().sum() # Verificar valores nulos

perf_pred['Attrition_17'].value_counts() # Verificar valores nulos

#Importancia de las variables del modelo
importances = perf_pred.feature_importances_
feature_importances_df = pd.DataFrame({'Feature': df_t.columns, 'Importance': importances})
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)#Base de datos con la importancia de mayor a menor
 #Ruta de la decisión
#perf_pred.tree_.node_count
#dense_matrix = pd.DataFrame(perf_pred.decision_path(df_t).todense())
#ruta= pd.concat([perf_pred, dense_matrix,], axis=1)
#ruta=ruta[ruta["renuncia"]==1]
#ruta.to_sql("Ruta_decision",con,if_exists="replace") ## llevar predicciones a BD con ID Empleados
 #Al tener un árbol de este tamaño se hace complejo observar la ruta de decisión
#Guardado en Excel
#ruta=ruta.head()