
import funciones as funciones  ###archivo de funciones propias
import pandas as pd ### para manejo de datos
import joblib
import openpyxl ## para exportar a excel
import numpy as np


### funcion para cargar objeto guardado ###
rf_final = joblib.load("salidas\\rf_final.pkl")
list_oe=joblib.load("salidas\\list_oe.pkl")
list_le=joblib.load("salidas\\list_le.pkl")
list_dd=joblib.load("salidas\\list_dd.pkl")
list_cat=joblib.load("salidas\\list_cat.pkl")
var_names=joblib.load("salidas\\var_names.pkl")
scaler=joblib.load("salidas\\scaler.pkl") 


###### el despliegue consiste en dejar todo el código listo para una ejecucion automática en el periodo definido:
###### en este caso se ejecutara el proceso de entrenamiento y prediccion anualmente.
#if __name__=="__main__":

### conectarse a la base de datos ###
#base de datos de 2016 para predecir 2017
df_pred= 'https://raw.githubusercontent.com/santiagoalmeida08/RH_PROYECT/main/data_hr_proyect/baseprediccion.csv'
df = pd.read_csv(df_pred,sep=',')
df.isnull().sum() # Verificar valores nulos
 # verificar que tenga las mismas columnas que df4
    ####Otras transformaciones en python (imputación, dummies y seleccion de variables)

df = df.drop(['Over18','StandardHours','EmployeeCount'], axis=1) # Eliminar columnas que no aportan información relevante
df.columns

def preparar_datos (df):
   
    #######Cargar y procesar nuevos datos ######
   
    
    #### Cargar modelo y listas 
    
   
    #list_cat=joblib.load("salidas\\list_cat.pkl")
    list_oe = joblib.load("salidas\\list_oe.pkl")
    list_le=joblib.load("salidas\\list_le.pkl")
    list_dummies=joblib.load("salidas\\list_dd.pkl")
    var_names=joblib.load("salidas\\var_names.pkl")
    scaler=joblib.load( "salidas\\scaler.pkl") 

    ####Ejecutar funciones de transformaciones
    
    df=funciones.nulos(df)
    df_dummies = df.copy()
    df_dummies= funciones.encode_data(df, list_le, list_dummies,list_oe)
    df_dummies= df_dummies.loc[:,~df_dummies.columns.isin(['EmployeeID '])]
    
    
    #X2=scaler.transform(df_dummies)
    #X=pd.DataFrame(X2,columns=df_dummies.columns)
    #X=X[var_names]
    
    return df_dummies

df_t= preparar_datos(df)
df_t.columns

##Cargar modelo y predecir
rf_final = joblib.load("salidas\\rf_final.pkl")
predicciones=rf_final.predict(df_t)
pd_pred=pd.DataFrame(predicciones, columns=['Attrition_17'])


###Crear base con predicciones ####

perf_pred=pd.concat([df['EmployeeID'],df_t,pd_pred],axis=1)
   
####LLevar a BD para despliegue 
#perf_pred.loc[:,['EmployeeID', 'Attrition_17']].to_sql("perf_pred",conn,if_exists="replace") ## llevar predicciones a BD con ID Empleados
    

####ver_predicciones_bajas ###
emp_pred_bajo=perf_pred.sort_values(by=["Attrition_17"],ascending=True).head(10)
    
emp_pred_bajo.set_index('EmployeeID', inplace=True) 
pred=emp_pred_bajo.T
    
coeficientes=pd.DataFrame( np.append(rf_final.intercept_,rf_final.coef_) , columns=['coeficientes'])  ### agregar coeficientes
perf_pred.to_excel("salidas\\predicciones.xlsx")   #### exportar todas las  predicciones 

pred.to_excel("salidas\\prediccion.xlsx")   #### exportar predicciones mas bajas y variables explicativas
coeficientes.to_excel("salidas\\coeficientes.xlsx") ### exportar coeficientes para analizar predicciones
    








    
