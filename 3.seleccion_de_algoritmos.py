
# Seleccion de variables para el modelo #

# Create a Random Forest classifier
rf = RandomForestClassifier()

# Fit the model on the selected variables and the target variable
df_class = df_no_null2.copy()
df_class = df_class.drop('retirement_reason', axis = 1)
df_class['Attrition'] = df_class['Attrition'].replace({'yes':1, 'no':0})
df_class = df_class.drop(['fecha_info','EmployeeID'], axis = 1)

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

pd.DataFrame(importances, index = df_class_norm.columns, columns = ['importance']).sort_values('importance', ascending = False).head(14)


