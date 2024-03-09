import os
import tempfile
import numpy as np
import pandas as pd
import datetime as dt

# Función para calcular recency para cada fila
def calcular_recency(row):
    for i, columna in enumerate(columnas_deuda):
        if row[columna] > 0:
            return i
    # Si no hay deuda en ninguno de los meses, entonces recency es el último mes (0-indexed)
    return len(columnas_deuda)


# Función para calcular Frequency para cada fila
def calcular_frequency(row):
    meses_con_deuda = sum(row[columna] > 0 for columna in columnas_deuda)
    return meses_con_deuda

# Función para calcular Monetary para cada fila
def calcular_monetary(row):
    for columna in reversed(columnas_deuda):
        if row[columna] > 0:
            return row[columna]
    # Si no hay deuda en ninguno de los meses, retorna 0
    return 0

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    #Read Data
    df = pd.read_csv(
        f"{base_dir}/input/storedata_total.csv"
    )

    # Convertir variables categóricas numéricas a tipo object
    df[['COD_OFI', 'COD_COM']] = df[['COD_OFI', 'COD_COM']].astype('object')

    # Reemplazar valores de edad fuera del rango válido con NaN
    df.loc[(df['EDAD'] < 0) | (df['EDAD'] > 122), 'EDAD'] = np.nan

    # Eliminar filas con valores faltantes
    df.dropna(inplace=True)

    # Crear una variable RENTA_EDAD
    df['RENTA_EDAD'] = df['RENTA'] * df['EDAD']

    # Calcular la diferencia entre las deudas de los últimos dos meses
    df['DIF_SEPT_AGOS'] = df['D_Septiembre'] - df['D_Agosto']

    columnas_deuda = ['D_Septiembre', 'D_Agosto', 'D_Julio', 'D_Junio', 'D_Mayo', 'D_Abril', 'D_Marzo']

    # Calcular la deuda promedio de cada cliente
    df['DEUDA_PROM'] = df[columnas_deuda].mean(axis=1)

    # Calcular Recency y Frequency
    df['Recency'] = df.apply(calcular_recency, axis=1)
    df['Frequency'] = df.apply(calcular_frequency, axis=1)
    df['Monetary'] = df.apply(calcular_monetary, axis=1)

    # Eliminar columnas no deseadas
    columnas_a_eliminar = ['RENTA', 'D_Mayo', 'D_Agosto', 'RENTA_EDAD', 'Monetary']
    df.drop(columns=columnas_a_eliminar, axis=1, inplace=True)

    # Agrupar la variable CIUDAD y reemplazar valores
    df['CIUDAD_AGRUP'] = np.where(df['CIUDAD'] == 'SANTIAGO', 1, 0)

    # Reemplazar valores de NIV_EDUC específicos con NaN
    df.loc[df['NIV_EDUC'].isin(['EUN', 'BAS']), 'NIV_EDUC'] = np.nan

    # Eliminar filas con valores faltantes
    df.dropna(inplace=True)

    # Eliminar columnas no deseadas
    columnas_a_eliminar = ['COD_COM', 'CIUDAD']
    df.drop(columns=columnas_a_eliminar, axis=1, inplace=True)

    # Calcular el promedio de Recency por oficina y crear una nueva variable
    promedio_recency_por_oficina = df.groupby('COD_OFI')['Recency'].mean().reset_index()
    promedio_recency_por_oficina['OFICINA_CON_PAGOS_AL_DIA'] = np.where(promedio_recency_por_oficina['Recency'] == 0, 'SI', 'NO')
    df = pd.merge(df, promedio_recency_por_oficina[['COD_OFI', 'OFICINA_CON_PAGOS_AL_DIA']], on='COD_OFI', how='left')

    # Crear una nueva variable CLIENTE_AL_DIA
    df['CLIENTE_AL_DIA'] = np.where(df['Recency'] == 0, 'SI', 'NO')

    # Eliminar columnas no deseadas
    columnas_a_eliminar = ['COD_OFI', 'OFICINA_CON_PAGOS_AL_DIA']
    df.drop(columns=columnas_a_eliminar, axis=1, inplace=True)

    # Convertir la variable EDAD de float a int
    df['EDAD'] = df['EDAD'].astype(int)

    # Codificar variables categóricas
    codificacion_binaria = {'M': 1, 'F': 0, 'SI': 1, 'NO': 0, 'FUGA': 1, 'NO FUGA': 0, 'SANTIAGO': 1, 'OTRAS': 0}
    df.replace(codificacion_binaria, inplace=True)

    # Mapear y codificar la variable NIV_EDUC
    nivel_educacion = {'MED': 1, 'TEC': 2, 'UNV': 3}
    df['NIV_EDUC'] = df['NIV_EDUC'].map(nivel_educacion)

    # Crear variables dummy para E_CIVIL
    dummy_variables = pd.get_dummies(df['E_CIVIL'])
    df = pd.concat([df, dummy_variables], axis=1)
    df.drop('E_CIVIL', axis=1, inplace=True)

    # Eliminar columnas no deseadas
    columnas_a_eliminar = ['CIUDAD_AGRUP', 'CLIENTE_AL_DIA', 'D_Marzo', 'Frequency', 'Recency', 'SEGURO', 'SEP', 'SOL', 'VIU']
    df.drop(columns=columnas_a_eliminar, axis=1, inplace=True)

    
    y = df.pop("FUGA")
    X_pre = df
    y_pre = y.to_numpy().reshape(len(y), 1)
    X = np.concatenate((y_pre, X_pre), axis=1)
    
    # Establecer el seed para hacer la división replicable
    np.random.seed(42)
    np.random.shuffle(X)
    
    # Split in Train, Test and Validation Datasets
    train, validation, test = np.split(X, [int(.7*len(X)), int(.85*len(X))])
    train_rows = np.shape(train)[0]
    validation_rows = np.shape(validation)[0]
    test_rows = np.shape(test)[0]
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    validation = pd.DataFrame(validation)
    # Convert the label column to integer
    train[0] = train[0].astype(int)
    test[0] = test[0].astype(int)
    validation[0] = validation[0].astype(int)
    # Save the Dataframes as csv files
    train.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    validation.to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    test.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)