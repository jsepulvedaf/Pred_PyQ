# -*- coding: utf-8 -*-

"""
Created on Thu Oct  3 08:21:19 2024

@author: jsepulvedaf
"""
import streamlit as st
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from io import BytesIO
import plotly.express as px


def help_button(help_text, button_text="ℹ️ Ayuda"):
    if st.button(button_text):
        st.info(help_text)


texto="""
# Ingreso del Factor Multiplicador Z para Outliers

El factor multiplicador Z es un valor utilizado en la detección de outliers (valores atípicos) mediante el método de puntuación Z (Z-score). Este método identifica outliers basándose en cuántas desviaciones estándar un dato está alejado de la media.

## ¿Qué es el factor Z?

- Es un número que determina cuán "extremo" debe ser un valor para considerarse un outlier.
- Típicamente, se usan valores entre 2 y 3, pero puede variar según el contexto.

## Cómo elegir el factor Z:

- Un valor Z más bajo (ej. 2) es más sensible y detectará más outliers.
- Un valor Z más alto (ej. 3) es más conservador y detectará menos outliers.
"""



# Función para eliminar outliers y ceros, reemplazándolos por NaN
def elimina_outliers_y_zeros(df, columnas, threshold):
    df_outliers = df.copy()  # Crea una copia para guardar outliers
    df_clean = df.copy()
    for col in columnas:
        col_zscore = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
        # Reemplaza outliers y valores en 0 por vacíos (NaN)
        df_clean[col] = df_clean[col].where((col_zscore.abs() < threshold) & (df_clean[col] != 0), np.nan)
    return df_clean  # Retorna solo el dataframe limpiado
def imputar_con_promedio(df, variable):
    for i in range(len(df)):
        if pd.isna(df.loc[i, variable]):
            # Obtiene los 10 valores inmediatamente anteriores, si existen
            prev_values = df[variable].iloc[max(0, i-10):i]
            # Imputa el valor con el promedio de los valores anteriores si no están vacíos
            if not prev_values.isna().all():
                df.loc[i, variable] = prev_values.mean()
    return df
# Función para predecir los datos faltantes
def predecir_e_imputar(df_prediccion, model, scaler, variable_pred, feature_cols):
    df_pred = df_prediccion.copy()

    # Normalizar las columnas de presiones y caudales
    df_pred[st.session_state['all_cols']] = scaler.transform(df_pred[st.session_state['all_cols']])

    # Localizar los valores NaN en la columna a predecir
    valores_faltantes = df_pred[variable_pred].isna()

    if valores_faltantes.any():
        # Tomar solo las filas con NaN en la variable a predecir
        X_pred = df_pred.loc[valores_faltantes, feature_cols]

        if not X_pred.empty:
            # Realizar la predicción sobre los valores faltantes
            predicciones_norm = model.predict(X_pred)

            # Crear un array temporal con las columnas de all_cols para desescalar
            temp_array = np.zeros((len(predicciones_norm), len(st.session_state['all_cols'])))
            temp_array[:, st.session_state['all_cols'].index(variable_pred)] = predicciones_norm.flatten()

            # Desescalar las predicciones
            predicciones = scaler.inverse_transform(temp_array)[:, st.session_state['all_cols'].index(variable_pred)]

            # Imputar las predicciones en los valores NaN del DataFrame original
            df_prediccion.loc[valores_faltantes, variable_pred] = predicciones

        # Después de predecir, imputar los valores que aún son NaN con el promedio de los 10 valores anteriores
        df_prediccion = imputar_con_promedio(df_prediccion, variable_pred)

    return df_prediccion

# Función para graficar datos interactivos
def plot_interactive(data, columns, title, y_label):
    fig = px.line(data, x='fecha', y=columns, title=title)
    fig.update_xaxes(title_text='Fecha y Hora')
    fig.update_yaxes(title_text=y_label)
    return fig

#======================== Principal ========================================
st.title("Predicción e Imputación de Presiones y Caudales usando Redes Neuronales")
st.write(" ")
st.sidebar.title("Bienvenido")
with st.sidebar:
    st.title("Funcionalidad")
    st.write ("""La aplicacion realiza un flujo de 
              trabajo completo para la predicción e imputación de datos
              faltantes en series de tiempo, utilizando técnicas de aprendizaje 
              automático y visualización. Específicamente, se enfoca en la detección
              de outliers, la normalización de datos, el entrenamiento de una red neuronal
              y la generación de gráficas interactivas para analizar los resultados.""")
    st.title("Conceptos Clave")
    show_instructions = st.checkbox("Mostrar especificaciones archivo de entrada")
    
    if show_instructions:
        st.markdown("""
        ### Instrucciones:
         1. Carga tu archivo Excel con los datos de presiones y caudales.
         2. La columna de fecha debe llamarse 'fecha' y el formato debe ser (D/M/A h : mm).
         3. Las columnas de presiones deben empezar con la letra P (por ejemplo: P_AZP, P_pto_Critico, o P1).
         4. Las columnas de caudales deben empezar con la letra C (por ejemplo: C1, Caudal1).
        """)
    st.image("formato.png", caption="ejemplo de configuracion archivo")
    show_instructions = st.checkbox("Conceptos factor multiplicador Z")
    
    if show_instructions:
        st.markdown(texto)    
        
    show_instructions = st.checkbox("Como interpretar la curva de arendizaje")    
    if show_instructions:
         st.markdown("https://www.baeldung.com/cs/loss-vs-epoch-graphs")    
   
    show_instructions = st.checkbox("Conceptos de Error minimo cuadrado")    
    if show_instructions:
         st.markdown("https://encord.com/glossary/mean-square-error-mse/")          
        
st.header('Carga de archivo')



uploaded_file = st.file_uploader("Cargar archivo Excel con la data", type=["xlsx"])

if uploaded_file is not None:
    df_i = pd.read_excel(uploaded_file)
    df = df_i.copy()
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['hora'] = df['fecha'].dt.hour
    df['minuto'] = df['fecha'].dt.minute
    df['Dia_semana'] = df['fecha'].dt.dayofweek

    pressure_cols = [col for col in df.columns if col.startswith('P')]
    flow_cols = [col for col in df.columns if col.startswith('C')]

    st.write(f"Columnas de presión detectadas: {pressure_cols}")
    st.write(f"Columnas de caudal detectadas: {flow_cols}")

    all_cols = pressure_cols + flow_cols
    st.header('Variable a Predecir')
    variable_pred = st.selectbox("Selecciona la variable a predecir (para imputación):", all_cols)
    
    st.header("Detección de Outliers")
    col1, col2 = st.columns(2)
        
    
    Desvest = st.slider("Factor Z", min_value=1.0, max_value=5.0, value=2.5, step=0.5)
      
    
          
    
    st.write(f"Has seleccionado un factor Z de {Desvest}")
    
    
    df_prediccion = elimina_outliers_y_zeros(df, pressure_cols + flow_cols, Desvest)
    df_train = df_prediccion.dropna(subset=all_cols)

    scaler = MinMaxScaler()
    df_train_normalized = df_train.copy()
    df_train_normalized[all_cols] = scaler.fit_transform(df_train[all_cols])

    feature_cols = [col for col in all_cols if col != variable_pred] + ['hora', 'minuto', 'Dia_semana']
    X = df_train_normalized[feature_cols]
    y = df_train_normalized[variable_pred]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    html_temp = """ <div style= background-color:#c9ffcb;padding: 10px;  P {color:WHITE;}><h4> DATOS CRUDOS </h4> </div>"""
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Gráfica de datos crudos
    with st.expander("Gráfica de datos crudos"):
        st.plotly_chart(plot_interactive(df, all_cols, 'Datos crudos', 'Valores'))
        
        
    st.header('Entrenar Red neuronal para variable seleccionada')
    # Ajuste en la sección donde se ejecuta la predicción e imputación
    if st.button('Entrenar red neuronal para predecir e imputación'):
        st.write(f"Entrenando la red neuronal para predecir {variable_pred}...")
    
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'), 
            Dense(1)
        ])
    
        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        loss = model.evaluate(X_test, y_test, verbose=0)
        st.write(f'Pérdida (MSE) en el conjunto de prueba: {loss}')
    
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['variable_pred'] = variable_pred
        st.session_state['feature_cols'] = feature_cols
        st.session_state['all_cols'] = all_cols
    
        # Imputación de los datos faltantes
        df_prediccion_imputed = predecir_e_imputar(df_prediccion.copy(), model, scaler, variable_pred, feature_cols)


        html_temp = """ <div style= background-color:#c9ffcb;padding: 10px;  P {color:WHITE;}><h4> CURVA DE APRENDIZAJE </h4> </div>"""
        st.markdown(html_temp, unsafe_allow_html=True)

        with st.expander("Gráfica de la función de pérdida (loss function) a lo largo de las épocas de entrenamiento"):
            fig = px.line(x=range(1, len(history.history['loss'])+1), 
                          y=[history.history['loss'], history.history['val_loss']], 
                          labels={'x': 'Época', 'y': 'Pérdida'},
                          title='')
            fig.update_layout(legend_title_text='Tipo de Pérdida')
            fig.data[0].name = 'Entrenamiento'
            fig.data[1].name = 'Validación'
            st.plotly_chart(fig)

        # Gráfica de datos de entrenamiento
        with st.expander("Gráfica de datos para  entrenamiento red neuronal"):
            st.plotly_chart(plot_interactive(df_train, all_cols, 'Datos de entrenamiento', 'Valores'))

        # Gráfica de datos imputados (predicciones)
        with st.expander("Gráfica de datos imputados (predicciones)"):
            st.plotly_chart(plot_interactive(df_prediccion_imputed, all_cols, 'Datos imputados', 'Valores'))

        # Comparación entre datos crudos y datos imputados
        with st.expander(f"Comparación entre datos crudos y datos imputados de {variable_pred}"):
            df_comparison = pd.DataFrame({
                'fecha': df['fecha'],
                'Crudo': df[variable_pred],
                'Imputado': df_prediccion_imputed[variable_pred]
            })
            fig_comparison = px.line(df_comparison, x='fecha', y=['Crudo', 'Imputado'], title=f'Comparación de {variable_pred}')
            fig_comparison.update_xaxes(title_text='Fecha y Hora')
            fig_comparison.update_yaxes(title_text=variable_pred)
            st.plotly_chart(fig_comparison)

        
        
       
        output = BytesIO()
        df_prediccion_imputed.to_excel(output, index=False)

        # Botón de descarga con nombre personalizado
        st.download_button(
            label="Descargar datos imputados",
            data=output,
            file_name="",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        st.write("Desarrollado por **Julián M. Sepúlveda**. Contacto: [jsepulvedaf@gmail.com](mailto:jsepulvedaf@gmail.com)")
