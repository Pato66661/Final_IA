import joblib
import pandas as pd

def cargar_modelo(ruta_modelo="modelo_random.pkl"):
    """Carga el modelo entrenado desde un archivo .pkl"""
    try:
        modelo = joblib.load(ruta_modelo)
        print("Modelo cargado exitosamente")
        return modelo
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_modelo}")
        return None
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")
        return None

def predecir_enfermedades_cardiacas(modelo):
    """Realiza predicciones sobre datos de pacientes"""
    # Datos de ejemplo para múltiples pacientes
    datos_pacientes = [
        [65, 1, 3, 120, 200, 0, 0, 130, 0, 0.0, 1, 0, 1],  # Paciente 1
        [45, 0, 2, 140, 280, 1, 1, 160, 1, 2.3, 2, 2, 2]    # Paciente 2
    ]
    
    # Columnas correspondientes a las características del modelo
    columnas = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    
    try:
        # Crear DataFrame con los datos
        pacientes = pd.DataFrame(datos_pacientes, columns=columnas)
        
        # Realizar predicciones
        predicciones = modelo.predict(pacientes)
        probabilidades = modelo.predict_proba(pacientes)[:, 1] * 100  # Convertir a porcentaje
        
        # Añadir resultados al DataFrame
        resultados = pacientes.copy()
        resultados['Predicción'] = ['Enfermedad' if p == 1 else 'Sano' for p in predicciones]
        resultados['Probabilidad (%)'] = probabilidades.round(2)
        
        return resultados
    
    except Exception as e:
        print(f"Error al realizar predicciones: {str(e)}")
        return None

if __name__ == "__main__":
    # Paso 1: Cargar el modelo
    modelo = cargar_modelo()
    
    if modelo is not None:
        # Paso 2: Realizar predicciones
        resultados = predecir_enfermedades_cardiacas(modelo)
        
        if resultados is not None:
            print("\nRESULTADOS DE PREDICCIÓN PARA PACIENTES")
            print("---------------------------------------")
            # Mostrar solo las columnas más relevantes para el reporte
            print(resultados[['age', 'sex', 'Predicción', 'Probabilidad (%)']])