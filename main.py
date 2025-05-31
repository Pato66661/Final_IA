from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Configuración para archivos estáticos (robots.txt, etc.)
#app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Cargar el modelo de enfermedades cardíacas
try:
    model = joblib.load("modelo_random.pkl")
    print("Modelo cargado correctamente")
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")
    raise e

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    """Muestra el formulario de entrada para los datos del paciente"""
    # Valores por defecto para el ejemplo
    default_values = {
        "age": 65,
        "sex": 1,
        "cp": 3,
        "trestbps": 120,
        "chol": 200,
        "fbs": 0,
        "restecg": 0,
        "thalach": 130,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 1,
        "ca": 0,
        "thal": 1
    }
    return templates.TemplateResponse("prueba_actualizada.html", {"request": request, **default_values})

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    age: int = Form(...),
    sex: int = Form(...),
    cp: int = Form(...),
    trestbps: int = Form(...),
    chol: int = Form(...),
    fbs: int = Form(...),
    restecg: int = Form(...),
    thalach: int = Form(...),
    exang: int = Form(...),
    oldpeak: float = Form(...),
    slope: int = Form(...),
    ca: int = Form(...),
    thal: int = Form(...)
):
    """Procesa los datos del formulario y devuelve la predicción"""
    try:
        # Validación adicional de los datos
        if age < 18 or age > 120:
            raise ValueError("La edad debe estar entre 18 y 120 años")
        
        # Crear array con las características en el orden correcto
        features = np.array([[
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]])
        
        # Realizar predicción
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1] * 100  # Probabilidad de enfermedad
        
        # Formatear resultado
        result_class = "Riesgo cardíaco detectado" if prediction == 1 else "Sin riesgo cardíaco significativo"
        result_message = f"{result_class} (Probabilidad: {proba:.2f}%)"
        
        return templates.TemplateResponse(
            "prueba_actualizada.html",
            {
                "request": request,
                "result": result_message,
                "show_result": True,
                "age": age,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal
            }
        )
    
    except Exception as e:
        return templates.TemplateResponse(
            "prueba_actualizada.html",
            {
                "request": request,
                "error": f"Error al procesar la solicitud: {str(e)}",
                "age": age,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal
            }
        )

# Manejo de robots.txt
@app.get("/robots.txt")
def robots_txt():
    return "User-agent: *\nDisallow: /"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
