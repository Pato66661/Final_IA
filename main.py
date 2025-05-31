from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Cargar el modelo de enfermedades cardíacas
model = joblib.load("modelo_random.pkl")

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    """Muestra el formulario de entrada para los datos del paciente"""
    return templates.TemplateResponse("prueba_actualizada.html", {"request": request})

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
        # Crear array con las características en el orden correcto
        features = np.array([[
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]])
        
        # Realizar predicción
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1] * 100  # Probabilidad de enfermedad
        
        # Formatear resultado
        result_class = "Enfermedad cardíaca" if prediction == 1 else "Sano"
        result_message = (
            f"Resultado: {result_class} "
            f"(Probabilidad: {proba:.2f}%)"
        )
        
        return templates.TemplateResponse(
            "heart_form.html",
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
            "heart_form.html",
            {
                "request": request,
                "error": f"Error al procesar la solicitud: {str(e)}"
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)