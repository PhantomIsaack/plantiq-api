from flask import Flask, request, jsonify
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import uuid
from deepseek_api import generar_informe


# Configuración inicial
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargar el modelo
modelo = load_model('modelo_manzanas.h5')
print("✅ Modelo cargado correctamente.")

# Función para predecir
def predecir_imagen(ruta_imagen):
    img = Image.open(ruta_imagen).resize((100, 100)).convert("RGB")
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediccion = modelo.predict(img_array)[0][0]

    print(f"📊 Predicción del modelo: {prediccion:.4f}")

    if prediccion >= 0.3:
        prompt = (
            "Analiza una manzana y genera un informe agrícola que incluya: "
            "tipo de suelo ideal, nutrientes necesarios, clima óptimo, temperatura, "
            "cuidados generales como riego y fertilización."
        )
        informe = generar_informe(prompt)
        return f"Sí es una manzana\n\n{informe}"

    elif prediccion >= 0.1:
        return "Parece una manzana, pero no estoy 100% seguro.\nPuedes hacer preguntas si lo deseas."

    else:
        return "No se reconoce como manzana"



# Ruta principal
@app.route('/')
def inicio():
    return "API de Plantiq activa. Usa /analizar para enviar una imagen."

# Ruta para analizar imagen
@app.route('/analizar', methods=['POST'])
def analizar():
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400

    archivo = request.files['imagen']
    nombre_archivo = f"{uuid.uuid4().hex}.jpg"
    ruta_guardada = os.path.join(UPLOAD_FOLDER, nombre_archivo)
    archivo.save(ruta_guardada)

    resultado = predecir_imagen(ruta_guardada)

    os.remove(ruta_guardada)  # Borrar imagen después de usarla

    return jsonify({'resultado': resultado})

@app.route('/preguntar', methods=['POST'])
def preguntar():
    pregunta = request.form.get('pregunta')
    contexto = request.form.get('contexto')

    # Validación sencilla
    if any(palabra in pregunta.lower() for palabra in ["riego", "suelo", "clima", "fertilizante", "temperatura", "fruta", "sembrar", "cultivar"]):
        prompt = f"Tema: {contexto}\n\nPregunta del usuario: {pregunta}"
        respuesta = generar_informe(prompt)
        return respuesta
    else:
        return "❌ Esa pregunta no está en nuestro enfoque."

@app.route('/chat', methods=['POST'])
def chat():
    pregunta = request.form.get('pregunta')
    print("📩 Pregunta recibida:", pregunta)

    if not pregunta:
        return "❌ No se recibió ninguna pregunta.", 400

    respuesta = generar_informe(f"Tengo esta duda agrícola: {pregunta}")
    return respuesta


# Ejecutar servidor
if __name__ == '__main__':
    app.run(debug=True)
