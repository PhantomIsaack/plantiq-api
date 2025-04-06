from openai import OpenAI

# 🔐 Reemplaza con tu clave de API de NVIDIA NGC
NVIDIA_API_KEY = "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"

# Cliente de OpenAI apuntando al endpoint de NVIDIA
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

def generar_informe(prompt_usuario):
    try:
        print("🧠 Prompt enviado a Deepseek vía NVIDIA:")
        print(prompt_usuario)

        response = client.chat.completions.create(
            model="deepseek-ai/deepseek-r1",
            messages=[
                {"role": "system", "content": "Eres un experto en agricultura y cultivos."},
                {"role": "user", "content": prompt_usuario}
            ],
            temperature=0.6,
            max_tokens=1024
        )

        print("✅ Respuesta recibida correctamente.")
        return response.choices[0].message.content

    except Exception as e:
        print("❌ Error al consultar NVIDIA NIM:")
        print(e)
        return "⚠️ Error al consultar el modelo Deepseek desde NVIDIA."
