import gradio as gr
import requests

URL = "http://127.0.0.1:8000/predict"

def consumir_api(dia, mes, municipio, distribuidora):

    payload = {
        "dia": dia,
        "mes": mes,
        "cups_municipio": municipio,
        "cups_distribuidor": distribuidora
    }
    r = requests.post(URL, json=payload)
    if r.status_code != 200:
        return f"Error de la API: {r.text}" # Esto te dirá el error real
    return r.json()["prediccion_kWh"]

iface = gr.Interface(
    fn=consumir_api,
    inputs=[
        gr.Number(),
        gr.Number(),
        gr.Text(),
        gr.Text()
    ],
    outputs="text"
)

iface.launch()
