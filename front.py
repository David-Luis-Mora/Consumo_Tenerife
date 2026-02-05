import gradio as gr
import requests

URL = "https://TU_API_RENDER/predict"

def consumir_api(dia, mes, municipio, distribuidora):

    payload = {
        "dia": dia,
        "mes": mes,
        "cups_municipio": municipio,
        "cups_distribuidor": distribuidora
    }

    r = requests.post(URL, json=payload)
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
