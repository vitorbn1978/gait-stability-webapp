import streamlit as st
import pandas as pd
import numpy as np
import antropy as ant
import nolds
import matplotlib.pyplot as plt

def calcular_indices(df):
    y_serie = df['Y Escalonado (metros)'].dropna().values
    mos_validos = df['Margem de Estabilidade (MoS)'].dropna()
    stepwidth_validos = df['Step Width 2D (metros)'].dropna()

    rms = np.sqrt(np.mean(np.square(y_serie)))
    sampen = ant.sample_entropy(y_serie)
    lyap = nolds.lyap_r(y_serie)

    mos_media = mos_validos.mean() * 100
    mos_max = mos_validos.max() * 100
    mos_min = mos_validos.min() * 100

    stepwidth_media = stepwidth_validos.mean() * 100
    stepwidth_max = stepwidth_validos.max() * 100
    stepwidth_min = stepwidth_validos.min() * 100

    return {
        'RMS_Y_Escalonado': [rms],
        'SampleEntropy_Y_Escalonado': [sampen],
        'LyapunovExponent_Y_Escalonado': [lyap],
        'MoS_média_cm': [mos_media],
        'MoS_máx_cm': [mos_max],
        'MoS_mín_cm': [mos_min],
        'StepWidth_média_cm': [stepwidth_media],
        'StepWidth_máx_cm': [stepwidth_max],
        'StepWidth_mín_cm': [stepwidth_min]
    }

st.title("Biomecânica da Marcha 2D – Estabilidade")

uploaded_file = st.file_uploader("Envie o arquivo .csv exportado", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    indices = calcular_indices(df)
    df_indices = pd.DataFrame(indices)

    st.subheader("Resultados dos índices (em cm):")
    st.table(df_indices)  # Exibe como tabela interativa

    # Download dos índices
    st.download_button(
        label="Baixar índices calculados (.csv)",
        data=df_indices.to_csv(index=False).encode('utf-8'),
        file_name="indices_calculados.csv",
        mime="text/csv"
    )

    st.subheader("Gráfico – MoS e Step Width (cm)")
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    ax[0].bar(['Média', 'Mín', 'Máx'], [indices['MoS_média_cm'][0], indices['MoS_mín_cm'][0], indices['MoS_máx_cm'][0]])
    ax[0].set_title("MoS (cm)")
    ax[1].bar(['Média', 'Mín', 'Máx'], [indices['StepWidth_média_cm'][0], indices['StepWidth_mín_cm'][0], indices['StepWidth_máx_cm'][0]])
    ax[1].set_title("Step Width (cm)")
    st.pyplot(fig)

    st.subheader("Série Temporal do Centro de Massa (Y)")
    st.line_chart(df['Y Escalonado (metros)'])

    st.subheader("Série Temporal da MoS (cm)")
    st.line_chart(df['Margem de Estabilidade (MoS)']*100)
else:
    st.info("Faça o upload de um arquivo para começar.")

st.markdown("---")
st.markdown("Biomecânica da Corrida 2D - Medidas Angulares | Streamlit")

