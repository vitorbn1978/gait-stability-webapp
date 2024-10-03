import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from io import BytesIO

# Funções para processamento
def process_video(video_path, estatura, peso):
    # Inicializando o Mediapipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    cm_path_raw = []
    cm_path_scaled = []
    step_widths_2d = []
    mos_values = []

    # Abrindo o vídeo
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frequency = cap.get(cv2.CAP_PROP_FPS)

    # Verificando se o vídeo foi aberto corretamente
    if not cap.isOpened():
        st.error("Erro ao abrir o vídeo. Verifique o formato e tente novamente.")
        return None

    # Configurando o vídeo de saída
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('processed_video.mp4', fourcc, 20.0, (frame_width, frame_height))

    frame_count = 0
    lajc_positions = []
    rajc_positions = []
    cm_positions = []
    total_step_width = 0
    total_mos = 0

    # Processando frame por frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            body_com = calculate_body_com(landmarks, estatura, peso)

            if body_com is None or len(body_com) < 3:
                continue  # Ignora frames com erro no cálculo do COM

            lajc = np.array([landmarks[23].x, landmarks[23].y])
            rajc = np.array([landmarks[24].x, landmarks[24].y])

            cm_positions.append(body_com)
            lajc_positions.append(lajc)
            rajc_positions.append(rajc)

            cm_x_raw = body_com[0]
            cm_y_raw = body_com[1]
            cm_x_scaled = cm_x_raw * estatura
            cm_y_scaled = cm_y_raw * estatura

            cm_path_raw.append((cm_x_raw, cm_y_raw))
            cm_path_scaled.append((cm_x_scaled, cm_y_scaled))

            # Step Width Calculation
            step_width_2d = calculate_distance_2d(lajc, rajc)
            step_widths_2d.append(step_width_2d)
            total_step_width += step_width_2d
            average_step_width = total_step_width / len(step_widths_2d)

            # Desenhando o centro de massa no vídeo
            cm_x_pixel = int(cm_x_raw * frame_width)
            cm_y_pixel = int(cm_y_raw * frame_height)
            cv2.circle(frame, (cm_x_pixel, cm_y_pixel), 5, (0, 0, 255), -1)
            cv2.putText(frame, "CM", (cm_x_pixel, cm_y_pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Salvando o frame processado
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # Verificando se o processamento foi bem-sucedido
    if frame_count == 0:
        st.error("Nenhum frame foi processado. Verifique se o vídeo possui conteúdo válido.")
        return None

    # Salvando os dados em CSV
    df_cm = pd.DataFrame({
        'X Bruto (normalizado)': [pos[0] for pos in cm_path_raw],
        'Y Bruto (normalizado)': [pos[1] for pos in cm_path_raw],
        'X Escalonado (metros)': [pos[0] for pos in cm_path_scaled],
        'Y Escalonado (metros)': [pos[1] for pos in cm_path_scaled],
        'Step Width 2D (metros)': step_widths_2d
    })
    
    df_cm.to_csv('centro_de_massa_e_step_widths_mos.csv', index=False)

    return 'processed_video.mp4'

# Funções auxiliares permanecem as mesmas...

# Interface do Streamlit
st.title("Gait Stability COM Analysis Web App")
st.write("Este aplicativo calcula a movimentação do centro de massa (COM), a margem de estabilidade (MoS) e o step width, baseando-se em medidas 2D a partir do vídeo enviado.")
st.write("Contato: Dr. Vitor Bertoli Nascimento, bertolinascimento.vitor@gmail.com")

# Entrada de dados
estatura = st.number_input("Estatura (em metros)", min_value=0.5, max_value=2.5, value=1.68)
peso = st.number_input("Peso (em kg)", min_value=30.0, max_value=150.0, value=70.0)

# Upload de vídeo
uploaded_file = st.file_uploader("Escolha um vídeo", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    video_path = 'input_video.mp4'
    with open(video_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Processando o vídeo
    st.write(f"Processando o vídeo para estatura: {estatura} m e peso: {peso} kg...")
    processed_video_path = process_video(video_path, estatura, peso)

    # Verificando se o vídeo foi processado com sucesso
    if processed_video_path:
        # Exibindo o vídeo processado
        st.video(processed_video_path)

        # Oferecendo o download do vídeo processado
        with open(processed_video_path, "rb") as video_file:
            st.download_button(label="Baixar Vídeo Processado", data=video_file, file_name="processed_video.mp4", mime="video/mp4")

        # Oferecendo o download do CSV com os resultados
        with open("centro_de_massa_e_step_widths_mos.csv", "rb") as f:
            csv_data = f.read()
        st.download_button(label="Baixar CSV com Resultados", data=csv_data, file_name="centro_de_massa_e_step_widths_mos.csv", mime="text/csv")


