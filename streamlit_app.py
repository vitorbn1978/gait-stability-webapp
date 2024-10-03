import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from io import BytesIO

# Funções para processamento
def process_video(video_path):
    # Inicializando o Mediapipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    cm_path_raw = []
    cm_path_scaled = []
    step_widths_2d = []
    mos_values = []

    # Estatura e peso do indivíduo
    height = 1.68  # em metros
    weight = 70.0  # em kg

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frequency = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('processed_video.mp4', fourcc, 20.0, (frame_width, frame_height))

    frame_count = 0
    lajc_positions = []
    rajc_positions = []
    cm_positions = []
    total_step_width = 0
    total_mos = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            body_com = calculate_body_com(landmarks, height, weight)

            lajc = np.array([landmarks[23].x, landmarks[23].y, landmarks[23].z])
            rajc = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z])

            cm_positions.append(body_com)
            lajc_positions.append(lajc)
            rajc_positions.append(rajc)

            cm_x_raw = body_com[0]
            cm_y_raw = body_com[1]
            cm_x_scaled = cm_x_raw * height
            cm_y_scaled = cm_y_raw * height

            cm_path_raw.append((cm_x_raw, cm_y_raw))
            cm_path_scaled.append((cm_x_scaled, cm_y_scaled))

            # Cálculo da Step Width em 2D (ignorando z)
            left_ankle_2d = [landmarks[27].x * height, landmarks[27].y * height]
            right_ankle_2d = [landmarks[28].x * height, landmarks[28].y * height]
            step_width_2d = calculate_distance_2d(left_ankle_2d, right_ankle_2d)
            step_widths_2d.append(step_width_2d)

            total_step_width += step_width_2d
            average_step_width = total_step_width / len(step_widths_2d)

            cm_x_pixel = int(cm_x_raw * frame_width)
            cm_y_pixel = int(cm_y_raw * frame_height)

            cv2.circle(frame, (cm_x_pixel, cm_y_pixel), 5, (0, 0, 255), -1)
            cv2.putText(frame, "CM", (cm_x_pixel, cm_y_pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Calculando a velocidade do CM e a MoS
        if frame_count > 0:
            velocity_cm = calculate_velocity(cm_positions, frequency)
            xcom_r = calculate_xcom(cm_positions[-1], velocity_cm[-1], lajc_positions[-1], rajc_positions[-1])
            mos_r = calculate_mos(cm_positions[-1], xcom_r, lajc_positions[-1], rajc_positions[-1])
            mos_values.append(mos_r)
            total_mos += mos_r

            # Exibir os valores de MoS e média da Step Width no vídeo
            cv2.putText(frame, f"MoS: {mos_r:.2f} m", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Avg Step Width: {average_step_width:.2f} m", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # Salvando os dados em CSV (opcional para download)
    df_cm = pd.DataFrame({
        'X Bruto (normalizado)': [pos[0] for pos in cm_path_raw],
        'Y Bruto (normalizado)': [pos[1] for pos in cm_path_raw],
        'X Escalonado (metros)': [pos[0] for pos in cm_path_scaled],
        'Y Escalonado (metros)': [pos[1] for pos in cm_path_scaled],
        'Step Width 2D (metros)': step_widths_2d,
        'Margem de Estabilidade (MoS)': mos_values
    })
    
    df_cm.to_csv('centro_de_massa_e_step_widths_mos.csv', index=False)

    return 'processed_video.mp4'

# Funções auxiliares
def calculate_body_com(landmarks, height, weight):
    # Função do COM aqui
    pass

def calculate_velocity(cm_positions, frequency):
    # Função de cálculo da velocidade aqui
    pass

def calculate_xcom(cm, velocity_cm, lajc, rajc):
    # Função do XCoM aqui
    pass

def calculate_mos(cm, xcom, lajc, rajc):
    # Função do MoS aqui
    pass

def calculate_distance_2d(point1, point2):
    # Função de cálculo da distância 2D aqui
    pass

# Streamlit UI
st.title("Gait Stability COM Analysis Web App")
st.write("Faça o upload de um vídeo para realizar a análise.")

uploaded_file = st.file_uploader("Escolha um vídeo", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Salvar o arquivo temporariamente
    video_path = 'input_video.mp4'
    with open(video_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Processar o vídeo
    st.write("Processando o vídeo...")
    processed_video_path = process_video(video_path)

    # Exibir o vídeo processado
    st.video(processed_video_path)

    # Oferecer o download do CSV
    with open("centro_de_massa_e_step_widths_mos.csv", "rb") as f:
        csv_data = f.read()
    st.download_button(label="Baixar CSV com Resultados", data=csv_data, file_name="centro_de_massa_e_step_widths_mos.csv", mime="text/csv")
