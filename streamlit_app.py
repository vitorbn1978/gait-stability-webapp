import streamlit as st
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import tempfile
import antropy as ant
import nolds

# --- Funções auxiliares (copie suas versões exatas se quiser) ---
def calculate_segment_com(landmarks, indices):
    segment_points = np.array([[landmarks[idx].x, landmarks[idx].y, landmarks[idx].z] for idx in indices])
    com = np.mean(segment_points, axis=0)
    return com

def calculate_body_com(landmarks, height, weight):
    SEGMENT_WEIGHTS = {
        'head': 0.081 * weight,
        'torso': 0.497 * weight,
        'left_arm': 0.0265 * weight,
        'right_arm': 0.0265 * weight,
        'left_leg': 0.161 * weight,
        'right_leg': 0.161 * weight
    }
    segments = {
        'head': [0, 1, 2, 3, 4],
        'torso': [11, 12, 23, 24],
        'left_arm': [11, 13, 15],
        'right_arm': [12, 14, 16],
        'left_leg': [23, 25, 27],
        'right_leg': [24, 26, 28]
    }
    weighted_coms = []
    total_weight = 0
    for segment, indices in segments.items():
        segment_com = calculate_segment_com(landmarks, indices)
        weight = SEGMENT_WEIGHTS[segment]
        weighted_coms.append(segment_com * weight)
        total_weight += weight
    overall_com = np.sum(weighted_coms, axis=0) / total_weight
    return overall_com

def calculate_velocity(cm_positions, frequency):
    velocity = np.diff(cm_positions, axis=0) * frequency
    return np.vstack([velocity, velocity[-1]])

def calculate_xcom(cm, velocity_cm, lajc, rajc):
    ll = 0.001 * np.linalg.norm(cm - 0.5 * (lajc + rajc))
    wo_r = np.sqrt(9.8 / ll)
    xcom_r = cm + velocity_cm / wo_r
    return xcom_r

def calculate_mos(cm, xcom, lajc, rajc):
    r1 = lajc
    r2 = rajc
    borda = np.linalg.norm(r2 - r1)
    vet_xcom = np.cross(xcom - r1, r2 - r1)
    dist_xcom = np.linalg.norm(vet_xcom) / borda
    vet_cm = np.cross(cm - r1, r2 - r1)
    dist_cm = np.linalg.norm(vet_cm) / borda
    mos = np.minimum(dist_xcom, dist_cm)
    return mos

def calculate_distance_2d(point1, point2):
    return np.linalg.norm(np.array(point1[:2]) - np.array(point2[:2]))

# --- Streamlit App ---
st.title("Processamento de Vídeo Biomecânica 2D - Medidas e Índices")

uploaded_video = st.file_uploader("Faça upload do vídeo (.mp4)", type=["mp4"])

height = st.number_input("Estatura (m)", value=1.65, step=0.01)
weight = st.number_input("Peso (kg)", value=60.0, step=0.5)

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    st.video(tfile.name)

    if st.button("Processar vídeo"):
        st.info("Processando... pode levar alguns segundos/minutos.")

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        cap = cv2.VideoCapture(tfile.name)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        frequency = cap.get(cv2.CAP_PROP_FPS)
        
        cm_path_raw = []
        cm_path_scaled = []
        step_widths_2d = []
        mos_values = []
        lajc_positions = []
        rajc_positions = []
        cm_positions = []

        frame_count = 0

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
                left_ankle_2d = [landmarks[27].x * height, landmarks[27].y * height]
                right_ankle_2d = [landmarks[28].x * height, landmarks[28].y * height]
                step_width_2d = calculate_distance_2d(left_ankle_2d, right_ankle_2d)
                step_widths_2d.append(step_width_2d)
            else:
                step_widths_2d.append(np.nan)
                cm_path_raw.append((np.nan, np.nan))
                cm_path_scaled.append((np.nan, np.nan))
                lajc_positions.append(np.nan)
                rajc_positions.append(np.nan)
                cm_positions.append(np.nan)

            # Calculando a velocidade do CM e a MoS
            if frame_count > 0 and len(cm_positions) > 1 and type(cm_positions[-1]) is np.ndarray:
                velocity_cm = calculate_velocity(np.array(cm_positions), frequency)
                xcom_r = calculate_xcom(cm_positions[-1], velocity_cm[-1], lajc_positions[-1], rajc_positions[-1])
                mos_r = calculate_mos(cm_positions[-1], xcom_r, lajc_positions[-1], rajc_positions[-1])
                mos_values.append(mos_r)
            else:
                mos_values.append(np.nan)
            frame_count += 1

        cap.release()

        # Ajustar arrays para mesmo tamanho
        min_length = min(len(cm_path_raw), len(cm_path_scaled), len(step_widths_2d), len(mos_values))
        cm_path_raw = cm_path_raw[:min_length]
        cm_path_scaled = cm_path_scaled[:min_length]
        step_widths_2d = step_widths_2d[:min_length]
        mos_values = mos_values[:min_length]

        df_cm = pd.DataFrame({
            'X Bruto (normalizado)': [pos[0] for pos in cm_path_raw],
            'Y Bruto (normalizado)': [pos[1] for pos in cm_path_raw],
            'X Escalonado (metros)': [pos[0] for pos in cm_path_scaled],
            'Y Escalonado (metros)': [pos[1] for pos in cm_path_scaled],
            'Step Width 2D (metros)': step_widths_2d,
            'Margem de Estabilidade (MoS)': mos_values
        })

        st.success("Processamento concluído!")
        st.subheader("Resultados quadro a quadro (mostrando as primeiras linhas):")
        st.write(df_cm.head())

        # Cálculo de índices globais
        y_serie = df_cm['Y Escalonado (metros)'].dropna().values
        mos_validos = df_cm['Margem de Estabilidade (MoS)'].dropna()
        stepwidth_validos = df_cm['Step Width 2D (metros)'].dropna()

        rms = np.sqrt(np.mean(np.square(y_serie)))
        sampen = ant.sample_entropy(y_serie)
        lyap = nolds.lyap_r(y_serie)

        mos_media = mos_validos.mean() * 100
        mos_max = mos_validos.max() * 100
        mos_min = mos_validos.min() * 100

        stepwidth_media = stepwidth_validos.mean() * 100
        stepwidth_max = stepwidth_validos.max() * 100
        stepwidth_min = stepwidth_validos.min() * 100

        indices = {
            'RMS_Y_Escalonado': [rms],
            'SampleEntropy_Y_Escalonado': [sampen],
            'LyapunovExponent_Y_Escalonado': [lyap],
            'MoS_media_cm': [mos_media],
            'MoS_max_cm': [mos_max],
            'MoS_min_cm': [mos_min],
            'StepWidth_media_cm': [stepwidth_media],
            'StepWidth_max_cm': [stepwidth_max],
            'StepWidth_min_cm': [stepwidth_min]
        }
        df_indices = pd.DataFrame(indices)

        st.subheader("Índices globais extraídos:")
        st.table(df_indices)

        # Download dos arquivos
        st.download_button(
            "Baixar resultados quadro a quadro (.csv)",
            df_cm.to_csv(index=False).encode("utf-8"),
            file_name="resultados_marcha.csv",
            mime="text/csv"
        )
        st.download_button(
            "Baixar índices globais (.csv)",
            df_indices.to_csv(index=False).encode("utf-8"),
            file_name="indices_globais.csv",
            mime="text/csv"
        )

else:
    st.info("Faça upload de um vídeo para começar.")
