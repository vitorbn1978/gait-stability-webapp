import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Inicializando o Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Função para calcular o COM de um segmento
def calculate_segment_com(landmarks, indices):
    segment_points = np.array([[landmarks[idx].x, landmarks[idx].y, landmarks[idx].z] for idx in indices])
    com = np.mean(segment_points, axis=0)
    return com

# Função para calcular o COM do corpo
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

# Função para calcular a velocidade do CM
def calculate_velocity(cm_positions, frequency):
    velocity = np.diff(cm_positions, axis=0) * frequency
    return np.vstack([velocity, velocity[-1]])  # Para manter o mesmo número de frames

# Função para calcular o XCoM
def calculate_xcom(cm, velocity_cm, lajc, rajc):
    ll = 0.001 * np.linalg.norm(cm - 0.5 * (lajc + rajc))
    wo_r = np.sqrt(9.8 / ll)
    xcom_r = cm + velocity_cm / wo_r
    return xcom_r

# Função para calcular a Margem de Estabilidade (MoS)
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

# Função para calcular a distância euclidiana entre dois pontos 2D (ignorando z)
def calculate_distance_2d(point1, point2):
    return np.linalg.norm(np.array(point1[:2]) - np.array(point2[:2]))

# Listas para armazenar os dados
cm_path_raw = []
cm_path_scaled = []
step_widths_2d = []
mos_values = []

# Estatura e peso do indivíduo
height = 1.63  # em metros
weight = 60.0  # em kg

# Processamento de vídeo
cap = cv2.VideoCapture("nono_com.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frequency = cap.get(cv2.CAP_PROP_FPS)  # Obter a frequência de aquisição (FPS)
print("Frequência de aquisição (FPS):", frequency)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('processed_video.mp4', fourcc, 20.0, (frame_width, frame_height))

frame_count = 0
lajc_positions = []
rajc_positions = []
cm_positions = []

total_step_width = 0  # Soma total da Step Width para cálculo da média
total_mos = 0  # Soma total da MoS para cálculo da média

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        body_com = calculate_body_com(landmarks, height, weight)

        lajc = np.array([landmarks[23].x, landmarks[23].y, landmarks[23].z])  # tornozelo esquerdo
        rajc = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z])  # tornozelo direito

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

        # Coordenadas para desenhar o COM no vídeo
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

    cv2.imshow('Centro de Massa e MoS', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Garantir que todas as listas tenham o mesmo comprimento
min_length = min(len(cm_path_raw), len(cm_path_scaled), len(step_widths_2d), len(mos_values))

# Truncar todas as listas para o comprimento mínimo
cm_path_raw = cm_path_raw[:min_length]
cm_path_scaled = cm_path_scaled[:min_length]
step_widths_2d = step_widths_2d[:min_length]
mos_values = mos_values[:min_length]

# Criar o DataFrame
df_cm = pd.DataFrame({
    'X Bruto (normalizado)': [pos[0] for pos in cm_path_raw],
    'Y Bruto (normalizado)': [pos[1] for pos in cm_path_raw],
    'X Escalonado (metros)': [pos[0] for pos in cm_path_scaled],
    'Y Escalonado (metros)': [pos[1] for pos in cm_path_scaled],
    'Step Width 2D (metros)': step_widths_2d,
    'Margem de Estabilidade (MoS)': mos_values
})

# Salvar o DataFrame em um arquivo CSV
df_cm.to_csv('centro_de_massa_e_step_widths_mos.csv', index=False)

print("Processamento concluído. O vídeo foi salvo como 'processed_video.mp4', e os dados foram salvos em 'centro_de_massa_e_step_widths_mos.csv'.")
