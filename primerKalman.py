from modules import *

# Verificar forma del dataset
print("Dimensión dataset hits: ", hits.shape)

# Filtrar hits del volumen 8
hits_centrales = hits[hits.volume_id == 8]
print(f"Tengo {hits_centrales.module_id.nunique()} módulos y {hits_centrales.layer_id.nunique()} capas.")

# Extraer coordenadas
data = hits_centrales[['x', 'y', 'z']].values
data = np.array(data[:])  # Tomamos solo 300 puntos

# Seleccionar dos hits para la estimación inicial
hit_1 = hits_centrales[hits_centrales.module_id == 1].iloc[0]
hit_2 = hits_centrales[hits_centrales.module_id == 2].iloc[0]

x1, y1, z1 = hit_1[['x', 'y', 'z']]
x2, y2, z2 = hit_2[['x', 'y', 'z']]

dt = 1.0
slope_x, slope_y, slope_z = (x2-x1)/dt, (y2-y1)/dt, (z2-z1)/dt
x0 = np.array([x1, slope_x, y1, slope_y, z1, slope_z]).reshape(-1, 1)

# Definir matrices del filtro de Kalman
F = np.array([[1, dt, 0,  0,  0,  0],
              [0,  1,  0,  0,  0,  0],
              [0,  0,  1, dt,  0,  0],
              [0,  0,  0,  1,  0,  0],
              [0,  0,  0,  0,  1, dt],
              [0,  0,  0,  0,  0,  1]])

H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 0]])

C = np.eye(6) * 1e-3
Q = np.eye(6) * 0.01
R = np.eye(3) * 0.5

kf = KalmanFilter(C=C, F=F, H=H, Q=Q, R=R, x0=x0)

# Inicializar arrays para almacenamiento
x_predic = []
trajectory = []

# Aplicar filtro de Kalman
for i in range(len(data)):
    m = data[i].reshape(-1, 1)  # Asegurar forma correcta
    x_pred = kf.predict()
    x_predic.append(x_pred.flatten())  # Guardar predicción
    kf.update(m)

# Convertir predicciones en array NumPy
x_predic = np.array(x_predic)

# Suavizado RTS

print(f"Cantidad de estados en Kalman: {len(kf.states)}")
print(f"Primer estado: {kf.states[0]}")
print(f"Último estado: {kf.states[-1]}")

smoothed_states, smoothed_covariances = kf.smoothing_RTS()
smoothed_df = pd.DataFrame(smoothed_states[:, [0, 2, 4], 0], columns=["x", "y", "z"])

# Graficar resultados
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Datos originales
ax.plot(data[:, 0], data[:, 1], data[:, 2], 'o', label='Data', markersize=1)

# Predicciones
ax.plot(x_predic[:, 0], x_predic[:, 2], x_predic[:, 4], label='Predicted', color='green')

# Suavizado RTS
ax.plot(smoothed_df["x"], smoothed_df["y"], smoothed_df["z"], label='Smoothed', color='red')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
plt.legend()
#plt.show()

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.plot(smoothed_df['x'], smoothed_df['y'], '-', label='Smoothed', markersize=.6)
# plt.plot(data[:,0], data[:,1], 'o', label='Data', markersize=1)
# plt.xlabel('X (mm)')
# plt.ylabel('Y (mm)')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(smoothed_df['x'], smoothed_df['z'], '-', label='Smoothed', markersize=.6)
# plt.plot(data[:,0], data[:,2], 'o', label='Data', markersize=.6, alpha=0.5)
# plt.xlabel('X (mm)')
# plt.ylabel('Z (mm)')
# plt.legend()

# plt.show()
