import numpy as np

class KalmanFilter:
    def __init__(self, C, F, H, Q, R, x0):
        '''
        C: matriz de covarianza inicial
        F: matriz de propagación de parámetros del camino
        H: matriz de proyección
        Q: matriz de error adicional por ruido de proceso
        R: matriz de covarianza de ruido de medición
        '''
        self.C = C  # Covarianza del error inicial
        self.F = F  # Propagación del modelo del sistema (matriz de propagación)
        self.H = H  # Matriz de observación (proyección de las mediciones)
        self.Q = Q  # Matriz de covarianza del ruido del proceso
        self.R = R  # Matriz de covarianza del ruido de medición
        self.x = x0  # Estimación inicial del estado

        self.states = []  # Almacenar los estados
        self.gains = []   # Almacenar las ganancias de Kalman
        self.positions = []  # Almacenar solo las posiciones (x, y, z) para ploteo

    def predict(self):
        ''' Predicción del siguiente estado '''
        self.x = self.F @ self.x  # Predicción del siguiente estado
        self.C = self.F @ self.C @ self.F.T + self.Q  # Predicción de la covarianza del error
        self.x = self.x.reshape(-1, 1)
        return self.x
    
    def update(self, m):    # m: medición
        ''' Actualización del estado con la medición m'''
        m = m.reshape(-1, 1)
        r = (m - self.H @ self.x)  # Diferencia entre la medición y la proyección del estado
        S = self.H @ self.C @ self.H.T + self.R  # Covarianza del nuevo valor (con el ruido de medición)
        K = self.C @ self.H.T @ np.linalg.inv(S)  # Ganancia de Kalman (cuánto confiar en la medición)

        self.x = self.x + K @ r  # Estado actualizado
        self.C = (np.eye(self.C.shape[0]) - K @ self.H) @ self.C  # Covarianza del error actualizada
        
        # Almacenar el estado completo (6 dimensiones)
        self.states.append(self.x.copy())  # Guarda todo el estado (6 dimensiones)
        
        self.gains.append(K.copy())
        
        return self.x, K
    
    def smoothing_RTS(self):
        '''
        Proceso de iterativo de suavizado (RTS) desde el final. Se opera solo con (x, y, z) de los estados.
        '''
        n = len(self.states)  # Número de pasos
        #print('Número de pasos n = ', n)
        smoothed_states = np.zeros_like(self.states)

        # El último estado es igual al estado filtrado (no necesita ser modificado)
        smoothed_states[-1] = self.states[-1]

        # Iterar hacia atrás desde el penúltimo estado hasta el primero
        for t in range(n - 2, -1, -1):
            K = self.gains[t]
            F = self.F
            # Realizamos el suavizado sobre el estado completo, pero actualizamos solo las posiciones
            smoothed_states[t] = self.states[t] + K @ (smoothed_states[t + 1] - F @ self.states[t])[[0, 2, 4], :]  # Solo x, y, z
        
        smoothed_positions = smoothed_states[:, [0, 2, 4]]
        if smoothed_positions.shape[0] % 50 == 0:
            print(f"Dimensión de smoothed_positions: {smoothed_positions.shape}")
        
        return smoothed_states, smoothed_positions
    

def apply_kalmanfilter(hit1, hit2, hits_dict_all_volumes, Q_COEFF, 
                       get_initial_state, C, F, H, Q, R, volume_ids, campo_magnetico, 
                       apply_lorentz_correction, SMOOTHING, charge_sign, HITS_CERCANOS=True):
    x0 = get_initial_state(hit1, hit2)
    kf = KalmanFilter(C=C, F=F, H=H, Q=Q, R=R, x0=x0)
    pred_trajectory = []
    hits_vecinos_por_track = []  # Aquí vamos a almacenar los hits vecinos por trayectoria
    total_residual = 0.0

    # Iteramos por cada volumen y capa
    for volume_id in volume_ids:
        for layer in sorted(hits_dict_all_volumes[volume_id].keys()):
            hits_layer = hits_dict_all_volumes[volume_id][layer][['x', 'y', 'z']].values
            kf.predict()
            pred_pos = kf.x[[0, 2, 4]].flatten()
            pred_trajectory.append(pred_pos)

            Q_OVER_M = charge_sign * Q_COEFF
            B = campo_magnetico(pred_pos[2])
            apply_lorentz_correction(kf, B, Q_OVER_M)

            # Calculamos las distancias entre el hit actual y todos los hits en la capa
            distances = np.linalg.norm(hits_layer - pred_pos, axis=1)
            best_hit = hits_layer[np.argmin(distances)]  # El hit más cercano
            kf.update(best_hit)

            # Chi²: suma de residuos al cuadrado
            residual = best_hit - kf.x[[0, 2, 4]].flatten()
            total_residual += np.sum(residual**2)


            if HITS_CERCANOS:
                closest_idxs = np.argsort(distances)[:5]  # Los 5 más cercanos
                vecinos = hits_layer[closest_idxs[1:]]  # Excluye el mejor hit
                hits_vecinos_por_track.append((best_hit, vecinos))  # Guardamos el best_hit y los vecinos

    # Devolvemos las trayectorias predichas y los hits vecinos si es necesario
    if SMOOTHING:
        _, smoothed = kf.smoothing_RTS()
        return smoothed.squeeze(), hits_vecinos_por_track, total_residual
    else:
        return np.array(pred_trajectory), hits_vecinos_por_track, total_residual
    

### FUNCIONES AUXILIARES ###
def cos_angle(v1, v2):
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    return np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)

REDUCTION_FRACTION = 1.0
def reduce_hits(hits, fraction=REDUCTION_FRACTION):
    return hits.sample(frac=fraction, random_state=42)

def campo_magnetico(z):
    z = z / 2750
    return 0.03 * z**3 - (0.55 - 0.3 * (1 - z**2)) * z**2 + 1.002

OCTANTE = False
def get_hits_dict(hits, volume_ids, OCTANTE = OCTANTE, angle_range = None, p_range = None):
    '''
    Crea un diccionario de hits por volumen y capa, filtrando por el primer octante y/o rango de ángulo y p.
    hits: DataFrame con las columnas ['volume_id', 'layer_id', 'x', 'y', 'z']
    volume_ids: lista de IDs de volúmenes a considerar
    OCTANTE: si True, filtra por el primer octante (x, y, z > 0)
    angle_range: tupla (min, max) en grados para filtrar por ángulo theta
    p_range: tupla (min, max) para filtrar por p (momento) en GeV/c
    '''
    hits_dict_all = {}

    for volume_id in volume_ids:
        hits_volume = hits[hits.volume_id == volume_id].copy()
        hits_volume['r'] = np.sqrt(hits_volume.x**2 + hits_volume.y**2 + hits_volume.z**2)

        # Filtrar por el primer octante si OCTANTE es True
        if OCTANTE:
            hits_volume = hits_volume[(hits_volume['x'] > 0) & (hits_volume['y'] > 0) & (hits_volume['z'] > 0)]

        if angle_range is not None:
            # Calculamos theta en grados
            hits_volume['theta'] = np.degrees(np.arccos(hits_volume['z'] / hits_volume['r']))
            # Filtrar por rango de theta (en grados)
            hits_volume = hits_volume[(hits_volume['theta'] >= angle_range[0]) & (hits_volume['theta'] <= angle_range[1])]

        if p_range is not None:
            print(f"Rango de p: {p_range} GeV/c")
            # Calculamos p
            #hits_volume['p'] = np.sqrt(hits_volume['px']**2 + hits_volume['py']**2 + hits_volume['pz']**2)
            # Filtrar por rango de p
            hits_volume_all = hits_volume
            hits_volume = hits_volume[(hits_volume['p'] >= p_range[0]) & (hits_volume['p'] <= p_range[1])]
            print(f"Volumen {volume_id} tiene {len(hits_volume)} hits después del filtrado por p")
            print(f"% de hits conservados: {len(hits_volume) / len(hits_volume_all) * 100:.2f}%\n")


        # Crear el diccionario de hits por capa
        hits_dict = {
            layer: reduce_hits(hits_volume[hits_volume.layer_id == layer])
            for layer in hits_volume['layer_id'].unique()
        }

        hits_dict_all[volume_id] = hits_dict

    return hits_dict_all


def get_initial_state(hit1, hit2):
    # Asumimos que el vertex está en el origen (0, 0, 0)
    v = hit2 - hit1
    v_unit = v / np.linalg.norm(v)
    velocity = np.linalg.norm(v)
    return np.array([
        hit1[0], v_unit[0]*velocity,
        hit1[1], v_unit[1]*velocity,
        hit1[2], v_unit[2]*velocity
    ]).reshape(-1, 1)

DT = 1
def apply_lorentz_correction(kf, Bz, Q_OVER_M):
    velocity = np.array([kf.x[1, 0], kf.x[3, 0], kf.x[5, 0]])
    force = Q_OVER_M * np.cross(velocity, np.array([0, 0, 3 * Bz]))
    new_velocity = velocity + DT*force
    kf.x[1], kf.x[3], kf.x[5] = new_velocity


HITS_CERCANOS = True  # Cambiar a False si no se quieren mostrar los hits cercanos
import matplotlib.pyplot as plt
import random
import numpy as np
# ======== VISUALIZACIONES ========
def visualizar_3D_hits_y_tracks(volume_ids, hits_dict_all_volumes, tracks, hits_vecinos_por_track, TRACKS_TO_DRAW=30, VISUALIZAR=True):
    fig = plt.figure(figsize=(12, 8))

    # Subplot 3D
    ax = fig.add_subplot(111, projection='3d')

    if not HITS_CERCANOS:
        for volume_id in volume_ids:
            for layer in hits_dict_all_volumes[volume_id].keys():
                hits_layer = hits_dict_all_volumes[volume_id][layer]
                ax.scatter(hits_layer['x'], hits_layer['y'], hits_layer['z'], s=3, alpha=0.5)

    # Muestras aleatorias de las trayectorias
    muestras = random.sample(list(enumerate(tracks)), min(len(tracks), TRACKS_TO_DRAW))
    for i, (idx, track) in enumerate(muestras):
        ax.plot(track[:, 0], track[:, 1], track[:, 2], color='tab:red', alpha=0.8)

        if HITS_CERCANOS and idx < len(hits_vecinos_por_track):
            for best_hit, vecinos in hits_vecinos_por_track[idx]:
                # Plot best hit en azul
                ax.scatter(best_hit[0], best_hit[1], best_hit[2],
                           color='blue', s=25, alpha=0.9,
                           label='Best hit' if i == 0 else "")
                # Plot vecinos en verde (esto no se visualizará en la proyección XY)
                ax.scatter(vecinos[:, 0], vecinos[:, 1], vecinos[:, 2],
                           color='limegreen', s=20, alpha=0.3)

    ax.set_title('Trayectorias suavizadas con Kalman RTS')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Proyección XY 2D
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    
    # Mostrar proyección XY de las trayectorias sin los vecinos verdes
    for i, (idx, track) in enumerate(muestras):
        x, y, z = track[:, 0], track[:, 1], track[:, 2]
        ax2.plot(x, y, alpha=0.6)

        # Marcar todos los puntos del track
        for j in range(len(x)):
            ax2.scatter(x[j], y[j], s=15, color='blue', alpha=0.5)  # Marca cada hit considerado

        # Marcar los best_hits específicos de cada track con un color único
        if HITS_CERCANOS and idx < len(hits_vecinos_por_track):
            track_color = f"C{i}"  # Usar el mismo color por track (basado en el índice)
            vecinos_del_track = hits_vecinos_por_track[idx]
            for best_hit, vecinos in vecinos_del_track:
                ax2.scatter(best_hit[0], best_hit[1], s=20, color=track_color, alpha=0.6, label=f'Best hit Track {i}' if i == 0 else "")

    ax2.set_title('Proyección XY de las trayectorias')
    ax2.axis('equal')  # Para mantener la proporción de los ejes X e Y
    plt.tight_layout()

    if VISUALIZAR:
        plt.show()