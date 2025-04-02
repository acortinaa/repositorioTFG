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