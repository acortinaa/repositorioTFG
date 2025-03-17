'''
Script: Kalman Filter
'''
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

    def predict(self):
        ''' Predicción del siguiente estado '''
        self.x = self.F @ self.x  # Predicción del siguiente estado
        self.C = self.F @ self.C @ self.F.T + self.Q  # Predicción de la covarianza del error
        self.x = self.x.reshape(-1,1)
        return self.x
    
    def update(self, m):    # m: medición
        ''' Actualización del estado con la medición m '''
        m = m.reshape(-1,1)
        r = (m - self.H @ self.x) # Diferencia entre la medición y la proyección del estado
        S = self.H @ self.C @ self.H.T + self.R  # Covarianza del nuevo valor (con el ruido de medición)
        K = self.C @ self.H.T @ np.linalg.inv(S)  # Ganancia de Kalman (cuánto confiar en la medición)
        
        print(f"Dimensión de m: {m.shape}")
        print(f"Dimensión de r: {r.shape}")  
        print(f"Dimensión de K: {K.shape}")  
        print(f"Dimensión de self.x antes de actualizar: {self.x.shape}")  

        self.x = self.x + K @ r  # Estado actualizado
        self.C = (np.eye(self.C.shape[0]) - K @ self.H) @ self.C  # Covarianza del error actualizada
        return self.x
    
'''
Cálculo de la trayectoria por pares de hits
'''
