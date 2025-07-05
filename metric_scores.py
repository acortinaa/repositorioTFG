# Función de scoring con double majority rule
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from trackml.randomize import shuffle_hits
from trackml.score import score_event
from scipy.spatial import cKDTree


def valid_tracks_from_best_hits(hits_vecinos_por_track, truth_hits, min_ratio, threshold=5.0):
    """
    Evalúa tracks a partir de sus best_hits. Verifica si suficientes best_hits
    están cerca de hits truth con el mismo particle_id.

    Args:
        hits_vecinos_por_track: lista de lista de tuplas (best_hit, vecinos)
        truth_hits: DataFrame con columnas ['tx', 'ty', 'tz', 'particle_id']
        threshold: distancia máxima para considerar un hit como match
        min_ratio: proporción mínima de hits que deben tener el mismo particle_id

    Returns:
        valid_count: número de tracks válidos
        valid_track_indices: conjunto de índices de tracks válidos
        track_to_pid: mapeo de índice de track a particle_id del truth
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(truth_hits[['tx', 'ty', 'tz']].values)
    truth_particles = truth_hits['particle_id'].values
    valid_count = 0
    valid_track_indices = set()
    track_to_pid = {}

    for idx, track_hits in enumerate(hits_vecinos_por_track):
        best_hits = np.array([hit for hit, _ in track_hits])#[3:]])
        
        if len(best_hits) == 0:
            continue

        dists, indices = tree.query(best_hits, distance_upper_bound=threshold)
        valid_idx = indices < len(truth_particles)
        matched_particles = truth_particles[indices[valid_idx]]

        if len(matched_particles) == 0:
            continue

        particle_counts = pd.Series(matched_particles).value_counts()
        best_pid = particle_counts.index[0]
        best_pid_count = particle_counts.iloc[0]
        ratio = best_pid_count / len(best_hits)

        if ratio >= min_ratio:
            valid_count += 1
            valid_track_indices.add(idx)
            track_to_pid[idx] = best_pid

    return valid_count, valid_track_indices, track_to_pid



def valid_tracks_from_kalman_tracks(tracks, pids_por_track, truth_hits, min_ratio=0.5, threshold=5.0):
    from scipy.spatial import cKDTree
    valid_count = 0
    valid_track_indices = set()

    for idx, track in enumerate(tracks):
        pid = pids_por_track.get(idx, -1)
        if pid == -1:
            continue
        truth_pid_hits = truth_hits[truth_hits['particle_id'] == pid][['tx', 'ty', 'tz']].values
        if len(truth_pid_hits) == 0:
            continue
        tree = cKDTree(truth_pid_hits)
        dists, _ = tree.query(track, distance_upper_bound=threshold)
        matched = dists < threshold
        ratio = np.sum(matched) / len(track)
        if ratio >= min_ratio:
            valid_count += 1
            valid_track_indices.add(idx)

    return valid_count, valid_track_indices, pids_por_track

def valid_tracks_from_kalman_tracks_no_pid(tracks, truth_hits, min_ratio=0.5, threshold=5.0):
    from scipy.spatial import cKDTree
    valid_count = 0
    valid_track_indices = set()

    tree = cKDTree(truth_hits[['tx', 'ty', 'tz']].values)

    for idx, track in enumerate(tracks):
        dists, _ = tree.query(track, distance_upper_bound=threshold)
        matched = dists < threshold
        ratio = np.sum(matched) / len(track)
        if ratio >= min_ratio:
            valid_count += 1
            valid_track_indices.add(idx)

    return valid_count, valid_track_indices






def score_with_double_majority(kalman_tracks, truth_tracks, truth_weights, threshold=10.0):
    total_score = 0.0

    for k_track in kalman_tracks:
        match_counts = []

        # Comparar el track de Kalman con las partículas
        for idx, t_track in enumerate(truth_tracks):
            # Calcular la distancia entre el track de Kalman y la partícula
            distances = cdist(k_track, t_track)
            min_dists = np.min(distances, axis=1)
            match = min_dists < threshold
            match_counts.append((idx, np.sum(match)))  # Contamos los hits coincidentes

        # Ordenar por la cantidad de coincidencias y asegurarse de que haya mayoría
        match_counts.sort(key=lambda x: x[1], reverse=True)
        if not match_counts or match_counts[0][1] <= len(k_track) // 2:
            continue  # Si no hay mayoría de puntos, descartamos el track

        best_truth_idx, _ = match_counts[0]  # La mejor coincidencia de partícula
        t_track = truth_tracks[best_truth_idx]
        t_weights = truth_weights[best_truth_idx]

        # Verificar si más de la mitad de los puntos de la partícula están en el track
        distances = cdist(t_track, k_track)
        min_dists = np.min(distances, axis=1)
        match_truth = min_dists < threshold
        if np.sum(match_truth) <= len(t_track) // 2:
            continue  # Si no hay mayoría de la partícula en el track, descartamos

        # Calcular la puntuación: suma de los pesos de los hits de intersección
        score = np.sum(t_weights[match_truth])
        total_score += score

    return total_score