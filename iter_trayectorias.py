import numpy as np
from top_momentum_module import top_momentum
from kalman_filter import get_hits_dict, apply_kalmanfilter, get_initial_state, cos_angle, campo_magnetico, apply_lorentz_correction


def track_finding(hits, truth, top_particles, pt_min, pt_max,
                                    DT=1.0, COS_THRESHOLD=0, Q_COEFF_BASE=0.01, 
                                    SMOOTHING=True, OCTANTE=False):

    if (top_particles['pt'] > 3).any():
        Q_COEFF = 0.01
    else:
        Q_COEFF = Q_COEFF_BASE

    volume_ids = hits['volume_id'].unique()
    volume_ids = volume_ids[[1, 4, 7]]  # Selección específica de volúmenes
    first_volume = volume_ids[0]

    hits_dict_all_volumes = get_hits_dict(hits, volume_ids, OCTANTE=OCTANTE)

    first_layer = sorted(hits_dict_all_volumes[first_volume].keys())[0]
    second_layer = sorted(hits_dict_all_volumes[first_volume].keys())[1]
    third_layer = sorted(hits_dict_all_volumes[first_volume].keys())[2]
    vertex = np.array([0, 0, 0])

    # Matrices de Kalman
    F = np.array([[1, DT, 0,  0,  0,  0],
                [0,  1,  0,  0,  0,  0],
                [0,  0,  1, DT,  0,  0],
                [0,  0,  0,  1,  0,  0],
                [0,  0,  0,  0,  1, DT],
                [0,  0,  0,  0,  0,  1]])

    H = np.array([[1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0]])

# Es posible que los coeficientes tengan que cambiarse al pasar de tratar intervalos al evento completo
    C = np.eye(6) * 1e-1    #1e-1 mejor candidato
    Q = np.eye(6) * 1e-2    #1e-2 mejor candidato
    R = np.eye(3) * 1e-2    #1e-2 mejor candidato

    tracks, hits_vecinos_por_track, cos_values = [], [], []
    positivos, negativos = 0, 0
    tripletes_correctos = 0
    total_tripletes = 0


    for i in range(len(hits_dict_all_volumes[first_volume][first_layer])):
        # Hit 1
        hit1_row = hits_dict_all_volumes[first_volume][first_layer].iloc[i]
        hit1 = hit1_row[['x', 'y', 'z']].values
        v1 = hit1 - vertex

        # Hit 2
        hits_second_df = hits_dict_all_volumes[first_volume][second_layer]
        hits_second = hits_second_df[['x', 'y', 'z']].values
        hit2_idx = np.argmin(np.linalg.norm(hits_second - hit1, axis=1))
        hit2_row = hits_second_df.iloc[hit2_idx]
        hit2 = hit2_row[['x', 'y', 'z']].values
        v2 = hit2 - vertex

        if cos_angle(v1, v2) < COS_THRESHOLD:
            continue

        # Hit 3
        hits_third_df = hits_dict_all_volumes[first_volume][third_layer]
        hits_third = hits_third_df[['x', 'y', 'z']].values
        hit3_idx = np.argmin(np.linalg.norm(hits_third - hit2, axis=1))
        hit3_row = hits_third_df.iloc[hit3_idx]
        hit3 = hit3_row[['x', 'y', 'z']].values
        v3 = hit3 - vertex

        if cos_angle(v2, v3) < COS_THRESHOLD:
            continue

        cos_values.append(cos_angle(v1, v2))

        # Contar total de tripletes
        total_tripletes += 1

        # Obtener hit_ids
        hit1_id = hit1_row['hit_id']
        hit2_id = hit2_row['hit_id']
        hit3_id = hit3_row['hit_id']

        # Obtener particle_ids asociados
        pid1 = truth[truth['hit_id'] == hit1_id]['particle_id'].values[0]
        pid2 = truth[truth['hit_id'] == hit2_id]['particle_id'].values[0]
        pid3 = truth[truth['hit_id'] == hit3_id]['particle_id'].values[0]

        if pid1 == pid2 == pid3:
            tripletes_correctos += 1

        # Ejecutar filtro de Kalman
        track_posit, hits_vec_posit, chi2_posit = apply_kalmanfilter(
            hit1, hit2, hit3, hits_dict_all_volumes, Q_COEFF,
            get_initial_state, C, F, H, Q, R, volume_ids,
            campo_magnetico, apply_lorentz_correction, SMOOTHING, +1)

        track_negat, hits_vec_negat, chi2_negat = apply_kalmanfilter(
            hit1, hit2, hit3, hits_dict_all_volumes, Q_COEFF,
            get_initial_state, C, F, H, Q, R, volume_ids,
            campo_magnetico, apply_lorentz_correction, SMOOTHING, -1)

        if chi2_posit < chi2_negat:
            tracks.append(track_posit)
            hits_vecinos_por_track.append(hits_vec_posit)
            positivos += 1
        else:
            tracks.append(track_negat)
            hits_vecinos_por_track.append(hits_vec_negat)
            negativos += 1

    triplet_precision = 100 * tripletes_correctos / total_tripletes if total_tripletes > 0 else 0.0
    print(f"Rango pT: {pt_min:.2f} - {pt_max:.2f} GeV/c")
    print(f"Total tracks: {len(tracks)}, Positivos: {positivos}, Negativos: {negativos}")
    print(f"\nTripletes con mismo particle_id: {tripletes_correctos}/{total_tripletes} "
          f"({100 * tripletes_correctos / total_tripletes:.2f}%)")

    return tracks, hits_vecinos_por_track, top_particles, volume_ids, hits_dict_all_volumes, truth, triplet_precision


### TRACKFINDING WITH ML
def track_finding_from_triplet_coords(X_triplets, hits, truth, top_particles,
                                      volume_ids, hits_dict_all_volumes, 
                                      DT=1.0, COS_THRESHOLD=None,
                                      Q_COEFF_BASE=0.01, SMOOTHING=True):

    import numpy as np

    # Kalman matrices
    F = np.array([[1, DT, 0,  0,  0,  0],
                  [0,  1,  0,  0,  0,  0],
                  [0,  0,  1, DT,  0,  0],
                  [0,  0,  0,  1,  0,  0],
                  [0,  0,  0,  0,  1, DT],
                  [0,  0,  0,  0,  0,  1]])

    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0]])

    C = np.eye(6) * 1e-1
    Q = np.eye(6) * 1e-2
    R = np.eye(3) * 1e-2

    # Coeficiente Q dinámico
    if (top_particles['pt'] > 3).any():
        Q_COEFF = 0.01
    else:
        Q_COEFF = Q_COEFF_BASE

    # Resultados
    tracks = []
    hits_vecinos_por_track = []
    cos_values = []
    tripletes_correctos = 0
    total_tripletes = 0
    positivos = 0
    negativos = 0

    for triplet in X_triplets:
        hit1 = triplet[0]  # → [x1, y1, z1]
        hit2 = triplet[1]  # → [x2, y2, z2]
        hit3 = triplet[2]  # → [x3, y3, z3]

        total_tripletes += 1

        # Obtener particle_id de cada hit (aprox por coordenadas)
        pid1 = get_particle_id_from_hit(hit1, hits, truth)
        pid2 = get_particle_id_from_hit(hit2, hits, truth)
        pid3 = get_particle_id_from_hit(hit3, hits, truth)

        if pid1 == pid2 == pid3 and pid1 is not None:
            tripletes_correctos += 1

        # Kalman positivo
        track_posit, hits_vec_posit, chi2_posit = apply_kalmanfilter(
            hit1, hit2, hit3, hits_dict_all_volumes, Q_COEFF,
            get_initial_state, C, F, H, Q, R, volume_ids,
            campo_magnetico, apply_lorentz_correction, SMOOTHING, +1)

        # Kalman negativo
        track_negat, hits_vec_negat, chi2_negat = apply_kalmanfilter(
            hit1, hit2, hit3, hits_dict_all_volumes, Q_COEFF,
            get_initial_state, C, F, H, Q, R, volume_ids,
            campo_magnetico, apply_lorentz_correction, SMOOTHING, -1)

        if chi2_posit < chi2_negat:
            tracks.append(track_posit)
            hits_vecinos_por_track.append(hits_vec_posit)
            positivos += 1
        else:
            tracks.append(track_negat)
            hits_vecinos_por_track.append(hits_vec_negat)
            negativos += 1

    triplet_precision = 100 * tripletes_correctos / total_tripletes if total_tripletes > 0 else 0.0

    print(f"Rango pT: {top_particles['pt'].min():.2f} - {top_particles['pt'].max():.2f} GeV/c")
    print(f"Total tracks: {len(tracks)}, Positivos: {positivos}, Negativos: {negativos}")
    print(f"Tripletes correctos (mismo PID): {tripletes_correctos}/{total_tripletes} ({triplet_precision:.2f}%)")

    return tracks, hits_vecinos_por_track, top_particles, volume_ids, hits_dict_all_volumes, truth, triplet_precision

def get_particle_id_from_hit(hit_coords, hits_df, truth_df, tol=5e-2):
    cond_x = np.isclose(hits_df['x'], hit_coords[0], atol=tol)
    cond_y = np.isclose(hits_df['y'], hit_coords[1], atol=tol)
    cond_z = np.isclose(hits_df['z'], hit_coords[2], atol=tol)
    cond = cond_x & cond_y & cond_z
    hit_ids = hits_df.loc[cond, 'hit_id'].values
    if len(hit_ids) == 0:
        return None
    hit_id = hit_ids[0]
    particle_ids = truth_df[truth_df['hit_id'] == hit_id]['particle_id'].values
    return particle_ids[0] if len(particle_ids) > 0 else None
