import numpy as np
from top_momentum_module import top_momentum
from kalman_filter import get_hits_dict, apply_kalmanfilter, get_initial_state, cos_angle, campo_magnetico, apply_lorentz_correction

def track_finding(hits, truth, top_particles, pt_min, pt_max,
                                    DT=1, COS_THRESHOLD=0, Q_COEFF_BASE=0.1, 
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
    vertex = np.array([0, 0, 0])

    # Matrices de Kalman
    F = np.array([[1, DT, 0,  0,  0,  0],
                    [0,  1,  0,  0,  0,  0],
                    [0,  0,  1, DT,  0,  0],
                    [0,  0,  0,  1,  0,  0],
                    [0,  0,  0,  0,  1, DT],
                    [0,  0,  0,  0,  0,  1]])

    H_mat = np.array([[1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0]])

    C = np.eye(6) * 1e-3
    Q = np.eye(6) * 1e-4
    R = np.eye(3) * 1e-4

    tracks, hits_vecinos_por_track, cos_values = [], [], []
    positivos, negativos = 0, 0

    for i in range(len(hits_dict_all_volumes[first_volume][first_layer])):
        hit1 = hits_dict_all_volumes[first_volume][first_layer].iloc[i][['x', 'y', 'z']].values
        v1 = hit1 - vertex

        hits_second = hits_dict_all_volumes[first_volume][second_layer][['x', 'y', 'z']].values
        hit2 = hits_second[np.argmin(np.linalg.norm(hits_second - hit1, axis=1))]
        v2 = hit2 - vertex

        if cos_angle(v1, v2) < COS_THRESHOLD:
            continue

        cos_values.append(cos_angle(v1, v2))

        # Ejecutar filtro de Kalman
        track_posit, hits_vec_posit, chi2_posit = apply_kalmanfilter(hit1, hit2, hits_dict_all_volumes, Q_COEFF,
                                                    get_initial_state, C, F, H_mat, Q, R, volume_ids,
                                                    campo_magnetico, apply_lorentz_correction, SMOOTHING, +1)

        track_negat, hits_vec_negat, chi2_negat = apply_kalmanfilter(hit1, hit2, hits_dict_all_volumes, Q_COEFF,
                                                    get_initial_state, C, F, H_mat, Q, R, volume_ids,
                                                    campo_magnetico, apply_lorentz_correction, SMOOTHING, -1)

        if chi2_posit < chi2_negat:
            tracks.append(track_posit)
            hits_vecinos_por_track.append(hits_vec_posit)
            positivos += 1
        else:
            tracks.append(track_negat)
            hits_vecinos_por_track.append(hits_vec_negat)
            negativos += 1

    print(f"Rango pT: {pt_min:.2f} - {pt_max:.2f} GeV/c")
    print(f"Total tracks: {len(tracks)}, Positivos: {positivos}, Negativos: {negativos}")

    return tracks, hits_vecinos_por_track, top_particles, volume_ids, hits_dict_all_volumes, truth
