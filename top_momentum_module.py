from trackml.dataset import load_event
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def top_momentum(hits, particles, truth, pt_min=3.5, pt_max=np.inf, r_max=2.6, vz_bounds=(-25, 25), VISUALIZAR = False):
    # Calcular pT y r
    particles['pt'] = np.sqrt(particles['px']**2 + particles['py']**2)
    particles['r'] = np.sqrt(particles.vx**2 + particles.vy**2 + particles.vz**2)
    #particles['phi'] = np.arctan2(particles.vy, particles.vx)
    #particles['theta'] = np.arccos(particles.vz / particles.r)

    if 'pt' not in particles.columns:
        particles['pt'] = np.sqrt(particles['px']**2 + particles['py']**2)

    # Filtro por pT
    particles = particles[(particles['pt'] >= pt_min) & (particles['pt'] <= pt_max)]

    # Filtro por zona central vertex
    particles = particles[(particles['r'] < r_max) & 
                          (particles['vz'] > vz_bounds[0]) & 
                          (particles['vz'] < vz_bounds[1])]

    print(f'Partículas seleccionadas: {len(particles)}')
    if not particles.empty:
        print(f'pT max = {particles.pt.max():.4f} GeV/c, pT min = {particles.pt.min():.4f} GeV/c')

    # Filtrado de truth e hits
    particle_ids = particles.particle_id.unique()
    truth = truth[truth.particle_id.isin(particle_ids)]
    hits_all = hits.copy()
    hits = hits[hits.hit_id.isin(truth.hit_id)]

    print("Los datos que tomo son un {:.4f}% de los datos originales".format(hits.shape[0]/hits_all.shape[0]*100))
    print(len(hits.hit_id.unique()), "hits seleccionados")

    print(f"→ particle_ids únicos: {len(particle_ids)}")
    print(f"→ truth con esos particle_ids: {len(truth)}")
    print(f"→ hits con esos hit_id: {len(hits)}")

    if VISUALIZAR:
        # Representación 3D
        fig = plt.figure(figsize=(6, 4))
        plt.suptitle('Hits seleccionados ' + rf'($p_T\ \in\ [{pt_min},\ {pt_max})\ GeV/c)$', fontsize=12)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(hits.x, hits.y, hits.z, 'o', markersize=.5, alpha=1., label='Hits')
        ax.legend(loc='best')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        plt.show()

    return hits, particles, truth
