# Importar librerías necesarias
from trackml.dataset import load_event
from trackml.utils import add_position_quantities, add_momentum_quantities, decode_particle_id

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


def top_momentum(path, event, pt_min=3.5, pt_max=np.inf):
    hits, cells, particles, truth = load_event(path+event)

    # Calcular pT
    particles['pt'] = np.sqrt(particles['px']**2 + particles['py']**2)

    # Filtrar partículas por intervalo de pt
    particles_filtered = particles[(particles['pt'] >= pt_min) & (particles['pt'] <= pt_max)]

    # Ordenar por pT descendente y seleccionar los top 100 (si hay más de 100)
    top_particles = particles_filtered.sort_values(by='pt', ascending=False)
    print(f'Número de partículas seleccionadas: {len(top_particles)}')
    if not top_particles.empty:
        print(f'pT mayor = {top_particles.pt.max():.4f} GeV/c, pT menor = {top_particles.pt.min():.4f} GeV/c')

    # Obtener IDs de partículas seleccionadas
    particle_ids = top_particles.particle_id.unique()

    # Filtrar hits y truth
    hits_all = hits.copy()
    truth = truth[truth.particle_id.isin(particle_ids)]
    hits = hits[hits.hit_id.isin(truth.hit_id)]

    print("Los datos que tomo son un {:.4f}% de los datos originales".format(hits.shape[0]/hits_all.shape[0]*100))
    print(len(hits.hit_id.unique()), "hits seleccionados")

    # Representación 3D
    fig = plt.figure(figsize=(6, 4))
    plt.suptitle('particles_reduced')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(hits.x, hits.y, hits.z, 'o', markersize=.5, alpha=1., label='Hits correspondientes')
    ax.legend(loc='best')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    # plt.show()

    return hits, truth, top_particles
