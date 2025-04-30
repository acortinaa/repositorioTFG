# Importar librerías necesarias
from trackml.dataset import load_event
from trackml.utils import add_position_quantities, add_momentum_quantities, decode_particle_id

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


def top_momentum(path, event):
    hits, cells, particles, truth = load_event(path+event)
    TOP_MOMENTUM = True
    if TOP_MOMENTUM:
        particles['pt'] = np.sqrt(particles['px']**2 + particles['py']**2)

        # # Crear el histograma de pT
        # plt.figure(figsize=(6, 4))
        # plt.hist(particles['pt'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)

        # # Etiquetas y título
        # plt.xlabel('$p_T$ (GeV/c)', fontsize=12)
        # plt.ylabel('Número de partículas', fontsize=12)
        # plt.title('Histograma del momento transversal ($p_T$)', fontsize=14)

        # # Mostrar el gráfico
        # plt.grid(True)
        # #plt.show()

        #print(particles.head())
        # Ordenar por pT descendente y seleccionar los 300 primeros
        top_particles = particles.sort_values(by='pt', ascending=False).head(100)
        print(f'Número de partículas seleccionadas: {len(top_particles)}')
        print(f'pT mayor = {top_particles.pt.max():.4f} GeV/c, pT menor = {top_particles.pt.min():.4f} GeV/c')

        # De ellos cojo el particle_id
        particle_ids = top_particles.particle_id.unique()

        # Con esos miro en truth los hits que corresponden
        hits_all = hits
        truth = truth[truth.particle_id.isin(particle_ids)]
        hits = hits[hits.hit_id.isin(truth.hit_id)]
        print("Los datos que tomo son un {:.4f}% de los datos originales".format(hits.shape[0]/hits_all.shape[0]*100))
        print(len(hits.hit_id.unique()), "hits seleccionados")

        # Represento el dataset de particles_reduced en 3D
        fig = plt.figure(figsize=(6, 4))
        plt.suptitle('particles_reduced')
        ax = fig.add_subplot(111, projection='3d')
        #ax.plot(top_particles.tx, top_particles.ty, top_particles.tz, 'o', markersize=1, label = 'Partículas seleccionadas')
        #ax.plot(particles_all.vx, particles_all.vy, particles_all.vz, 'x', alpha  = .6, markersize = .6, label = 'Todas las partículas')
        ax.plot(hits.x, hits.y, hits.z, 'o', markersize=.5, alpha= 1., label = 'Hits correspondientes')
        ax.legend(loc = 'best')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        #plt.show()
    return hits, truth, top_particles