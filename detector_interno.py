
# #============== CONSIDERO SOLO LOS HITS DE LA PARTE CENTRAL DEL DETECTOR ==============#

# def distance(particle):
#     ''' Distancia en mm de la partícula al origen'''
#     return np.sqrt(particle.vx**2 + particle.vy**2 + particle.vz**2)

# particles['r'] = distance(particles)
# particles['phi'] = np.arctan2(particles.vy, particles.vx)
# particles['theta'] = np.arccos(particles.vz / particles.r)

# print(particles.head())

# # Voy a coger solo las partículas con r < 2.6
# particles_all = particles
# particles = particles[particles.r < 2.6]

# # Con ese radio, voy a coger solo las partículas con z entre -25 y 25 mm
# particles = particles[(particles.vz > -25) & (particles.vz < 25)]

# # Histograma normalizado a 1 de la variable r
# plt.figure(figsize=(6, 6))
# plt.hist(particles_all.r, bins=100, range=(0, 100), density=False, alpha=0.5, label='All particles')
# plt.hist(particles.r, bins=100, range=(0, 100), density=False, alpha=0.5, label='Selected particles')
# plt.legend()
# plt.xlabel('r (mm)')    
# plt.ylabel('Number of particles')
# plt.grid(linestyle='--', alpha=0.6)
# plt.show()

# # Del truth cojo solo las partículas que están en particles
# truth = truth[truth.particle_id.isin(particles.particle_id)]

# # Cojo ahora los hits_id que están en truth
# hits_all = hits
# hits = hits[hits.hit_id.isin(truth.hit_id)]
# print(hits.head())

# print("Los datos que tomo son un {:.4f}% de los datos originales".format(hits.shape[0]/hits_all.shape[0]*100))


# # Represento el dataset de particles en 3D
# fig = plt.figure(figsize=(6, 6))
# plt.suptitle('Particles')
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(particles.vx, particles.vy, particles.vz, 'o', markersize=1, label = 'Partículas seleccionadas')
# #ax.plot(particles_all.vx, particles_all.vy, particles_all.vz, 'x', alpha  = .6, markersize = .6, label = 'Todas las partículas')
# ax.plot(hits.x, hits.y, hits.z, 'o', markersize=.2, alpha= .4, label = 'Hits correspondientes')
# ax.legend(loc = 'best')
# ax.set_xlabel('X (mm)')
# ax.set_ylabel('Y (mm)')
# ax.set_zlabel('Z (mm)')
# plt.show()
