from trackml.dataset import load_event
from trackml.utils import add_position_quantities, add_momentum_quantities, decode_particle_id

# get the particles data
particles = load_event('path/to/event000000123', parts=['particles'])
# decode particle id into vertex id, generation, etc.
particles = decode_particle_id(particles)
# add vertex rho, phi, r
particles = add_position_quantities(particles, prefix='v')
# add momentum eta, p, pt
particles = add_momentum_quantities(particles)