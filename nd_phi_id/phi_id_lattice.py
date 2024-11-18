import numpy as np
import itertools
from pid_lattice import generate_pid

def generate_phi_id(dim=2, max_source_atom=None, max_target_atom=None):

    # Generate source and target PID lattices
    if max_source_atom is None and max_target_atom is None:
        sources, source_atoms, source_matrix = generate_pid(dim)
        targets = [[id+dim for id in source] for source in sources]
        target_atoms = source_atoms
        target_matrix = source_matrix
    else:
        #TODO: Modify generate_pid to generate both rather than calling twice    
        sources, source_atoms, source_matrix = generate_pid(dim, max_atom=max_source_atom)
        targets, target_atoms, target_matrix = generate_pid(dim, max_atom=max_target_atom)
        targets = [[id+dim for id in source] for source in sources]

    # Construct Phi-ID atoms as product of source and target atoms
    phi_id_atoms = list(itertools.product(source_atoms, target_atoms))

    # Construct Phi-ID partial order matrix from source and target PID matrices
    phi_id_matrix = np.zeros((len(phi_id_atoms), len(phi_id_atoms)))
    for i in range(len(phi_id_atoms)):
        for j in range(i+1):
            phi_id_matrix[i][j] = source_matrix[i//len(target_atoms)][j//len(target_atoms)] and target_matrix[i%len(target_atoms)][j%len(target_atoms)]


    return {
        "matrix": phi_id_matrix,
        "atoms": phi_id_atoms,
        "sources": sources,
        "targets": targets
    }


#generate_phi_id(dim=3)
#generate_phi_id(dim=4, max_source_atom=[{0,1},{2},{3}], max_target_atom=[{0,1,2},{3}])