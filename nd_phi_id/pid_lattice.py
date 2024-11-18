import numpy as np
from itertools import combinations, chain, product
import llist

#TODO: Precompute and save rather than calling this
def generate_pid(dim=3, max_atom=None):
    
    # Get all composite sources
    s = set(range(dim))
    sources = chain.from_iterable(combinations(s, r) for r in range(dim+1))

    # Remove sources which are a superset of any source in max_atom 
    if max_atom is not None:
        sources = filter(lambda source: not any(max_atom_source.issubset(source) and not max_atom_source == set(source) for max_atom_source in max_atom), sources)
    sources = list(list(source) for source in sources)

    # Compute PID ordering on sources
    contains = [[set(b).issubset(a) for a in sources] for b in sources] 

    # Initialise atoms with single sources
    atoms = llist.dllist([[i] for i in range(len(sources))])

    # Build all PID sources
    max_atom_indices = []
    atom_1 = atoms.first
    while not atom_1.value == atoms[-1]:
        atom_1 = atom_1.next
        atom_2 = atoms.first
        while not atom_2.value == atom_1.value:
            atom_2 = atom_2.next
            if not any(contains[s2][atom_1.value[0]] for s2 in atom_2.value):
                joint_atom = llist.dllistnode(atom_2.value + atom_1.value) 
                atoms.insertnodebefore(joint_atom, atom_2)
        
        if max_atom is not None:
            if set(sources[atom_1.value[0]]) in max_atom:
                max_atom_indices += atom_1.value

    # Filter all remaining atoms not less than max_atom
    if max_atom is not None:
        atoms = list(filter(lambda atom: all(any(contains[source][max_atom_source] for source in atom) for max_atom_source in max_atom_indices), atoms))[1:]
    else:
        atoms = list(atoms)[1:]

    # Compute PID partial order for individual sources
    source_contains = np.fromfunction(np.vectorize(lambda i, j: any(contains[atom_source][i] for atom_source in atoms[j])), shape=(len(sources), len(atoms)), dtype=int)

    # Construct complete PID partial order (A1 < A2 iff A1 < S for all S in A2) 
    partial_order_matrix = np.array([np.logical_and.reduce(source_contains[atom,:]) for atom in atoms]).astype(float)

    return sources, atoms, partial_order_matrix


#generate_pid(dim=2)
#generate_pid(dim=3)
#generate_pid(dim=4)
#generate_pid(dim=5)
#generate_pid(dim=6, max_atom=[{0,1,2},{3},{4,5}])


