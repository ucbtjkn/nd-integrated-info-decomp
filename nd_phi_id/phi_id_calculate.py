import numpy as np
import itertools
import time
from measures import double_redundancy_mmi, local_entropy_discrete, local_entropy_mvn
from phi_id_lattice import generate_phi_id

def calculate_entropy(data, sources, targets, kind="gaussian"):
    
    if kind == "gaussian":
        data_mu = np.mean(data, axis=1) 
        data_cov = np.cov(data)
        def _h(idx):
            return local_entropy_mvn(data[idx].T, data_mu[idx], data_cov[np.ix_(idx, idx)]) if idx else 0.0
    else:
        def _h(idx):
            return local_entropy_discrete(data[idx, :]) if idx else 0.0

    h = np.empty((len(sources), len(targets), 249))
    for i in range(len(sources)):
        for j in range(len(targets)):
            h[i,j,:] = _h(sources[i]+targets[j])
    return h

def calculate_mutual_information(h):

    return (-h + h[0,:,:]) + h[:,[0],:]

def calculate_double_redundancy(mutual_information, atoms, double_redundancy="MMI"):

    if double_redundancy == "MMI":
        double_redundancy_function = double_redundancy_mmi
    elif double_redundancy == "CCS":
        print("Only MMI Double Redundancy Function has been implemented")
        #TODO: Implement generalised CCS double redundancy
        #double_redundancy_function = double_redundancy_ccs
    else:
        print("Only MMI Double Redundancy Function has been implemented")

    mi_means = np.mean(mutual_information, axis=2)
    return np.array([double_redundancy_function(mutual_information, mi_means, atom) for atom in atoms])


def calculate_information_atoms(double_redundancy, matrix):
    return np.linalg.solve(matrix, np.mean(double_redundancy, axis=1))


def phi_id(src, trg, lattice=None, kind="gaussian", double_redundancy="MMI", max_source_atom=None, max_target_atom=None):
    
    if lattice is None:  
        lattice = generate_phi_id(dim=src.shape[0], max_source_atom=max_source_atom, max_target_atom=max_target_atom)

    data = np.concatenate([src, trg])

    if kind == "gaussian":
        data = data / np.std(data, axis=1, ddof=1, keepdims=True) 
    elif kind == "discrete":
        #TODO: Implement discretisation for n values
        print("Discretisation not yet implemented - Discrete data will fail if not drawn from (0,1,...,(dim-1))")


    entropy = calculate_entropy(data, lattice["sources"], lattice["targets"], kind)
    mutual_information = calculate_mutual_information(entropy)
    double_redundancy_value  = calculate_double_redundancy(mutual_information, lattice["atoms"], double_redundancy)
    phi_id_atom_values = calculate_information_atoms(double_redundancy_value, lattice["matrix"])       

    return phi_id_atom_values


def temporal_phi_id(base_data, tau=1, lattice=None, kind="gaussian", double_redundancy="MMI", max_source_atom=None, max_target_atom=None):

    src = base_data[:,:-tau]
    trg = base_data[:,tau:]
    
    return phi_id(src, trg, lattice=lattice, kind=kind, double_redundancy=double_redundancy, max_source_atom=max_source_atom, max_target_atom=max_target_atom)


def phi_id_combinations(base_data, dim=2, tau=1, kind="gaussian", double_redundancy="MMI"):
    
    lattice = generate_phi_id(dim=dim)
    rtr = np.zeros((base_data.shape[0],)*dim)
    sts = np.zeros((base_data.shape[0],)*dim)
    tdmi = np.zeros((base_data.shape[0],)*dim)
    
    t = time.process_time()
    prev_index = (0,)*dim
    for index in itertools.combinations_with_replacement(range(base_data.shape[0]), dim):
        phi_id_atom_values = temporal_phi_id(base_data[index,:], tau, lattice, kind, double_redundancy)

        #TODO: Save other atoms:
        # All the info to construct full matrices is computed: 
        # e.g. rtr[i,j] = rtr[j,i]
        # But some requre permutation e.g. u1tu2[i,j] = u2tu1[j,i] 
        # Still need to figure out a general form of this for non-symmetrical n-dimensional atoms
        for perm in itertools.permutations(index):
            rtr[perm] = np.mean(phi_id_atom_values[0])
            sts[perm] = np.mean(phi_id_atom_values[-1])
            tdmi[perm] = np.sum(np.mean(phi_id_atom_values, axis=0))

        if (dim > 2 and index[-3] > prev_index[-3]):

            np.savetxt("results/rtr/rtr_{}".format(prev_index[:-2]), rtr[prev_index[:-2]], delimiter=',')
            np.savetxt("results/sts/sts_{}".format(prev_index[:-2]), sts[prev_index[:-2]], delimiter=',')
            np.savetxt("results/tdmi/tdmi_{}".format(prev_index[:-2]), tdmi[prev_index[:-2]], delimiter=',')
            
            print(prev_index)
            print(time.process_time() - t)
            t = time.process_time()
            prev_index = index

    if dim == 2:
        np.savetxt("results/rtr/rtr", rtr, delimiter=',')
        np.savetxt("results/sts/sts", sts, delimiter=',')
        np.savetxt("results/tdmi/tdmi", tdmi, delimiter=',')
    return rtr, sts, tdmi


#data = np.genfromtxt("test_data.csv", delimiter=",")
#phi_id_combinations(data, dim=3)