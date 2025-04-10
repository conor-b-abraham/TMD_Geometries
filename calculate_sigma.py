import MDAnalysis as mda
from MDAnalysis.analysis import leaflet
# from cmath import rect, phase
# from math import radians, degrees
import numpy as np
import sys
import os
from tqdm import tqdm
from scipy import stats

# FUNCTIONS
def find_leaflets(u):
    membrane_head_ag = u.select_atoms('(not protein and not resname CYP and not resname W WF ION CHOL and name PO4)') # PO4 atoms of Lipids
    # METHOD 1: with MDAnalysis leaflet finder
    LF = leaflet.LeafletFinder(u, membrane_head_ag) # leaflet finder instance
    g0meanz = np.mean(LF.groups(0).atoms.positions[:,2])
    g1meanz = np.mean(LF.groups(1).atoms.positions[:,2])
    if g0meanz > g1meanz: # make leaflet1 the upper leaflet
        leaflet1 = LF.groups(0)
        leaflet2 = LF.groups(1)
    if g1meanz > g0meanz:
        leaflet1 = LF.groups(1)
        leaflet2 = LF.groups(0)
    total_found = leaflet1.atoms.n_atoms+leaflet2.atoms.n_atoms

    # METHOD 2 (if leaflet finder does not find all components): Using membrane center of mass
    if total_found != membrane_head_ag.atoms.n_atoms:
        memCOM = membrane_head_ag.center_of_mass()[2]
        leaflet1 = membrane_head_ag.select_atoms(f'prop z > {memCOM}')
        leaflet2 = membrane_head_ag.select_atoms(f'prop z < {memCOM}')

        total_found = leaflet1.atoms.n_atoms+leaflet2.atoms.n_atoms
        if total_found != membrane_head_ag.atoms.n_atoms:
            sys.exit(' ERROR: Both methods for finding leaflets failed')
        else:
            print(' Method 2 for finding leaflets was successful. Please check leaflets carefully.')
    else:
        print(' Method 1 for finding leaflets was successful. Please check leaflets carefully.')

    print(f' Leaflet 1 (upper): {leaflet1.atoms.n_atoms} components found')
    print(f' Leaflet 2 (lower): {leaflet2.atoms.n_atoms} components found')
    print(f'             Total: {total_found}/{membrane_head_ag.atoms.n_atoms} found')

    return leaflet1, leaflet2

def calculate_membrane_normal(lf1nearpos, lf2nearpos):
    # transpose all position arrays
    lf1posT = lf1nearpos.T
    lf2posT = lf2nearpos.T

    # find plane of best fit for membrane positions
    # lf1 is the upper leaflet
    lf1mean = np.mean(lf1posT, axis=1, keepdims=True)
    lf1svd = np.linalg.svd(lf1posT-lf1mean)
    lf1normal = lf1svd[0][:, -1]
    if lf1normal[2] < 0:
        lf1normal = -lf1normal # point normal vector up in z direction

    # find plane of best fit for membrane positions
    # lf2 is the lower leaflet
    lf2mean = np.mean(lf2posT, axis=1, keepdims=True)
    lf2svd = np.linalg.svd(lf2posT-lf2mean)
    lf2normal = lf2svd[0][:, -1]
    if lf2normal[2] < 0:
        lf2normal = -lf2normal # point normal vector up in z direction
    membrane_normal = 0.5*(lf1normal+lf2normal) # mean of the two vectors will be considered the membrane normal
    membrane_normal = membrane_normal/np.linalg.norm(membrane_normal)
    return membrane_normal

def calculate_angle_180(v1, v2):
    uv1 = v1 / np.linalg.norm(v1)
    uv2 = v2 / np.linalg.norm(v2)
    angle = np.degrees(np.arccos(np.clip(np.dot(uv1, uv2), -1.0, 1.0)))
    return angle

def calculate_angle_360(v1, v2, normal):
    angle = calculate_angle_180(v1, v2)
    normal = normal / np.linalg.norm(normal)
    if np.dot(np.cross(v1, v2), normal)  > 0.0: # v1 clockwise from v2
        angle = 360-angle
    # if angle < 0:
    #     angle = 360-angle # put from 0 to 360
    return angle

def project_vector(plane_normal, vector):
    # project vector onto membrane plane
    projn = (np.dot(vector, plane_normal)/np.linalg.norm(plane_normal)**2)*plane_normal
    projected_vector = vector-projn
    return projected_vector/np.linalg.norm(projected_vector)

def get_tilt_vector(TMD, membrane_normal):
    protein_axis = np.linalg.svd(TMD.positions-TMD.center_of_mass())[2][0]
    if protein_axis[2] < 0:
        protein_axis = -protein_axis
    protein_axis = protein_axis / np.linalg.norm(protein_axis)
    tilt_vector = project_vector(membrane_normal, protein_axis)
    return tilt_vector

def get_reference_vector(reference_position, residue_position, membrane_normal):
    reference_vector = residue_position-reference_position
    reference_vector = project_vector(membrane_normal, reference_vector/np.linalg.norm(reference_vector))
    return reference_vector

# def mean_angle(deg):
#     return degrees(phase(sum(rect(1, radians(d)) for d in deg)/len(deg)))

def mean_angle(deg):
    sindeg = (1/len(deg))*np.sum(np.sin(np.radians(deg)))
    cosdeg = (1/len(deg))*np.sum(np.cos(np.radians(deg)))
    return np.degrees(np.arctan2(sindeg, cosdeg))

# SYSTEM REFERENCE VALUES
firstlast = {'t35':(1,36),'t40':(3,31)}


# MAIN

indir = sys.argv[1]
outdir = sys.argv[2]
memtype = sys.argv[3]


first = firstlast[memtype][0]
last = firstlast[memtype][1]

all_sigma, all_times = [], []
for r in range(1, 6):

    u = mda.Universe(f'{indir}/rep_{r}/3_production.tpr', f'{indir}/rep_{r}/3_nopbc_cropped.xtc')
    protein = u.select_atoms('protein or resname CYP')
    TMD = protein.select_atoms(f'resid {first}:{last} and name BB')
    protein_CA = protein.select_atoms('name BB')
    membrane = u.select_atoms('protein or resname CYP or (not resname W WF ION CHOL and name PO4)')
    membrane_near_protein = membrane.select_atoms(f'around 20 (protein or resname CYP)', updating=True)
    first_CA = protein.select_atoms(f'resid {first}:{first+3} and name BB')
    last_CA = protein.select_atoms(f'resid {last-3}:{last} and name BB')
    leaflet1, leaflet2 = find_leaflets(u) # top leaflet (1) and bottom leaflet (2)

    # main routine
    sigma = np.zeros((u.trajectory.n_frames, protein_CA.residues.n_residues))
    times = np.zeros(u.trajectory.n_frames)

    # check protein orientation
    if first_CA.center_of_mass()[2] > last_CA.center_of_mass()[2]: # If z-pos of N-term is above z-pos of C-term, flip mem normal
        normal_direction = -1
    else:
        normal_direction = 1

    for ts in tqdm(u.trajectory):
        times[ts.frame] = u.trajectory.time
        # set logic. phosphate in each membrane near protein
        lf1_near = leaflet1 & membrane_near_protein
        lf2_near = leaflet2 & membrane_near_protein

        # find membrane normal
        membrane_normal = normal_direction*calculate_membrane_normal(lf1_near.positions, lf2_near.positions)

        # find tilt vector
        tilt_vector = get_tilt_vector(TMD, membrane_normal)

        # find sigma angles
        for i, residue_position in enumerate(protein_CA.positions):
            if i < first-1: # N-term JMD
                reference_position = np.mean(first_CA.positions, axis=0)
            elif i > last-1: # C-term JMD
                reference_position = np.mean(last_CA.positions, axis=0)
            elif i <= first: # first two in helix
                # reference_position = np.mean(np.vstack((protein_CA.positions[i+1, :], protein_CA.positions[i+3:i+6, :])), axis=0)
                reference_position = np.mean(first_CA.positions, axis=0)
            elif i >= last-2: # last in helix
                # reference_position = np.mean(np.vstack((protein_CA.positions[i-1, :], protein_CA.positions[i-6:i-3, :])), axis=0)
                reference_position = np.mean(last_CA.positions, axis=0)

            else: # helix (not first or last)
                reference_position = np.mean(np.vstack((protein_CA.positions[i-3:i-1, :], protein_CA.positions[i+1:i+3, :])), axis=0)
            reference_vector = get_reference_vector(reference_position, residue_position, membrane_normal)
            sigma[ts.frame,i] = calculate_angle_360(tilt_vector, reference_vector, membrane_normal)
    np.savez(f'{outdir}/rep_{r}/sigmas.npz', times=times, sigmas=sigma)
    all_sigma.append(sigma)
    all_times.append(times)

all_sigma = np.vstack(all_sigma).T
mean_sigmas = np.array([mean_angle(v) for v in all_sigma])
np.save(f'{outdir}/mean_sigmas.npy', mean_sigmas)

