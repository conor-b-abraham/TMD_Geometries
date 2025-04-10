import MDAnalysis as mda
from MDAnalysis.analysis import leaflet
import numpy as np
import sys
import os
from tqdm import tqdm

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

def compute_angle(v1, v2):
    angle = np.rad2deg(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    angle = 180-angle if angle > 90 else angle
    return angle

def compute_kappa(C, N):
    Cv = np.linalg.svd(C.positions-C.center_of_mass())[2][0]
    Nv = np.linalg.svd(N.positions-N.center_of_mass())[2][0]
    return compute_angle(Nv, Cv)

def compute_theta(TMD, nv):
    Tv = np.linalg.svd(TMD.positions-TMD.center_of_mass())[2][0]
    return compute_angle(Tv, nv)

#MAIN

indir = sys.argv[1]
outdir = sys.argv[2]
first = 5
last = 32
hinge = 24

for r in range(1, 6):

    u = mda.Universe(f'{indir}/rep_{r}/3_production.tpr', f'{indir}/rep_{r}/3_nopbc_cropped.xtc')
    protein = u.select_atoms('protein or resname CYP')
    TMD = protein.select_atoms(f'resid {first}:{last} and name BB')
    # membrane_near_protein = u.select_atoms(f'(not protein and not resname W WF ION CHOL and name PO4) \
    # and ((around 20 (name BB and protein and resid {protein.residues.resids[0]}:{first+3})) \
    # or (around 20 (name BB and protein and resid {last-3}:{protein.residues.resids[-1]})))',
    # updating=True) # membrane PO4 atoms within 2 nm of the top or bottom 4 residues in transmembrane domain
    membrane_near_protein = u.select_atoms(f'(not protein and not resname CYP and not resname W WF ION CHOL and name PO4) and around 20 (protein or resname CYP)',
                                            updating=True)

    leaflet1, leaflet2 = find_leaflets(u)

    Nside = TMD.select_atoms(f'resid {first}:{hinge}')
    Cside = TMD.select_atoms(f'resid {hinge}:{last}')
    kappas = np.zeros(u.trajectory.n_frames)
    thetas = np.zeros(u.trajectory.n_frames)
    times = np.zeros(u.trajectory.n_frames)

    for ts in tqdm(u.trajectory, desc=f"{r}/5", leave=False):
        times[ts.frame] = u.trajectory.time

        # find leaflets PO4 atoms around protein
        lf1_near = leaflet1 & membrane_near_protein
        lf2_near = leaflet2 & membrane_near_protein

        # find membrane normal
        membrane_normal = calculate_membrane_normal(lf1_near.positions, lf2_near.positions)

        # find tilt angle
        thetas[ts.frame] = compute_theta(TMD, membrane_normal)

        # find hinge angle
        kappas[ts.frame] = compute_kappa(Cside, Nside)

    times = times/1000 # times in nanoseconds
    np.savez(f'{outdir}/rep_{r}/helix_angles.npz', times=times, thetas=thetas, kappas=kappas)

