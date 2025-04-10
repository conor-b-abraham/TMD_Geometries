import MDAnalysis as mda
from MDAnalysis.analysis import leaflet
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from mpmath import sec
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

def calc_membrane_width(lf1_near, lf2_near):
    memwidth = np.absolute(np.mean(lf1_near.positions[:,2])-np.mean(lf2_near.positions[:,2]))
    return memwidth

def calculate_density_profile(zpos, masses, bins): #, slicearea):
    # binlocs = np.digitize(zpos, bins, right=True)
    # densprof = np.array([np.sum(masses[binlocs==(i+1)]) for i in range(len(bins)-1)])
    # binvol = np.diff(bins)*slicearea
    # densprof = densprof / binvol
    densprof = np.histogram(zpos, bins)[0]
    if np.sum(masses) != 0:
        densprof = densprof / np.sum(masses)

    return densprof

def calc_depths(protBBpos, lf1nearpos, lf2nearpos, densprofags, density_profiles, rep):
    # transpose all position arrays
    lf1posT = lf1nearpos.T
    lf2posT = lf2nearpos.T
    protposT = protBBpos.T

    # find plane of best fit for membrane positions
    # lf1 is the upper leaflet
    lf1mean = np.mean(lf1posT, axis=1, keepdims=True)
    lf1svd = np.linalg.svd(lf1posT-lf1mean)
    lf1normal = lf1svd[0][:, -1]
    if lf1normal[2] < 0:
        lf1normal = -lf1normal # point normal vector out of the membrane (assume it points roughly along the z axis)
    lf1protpos = (protposT-lf1mean).T
    upperdiff = np.zeros(len(lf1protpos[:,0]))
    for i, r in enumerate(lf1protpos):
        upperdiff[i] = r @ lf1normal

    # positions of density groups relative to upper membrane
    upperdenspos = {}
    for agname, ag in densprofags.items():
        upperdenspos[agname] = np.zeros(ag.atoms.n_atoms)
        for i, r in enumerate((ag.positions.T-lf1mean).T):
            upperdenspos[agname][i] = r @ lf1normal


    #lf2 is the lower leaflet
    lf2mean = np.mean(lf2posT, axis=1, keepdims=True)
    lf2svd = np.linalg.svd(lf2posT-lf2mean)
    lf2normal = lf2svd[0][:, -1]
    if lf2normal[2] > 0: # point normal vector out of the membrane (assume it points roughly along the z axis)
        lf2normal = -lf2normal
    lf2protpos = (protposT-lf2mean).T
    lowerdiff = np.zeros(len(lf2protpos[:,0]))
    for i, r in enumerate(lf2protpos):
        lowerdiff[i] = r @ lf2normal

    # positions of density groups relative to lower membrane
    lowerdenspos = {}
    for agname, ag in densprofags.items():
        lowerdenspos[agname] = np.zeros(ag.atoms.n_atoms)
        for i, r in enumerate((ag.positions.T-lf2mean).T):
            lowerdenspos[agname][i] = r @ lf2normal

    # shortest distance to leaflet of density groups
    denspos = {}
    for agname, ag in densprofags.items():
        alldiffs = np.column_stack([upperdenspos[agname], lowerdenspos[agname]])
        allabsdiffs = np.absolute(alldiffs)
        density_profiles[agname][:, rep-1] = density_profiles[agname][:, rep-1] + calculate_density_profile(np.diagonal(alldiffs[:,np.argmin(allabsdiffs, axis=1)]), ag.masses, np.linspace(-25, 25, 101))

    alldiffs = np.column_stack([upperdiff, lowerdiff])
    allabsdiffs = np.absolute(alldiffs)
    indmin = np.argmin(allabsdiffs, axis=1)
    depths = np.diagonal(alldiffs[:,indmin])

    # Membrane Width
    # p1 = np.mean(lf1nearpos, axis=0)
    # p2 = np.mean(lf2nearpos, axis=0)
    # mn = (lf1normal+lf2normal)/2
    # mn = mn/np.linalg.norm(mn)
    # memwidth = np.linalg.norm(mn.dot(p1)-mn.dot(p2))
    return depths, upperdiff, lowerdiff, density_profiles#, memwidth

# MAIN
# get system dependent values
indir=sys.argv[1]
outdir=sys.argv[2]

# ---
bins = np.linspace(-25, 25, 101)

density_profiles = {
"lipid_p":np.zeros((len(bins)-1, 5)), # Lipid Phosphates
"chol_O3":np.zeros((len(bins)-1, 5)) # Cholesterol Hydroxyl
}
# ---

for rep in range(1, 6):
    # create universe

    TPR = f'{indir}/rep_{rep}/3_production.tpr'
    XTC = f'{indir}/rep_{rep}/3_nopbc_cropped.xtc'
    u = mda.Universe(TPR, XTC)

    # create atom groups
    leaflet1, leaflet2 = find_leaflets(u)
    protein = u.select_atoms('protein or resname CYP')
    protein_BB = protein.select_atoms('name BB')
    membrane_and_protein = u.select_atoms('protein or resname CYP or (not resname W WF ION CHOL and name PO4)')
    membrane_near_protein = membrane_and_protein.select_atoms(f'around 20 (protein or resname CYP)', updating=True)

    # main routine
    # ---
    # density profile atom groups
    chol_and_protein = u.select_atoms('protein or resname CYP or (resname CHOL and name ROH)')
    calcags = {
    "lipid_p":membrane_and_protein.select_atoms('around 20 (protein or resname CYP)', updating=True), # Lipid Phosphates
    "chol_O3":chol_and_protein.select_atoms("around 20 (protein or resname CYP)", updating=True) # Cholesterol Hydroxyl
    }
    if calcags["chol_O3"].atoms.n_atoms == 0:
        calcags.pop("chol_O3")
    # ---

    # main routine
    insertion_depths = np.zeros((u.trajectory.n_frames, protein_BB.atoms.n_atoms))
    dist_from_lower = np.zeros_like(insertion_depths)
    dist_from_upper = np.zeros_like(insertion_depths)

    times = np.zeros(u.trajectory.n_frames)
    membrane_width = np.zeros(u.trajectory.n_frames)
    for ts in tqdm(u.trajectory, desc=f"{rep}/5", leave=False):
        # record time
        times[ts.frame] = u.trajectory.time

        # PO4 in each membrane near protein
        lf1_near = leaflet1 & membrane_near_protein
        lf2_near = leaflet2 & membrane_near_protein

        # Membrane width near protein
        membrane_width[ts.frame] = calc_membrane_width(lf1_near, lf2_near)
        # Calculate Insertion Depths
        insertion_depths[ts.frame,:], dist_from_upper[ts.frame,:], dist_from_lower[ts.frame,:], density_profiles = calc_depths(protein_BB.positions, lf1_near.positions, lf2_near.positions, calcags, density_profiles, rep)

    # put times in nanoseconds
    times = times/1000

    # Divide by number of frames and convert from amu/angstrom^3 to kg/m^3
    for agname in density_profiles.keys():
        density_profiles[agname][:, rep-1] = density_profiles[agname][:, rep-1]/u.trajectory.n_frames #*(10000/(u.trajectory.n_frames*6.02214076))

    # add results to lists for output
    np.savez(f'{outdir}/rep_{rep}/insertion_depths.npz', times=times, Dins=insertion_depths, Dupper=dist_from_upper, Dlower=dist_from_lower, membrane_width=membrane_width)

# take the means & normalize
for agname in density_profiles.keys():
    meaned = np.mean(density_profiles[agname], axis=1)
    density_profiles[agname] = meaned/np.max(meaned)

# add bincenters to density profiles for output
density_profiles["bincenters"] = bins[:-1] + (bins[1]-bins[0])/2
np.savez(f'{outdir}/Dins_density_profiles.npz', **density_profiles)
