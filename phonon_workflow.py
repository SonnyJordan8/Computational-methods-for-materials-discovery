# MLIP phonon workflow (Aim 1) - version 3, simplified
#
# Calculates the phonon band structure of a crystal with a chosen MLIP.
# The route is the same for every model, only the calculator changes:
#
# 1. Build the crystal and relax atoms and cell with the MLIP (ASE)
# 2. Make displaced supercells (phonopy)
# 3. Calculate the forces on each displaced supercell with the MLIP
# 4. Make the force constants and save them (phonopy)
# 5. Calculate and plot the band structure (phonopy)
# 6. Compare with reference values and add one row to results/summary.csv
#
# Run from the V3 folder with the python of the model's conda environment:
#     python phonon_workflow.py --model MACE-MP-0-medium --material Si GaAs
#
# Sources:
# ASE relaxation   - https://docs.ase-lib.org/howtos_generated/howto_optimization/optimize_structure.html
# Phonopy steps    - https://phonopy.github.io/phonopy/phonopy-module.html
# Supervisor's code - band_structure_BURNETT_ASE_TO_PHONOPY/Na2SO4_processing.ipynb
#
# Not included yet: Born effective charges and the non-analytic correction,
# so GaAs shows no LO-TO splitting.

# Reads command-line options (--model, --material, --supercell, ...)
import argparse
# Writes the summary.csv results table
import csv
# Timer for each stage
import time
# Date and time stamp for each row of the results table
from datetime import datetime
# Handles file paths as objects
from pathlib import Path

import matplotlib
# Save figures without opening a window (also works on HPC)
matplotlib.use("Agg")
import numpy as np

# ASE imports
# Atoms - atomic symbols, positions, cell and periodicity
from ase import Atoms
# bulk - builds standard crystal structures
from ase.build import bulk
# FrechetCellFilter - lets the optimiser relax the cell as well as the atoms
from ase.filters import FrechetCellFilter
# write - saves structures to file
from ase.io import write
# FIRE - structural optimiser, used instead of BFGS to match Loew et al. 2025 and the Orb-v3 paper
from ase.optimize import FIRE

# Phonopy imports
# Phonopy - holds the supercell, displacements, forces and force constants
from phonopy import Phonopy
# get_band_qpoints_and_path_connections - turns a band path into a list of q-points
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
# PhonopyAtoms - phonopy's own structure class
from phonopy.structure.atoms import PhonopyAtoms

# My calculators file - model name -> function that builds the ASE calculator
from calculators import MODELS


# Settings
# These are the defaults. They can be changed on the command line.
DEFAULT_MODEL = "MACE-MP-0-medium"
DEFAULT_MATERIALS = ["Si", "GaAs"]
DEFAULT_DEVICE = "cpu"
SUPERCELL = 3             # N x N x N copies of the 2-atom primitive cell, 54 atoms for N = 3
DISPLACEMENT = 0.03       # how far phonopy moves an atom, in Angstrom - same as Matbench kappa settings

# These are the same for every run
IS_PLUSMINUS = True       # displace in both + and - directions - Orb-v3 paper Appendix G
FMAX = 0.005              # relaxation stops when all forces are below this, eV/Angstrom - Loew et al. 2025
MAX_STEPS = 500           # stops a stuck relaxation - Loew et al. 2025 / MatRIS paper
BAND_POINTS = 51          # q-points per section of the band path - phonopy docs example
IMAGINARY_LIMIT = -0.05   # THz, a frequency below this counts as an imaginary mode - Batatia et al. 2025

# Results are saved in a "results" folder next to this file
RESULTS_FOLDER = Path(__file__).resolve().parent / "results"


# Materials
# a_start is only a starting guess, the relaxation changes it.
# ref_a and ref_gamma_thz are reference values to compare against. None means no comparison.
MATERIALS = {
    "Si": {
        "symbols": "Si",
        "structure": "diamond",
        "a_start": 5.43,
        "ref_a": 5.431,              # Angstrom - carried over from Week 5, CITATION NEEDED
        "ref_gamma_thz": 15.53,      # THz - carried over from Week 5, CITATION NEEDED
    },
    "GaAs": {
        "symbols": "GaAs",
        "structure": "zincblende",
        "a_start": 5.65,
        "ref_a": 5.653,              # Angstrom - https://www.mateck.com/material/compounds/gallium-arsenide
        "ref_gamma_thz": None,       # needs the TO value, as there is no LO-TO splitting yet
    },
}

# Band path for fcc crystals (Si and GaAs are both fcc)
# Hard coded from phonopy docs - SeeK-path plug in can be used to avoid
FCC_BAND_PATH = [
    [[0, 0, 0], [0.5, 0, 0.5], [0.625, 0.25, 0.625]],
    [[0.375, 0.375, 0.75], [0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.25, 0.75]],
]
FCC_BAND_LABELS = ["$\\Gamma$", "X", "U", "K", "$\\Gamma$", "L", "W"]

# Column names of results/summary.csv
CSV_COLUMNS = [
    "date_time", "material", "model", "supercell", "displacement_A", "status",
    "relax_converged", "relax_steps", "max_force_eV_per_A",
    "a_angstrom", "a_diff_percent",
    "gamma_optical_thz", "gamma_diff_percent",
    "lowest_freq_thz", "imaginary_modes",
    "load_s", "relax_s", "forces_s", "bands_s",
]


# Stage 1: relaxation
def relax_structure(atoms, calc, folder):
    atoms.calc = calc
    cell_filter = FrechetCellFilter(atoms)
    optimiser = FIRE(cell_filter, logfile=str(folder / "relax.log"), trajectory=str(folder / "relax.traj"))
    # run() gives back True if the forces got below FMAX within MAX_STEPS
    converged = optimiser.run(fmax=FMAX, steps=MAX_STEPS)
    # Save the relaxed structure
    write(str(folder / "relaxed.cif"), atoms)
    return bool(converged), optimiser.nsteps


# Stage 2: force constants
# Follows the phonopy docs, sections "Pre-process" and "Post process".
def calculate_force_constants(atoms, calc, supercell, displacement, folder):
    # Step 1: turn the relaxed ASE structure into a phonopy structure
    unitcell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
    )

    # Step 2: set up the supercell
    # primitive_matrix="P" tells phonopy that the unit cell is already the primitive cell.
    # Without it phonopy can pick a different primitive cell, which mixes up the band path.
    phonon = Phonopy(
        unitcell,
        supercell_matrix=[[supercell, 0, 0], [0, supercell, 0], [0, 0, supercell]],
        primitive_matrix="P",
    )

    # Step 3: make the displaced supercells
    phonon.generate_displacements(distance=displacement, is_plusminus=IS_PLUSMINUS)
    supercells = phonon.supercells_with_displacements
    print(f"  displaced supercells: {len(supercells)}")

    # Step 4: calculate the forces on each displaced supercell with the MLIP
    # (VASP does this step in my supervisor's example)
    forces = []
    for supercell_atoms in supercells:
        # Turn the phonopy structure back into an ASE structure
        displaced = Atoms(
            symbols=supercell_atoms.symbols,
            cell=supercell_atoms.cell,
            scaled_positions=supercell_atoms.scaled_positions,
            pbc=True,
        )
        displaced.calc = calc
        forces.append(displaced.get_forces())

    # Step 5: give the forces to phonopy and make the force constants
    phonon.forces = forces
    phonon.produce_force_constants()

    # Step 6: save the structure, displacements, forces and force constants
    # phonopy.load("phonopy_params.yaml") reads this file back in
    phonon.save(str(folder / "phonopy_params.yaml"), settings={"force_constants": True})
    return phonon


# Stage 3: band structure
# Follows the phonopy docs, section "Band structure".
# Frequencies are in THz. Imaginary modes show up as negative frequencies.
def calculate_bands(phonon, folder, title):
    # Turn the band path into q-points and calculate the frequencies at each one
    qpoints, connections = get_band_qpoints_and_path_connections(FCC_BAND_PATH, npoints=BAND_POINTS)
    bands = phonon.run_band_structure(qpoints, path_connections=connections, labels=FCC_BAND_LABELS)

    # Save the band structure as band.yaml, the file my supervisor's notebook reads
    phonon.write_yaml_band_structure(filename=str(folder / "band.yaml"))

    # Plot with phonopy's own plot, as in my supervisor's notebook
    plt = phonon.plot_band_structure()
    fig = plt.gcf()
    fig.axes[0].set_ylabel("Frequency (THz)")
    fig.suptitle(title)
    fig.savefig(str(folder / "band_structure.png"), dpi=200)
    plt.close(fig)

    # Lowest frequency anywhere on the band path
    # bands.frequencies is a list with one array per section of the path
    lowest_frequency = np.concatenate(bands.frequencies).min()
    return float(lowest_frequency)


# Frequencies at the Gamma point, q = (0, 0, 0)
# Follows the phonopy docs, section "Phonon at arbitrary q-points".
def gamma_optical_frequency(phonon):
    gamma = phonon.run_qpoints([[0, 0, 0]])
    frequencies = np.sort(gamma.frequencies[0])
    # The 3 lowest are the acoustic modes (about 0 THz). The rest are optical.
    optical = frequencies[3:]
    return float(optical.mean())


# Adds one row to results/summary.csv
def save_row(row):
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    csv_file = RESULTS_FOLDER / "summary.csv"
    # Only write the column names if the file is new
    is_new = not csv_file.exists()
    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(CSV_COLUMNS)
        writer.writerow(row)


# Runs the whole workflow for one material
def run_material(material_name, model_name, calc, load_time, supercell, displacement):
    print(f"\n===== {material_name} | {model_name} =====")
    material = MATERIALS[material_name]

    # Each run gets its own folder, e.g. results/Si/MACE-MP-0-medium/sc3_d0.03
    folder = RESULTS_FOLDER / material_name / model_name / f"sc{supercell}_d{displacement}"
    folder.mkdir(parents=True, exist_ok=True)

    # Build the 2-atom primitive cell
    atoms = bulk(material["symbols"], material["structure"], a=material["a_start"])

    # Warm-up: the first calculation with a model is slow, so do one before timing
    atoms.calc = calc
    atoms.get_forces()

    # Stage 1: relaxation
    start = time.time()
    converged, steps = relax_structure(atoms, calc, folder)
    relax_time = time.time() - start
    # Largest force left on any atom after relaxation
    max_force = float(np.abs(atoms.get_forces()).max())
    # Cubic lattice constant. The fcc primitive cell has volume a^3 / 4.
    a = float((4 * atoms.get_volume()) ** (1 / 3))

    # Stage 2: force constants
    start = time.time()
    phonon = calculate_force_constants(atoms, calc, supercell, displacement, folder)
    forces_time = time.time() - start

    # Stage 3: band structure
    start = time.time()
    lowest_frequency = calculate_bands(phonon, folder, f"{material_name}: {model_name}")
    gamma_optical = gamma_optical_frequency(phonon)
    bands_time = time.time() - start

    # Compare with the reference values, as a percentage difference
    a_diff = ""
    if material["ref_a"] is not None:
        a_diff = (a - material["ref_a"]) / material["ref_a"] * 100
    gamma_diff = ""
    if material["ref_gamma_thz"] is not None:
        gamma_diff = (gamma_optical - material["ref_gamma_thz"]) / material["ref_gamma_thz"] * 100
    # Imaginary modes show up as negative frequencies
    imaginary_modes = lowest_frequency < IMAGINARY_LIMIT

    # Print the results
    print(f"  relaxation converged: {converged} ({steps} steps, max force {max_force:.5f} eV/A)")
    print(f"  lattice constant: {a:.4f} A")
    if a_diff != "":
        print(f"    difference from reference: {a_diff:.1f} %")
    print(f"  Gamma optical frequency: {gamma_optical:.2f} THz")
    if gamma_diff != "":
        print(f"    difference from reference: {gamma_diff:.1f} %")
    print(f"  lowest frequency: {lowest_frequency:.4f} THz (imaginary modes: {imaginary_modes})")
    print(f"  times: relax {relax_time:.1f} s, forces {forces_time:.1f} s, bands {bands_time:.1f} s")

    # Add one row to the results table, in the same order as CSV_COLUMNS
    save_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), material_name, model_name, supercell, displacement, "OK",
        converged, steps, max_force,
        a, a_diff,
        gamma_optical, gamma_diff,
        lowest_frequency, imaginary_modes,
        round(load_time, 2), round(relax_time, 2), round(forces_time, 2), round(bands_time, 2),
    ])
    print(f"  saved to {folder}")


# Run
# The code below only runs when this file is run directly, not when it is imported.
 # Load the model once and time it
start = time.time()
calc = MODELS[DEFAULT_MODEL](DEFAULT_DEVICE)
load_time = time.time() - start
print(f"loaded {DEFAULT_MODEL} in {load_time:.1f} s")

# Run each material. If one fails, record it and carry on with the next.
for material_name in DEFAULT_MATERIALS:
    try:
        run_material(material_name, DEFAULT_MODEL, calc, load_time, SUPERCELL, DISPLACEMENT)
    except Exception as error:
        status = f"CRASHED: {type(error).__name__}: {error}"
        print(f"  {status}")
        save_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), material_name, DEFAULT_MODEL,
            SUPERCELL, DISPLACEMENT, status,
        ])

