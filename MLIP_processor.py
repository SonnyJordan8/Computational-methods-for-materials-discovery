from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import time

from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.optimize import FIRE

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath

import phonopy
from phonopy import Phonopy
from phonopy.file_IO import write_FORCE_SETS
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections


# Results are saved in a "results" folder next to this file
RESULTS_FOLDER = Path(__file__).resolve().parent / "results"

# Settings that are the same for every run - May need to be edited depending on how tight convergence is needed

IS_PLUSMINUS = True       # displace in both + and - directions - Orb-v3 paper Appendix G
FMAX = 0.005              # relaxation stops when all forces are below this, eV/Angstrom - Loew et al. 2025
MAX_STEPS = 500           # stops a stuck relaxation - Loew et al. 2025 / MatRIS paper
BAND_POINTS = 51          # q-points per section of the band path - phonopy docs, "Band structure" section example

# Files saved in each run folder (the same layout as phononDB): the structure and displacements
# in a small yaml file, and the forces in a separate FORCE_SETS file. The force constants are
# not saved - they are recomputed from FORCE_SETS when a run is loaded, because a saved force
# constant matrix gets very large for big supercells.
DISPLACEMENTS_FILE = "phonopy_disp.yaml"
FORCE_SETS_FILE = "FORCE_SETS"

# Column names of results/summary.csv
CSV_COLUMNS = [
    "date_time", "material", "model", "supercell", "displacement_A", "status",
    "relax_converged", "relax_steps", "max_force_eV_per_A",
    "space_group_before", "space_group_after", "symmetry_kept",
    "volume_per_atom_A3",
    "gamma_optical_thz", "ref_gamma_optical_thz", "gamma_percent_diff",
    "lowest_freq_thz", "imaginary_modes",
    "load_s", "relax_s", "forces_s", "bands_s",
]

# Column names for performance_log.csv
PERFORMANCE_COLUMNS = [
    "date_time", "material", "model", "device", "supercell", "displacement_A", "stage",
    "wall_time_s", "cpu_time_s", "cores_used", "cores_available", "cpu_usage_percent",
]

# Performance log

def start_timer():
    """
    Starts timing a stage. Returns the wall clock time and the CPU time at this moment.
    Pass the result to performance_log when the stage has finished.

    """
    return time.time(), time.process_time()

def get_cores_available():
    """
    Returns the number of CPU cores this process is allowed to use.
    On a supercomputer this is the number of cores given to the job, not the whole node.

    """
    if hasattr(os, "sched_getaffinity"):    # Linux only
        return len(os.sched_getaffinity(0))
    return os.cpu_count()                   # Mac and Windows

def performance_log(material_name, model_name, device, supercell, displacement, stage, timer):
    """
    Adds one row to results/performance_log.csv for one stage of one run.
    "timer" is the value returned by start_timer before the stage began.
    Returns the wall time of the stage in seconds.

    """
    wall_start, cpu_start = timer
    wall_time = time.time() - wall_start            # time taken
    cpu_time = time.process_time() - cpu_start      # compute time, summed over all cores

    cores_used = cpu_time / wall_time               # average number of busy cores
    cores_available = get_cores_available()
    cpu_usage = cores_used / cores_available * 100

    RESULTS_FOLDER.mkdir(parents = True, exist_ok = True)
    csv_file = RESULTS_FOLDER / "performance_log.csv"
    is_new = not csv_file.exists()
    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(PERFORMANCE_COLUMNS)
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"), material_name, model_name, device,
            supercell, displacement, stage,
            round(wall_time, 2), round(cpu_time, 2), round(cores_used, 2),
            cores_available, round(cpu_usage, 1),
        ])
    return wall_time

# Folder generation for one run

def get_run_folder(material_name, model_name, supercell, displacement):
    """
    Returns the folder for one run, e.g. results/Si/MACE-MP-0-medium/sc3_d0.03
    The folder is not created here.

    """
    return RESULTS_FOLDER / material_name / model_name / f"sc{supercell}_d{displacement}"

# Symmetry finder

def get_space_group(material):
    """
    Returns the space group symbol and number of an ASE structure.
    Used after relaxation to check the MLIP has kept the symmetry.

    """
    structure_pmg = AseAtomsAdaptor.get_structure(material)
    finder = SpacegroupAnalyzer(structure_pmg)
    return finder.get_space_group_symbol(), finder.get_space_group_number()

def get_primitive_cell(material):
    """
    Function finds the primitive cell using pymatgen and returns the 
    ASE atoms object of the primitivie cell

    """
    structure_pmg = AseAtomsAdaptor.get_structure(material)
    space_group_analyser = SpacegroupAnalyzer(structure_pmg)
    primitive_structure_pmg = space_group_analyser.get_primitive_standard_structure()
    primitive_structure_ase = AseAtomsAdaptor.get_atoms(primitive_structure_pmg)
    return(primitive_structure_ase)

def get_band_path(primitive_material):
    """
    Function that finds the high symmetry K points of the band 
    and generates the labels for phonon band plot
    """

    structure_pmg = AseAtomsAdaptor.get_structure(primitive_material)
    kpath_object = HighSymmKpath(structure_pmg)
    # Dictionary, label -> q-point coordinates
    high_symm_points = kpath_object.kpath["kpoints"]

    # band_path: one list of q-points per continuous section of the path
    # band_labels: the label of every q-point, in the same order, as one flat list
    band_path = []
    band_labels = []
    for section in kpath_object.kpath["path"]:
        qpoints = []
        for label in section:
            qpoints.append(list(high_symm_points[label]))
            # Greek labels such as \Gamma need $...$ so matplotlib draws the symbol
            if label.startswith("\\"):
                label = f"${label}$"
            band_labels.append(label)
        band_path.append(qpoints)
    return band_path, band_labels

# Stage 1: relaxation

def relax_structure(material, calc, folder):
    """
    Returns the relaxed structure of the material using the MLIP and FIRE + FrechetCellFilter
    Saves the relax.log, relaxed.traj and the relaxed.cif in the folder
    the materials object is changed into the relaxed structure.
    """
    material.calc = calc
    cell_filter = FrechetCellFilter(material)
    optimiser = FIRE(cell_filter, logfile = str(folder / "relax.log"), trajectory = str(folder / "relaxed.traj"))
    converged = optimiser.run(fmax = FMAX, steps = MAX_STEPS)
    write(str(folder / "relaxed.cif"), material)
    return bool(converged), optimiser.nsteps

# Stage 2: force constant calculation

def force_constant_calculator(material, calc, supercell, displacement, folder):

    """
    Function thst calculates the displacements of the cell in phonopy,
    the forces on the atoms in ASE and the force constants in phonopy.
    Saves phonopy_disp.yaml (structure and displacements) and FORCE_SETS
    (displacements and forces) in the folder. 


    """
    cell = PhonopyAtoms(
        symbols = material.get_chemical_symbols(),
        cell = material.cell.array,
        scaled_positions = material.get_scaled_positions()
    )

    phonon = Phonopy(
        cell,
        supercell_matrix=[[supercell, 0, 0], [0, supercell, 0], [0, 0, supercell]],
        primitive_matrix="P",
    )

    phonon.generate_displacements(distance = displacement, is_plusminus = IS_PLUSMINUS)
    supercells = phonon.supercells_with_displacements
    print(f"Displaced supercells: {len(supercells)}")
    phonon.save(str(folder / DISPLACEMENTS_FILE), settings = {"force_sets": False})
    
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

    phonon.forces = forces
    phonon.produce_force_constants()

    write_FORCE_SETS(phonon.dataset, filename=str(folder / FORCE_SETS_FILE))
    return phonon

def load_run_phonon(run_folder):
    """
    Loads the saved phonons of one run folder (phonopy_disp.yaml and FORCE_SETS).
    Returns None if the run folder has no saved phonons.
    """
    displacements_file = run_folder / DISPLACEMENTS_FILE
    force_sets_file = run_folder / FORCE_SETS_FILE
    if not (displacements_file.exists() and force_sets_file.exists()):
        return None
    return phonopy.load(str(displacements_file), force_sets_filename=str(force_sets_file))

# Stage 3: Band structure

def band_calculator(phonon, band_path, band_labels):

    """
    Calculates the band structure of a Phonopy object aling the band path obtained by 
    get_band_path function. 
    Returns the dictionary:
    "distances" - a list with one array of x-axis positions per section of the path
    "frequencies" - a list with one array of frequencies (THz) per section of the path
    "connections" - a list saying whether each section joins on to the next one 

    """
    
    qpoints, connections = get_band_qpoints_and_path_connections(band_path, npoints = BAND_POINTS)
    phonon.run_band_structure(qpoints, path_connections = connections, labels = band_labels)
    band_dict = phonon.get_band_structure_dict()

    bands = {
    "distances": band_dict["distances"],
    "frequencies": band_dict["frequencies"],
    "connections": connections,
    }
    return bands

def save_bands(phonon, folder):

    """
    Saves the band structure as band.yaml in the folder.

    """
    phonon.write_yaml_band_structure(filename=str(folder / "band.yaml"))

def lowest_band_frequency(bands):
    """
    Returns the lowest frequency (THz) anywhere on the band path.
    A value below IMAGINARY_LIMIT means there are imaginary modes.

    """
    return float(np.concatenate(bands["frequencies"]).min())

def gamma_optical_frequency(phonon):
    """
    Returns the highest frequency (THz) at the Gamma point.

    """
    phonon.run_qpoints([[0, 0, 0]])
    gamma_frequencies = phonon.get_qpoints_dict()["frequencies"]
    return float(np.sort(gamma_frequencies[0])[-1])

# Results table

def results_table(row):
    """
    Adds one row to the results summary.csv file

    """
    RESULTS_FOLDER.mkdir(parents = True, exist_ok = True)
    csv_file = RESULTS_FOLDER / "summary.csv"
    is_new = not csv_file.exists()
    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(CSV_COLUMNS)
        writer.writerow(row)


# Load phonon band structures for comparison plot
def phonon_loader(material_name, supercell, displacement):
    """
    Loads the saved phonons of every model that has been run for this material
    with this supercell and displacement.
    Returns a dictionary with the model name -> Phonopy object
    """
    phonons = {}
    material_folder = RESULTS_FOLDER / material_name
    if not material_folder.exists():
        return phonons

    for model_folder in sorted(material_folder.iterdir()):
        if not model_folder.is_dir(): 
            continue
        phonon = load_run_phonon(model_folder / f"sc{supercell}_d{displacement}")
        if phonon is not None:
            phonons[model_folder.name] = phonon
    return phonons


# Plots
def plot_bands(datasets, reference_phonon, band_path, band_labels, title, figure_file, reference_label="REF"):
    """
    Plots the band structure of every Phonopy object in "datasets" (label -> Phonopy)
    on the same figure, along the band path from get_band_path.
    The reference data is added as a black line if reference_phonon is not None.
    Saves the figure to figure_file and returns its path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Every data set uses the same q-points, so they can share one x-axis.
    # The x-axis positions of the first data set are used for all of them,
    # because each model relaxes to a slightly different cell size.
    x_axis = None
    connections = None

    # Reference first so it sits underneath the MLIP lines
    if reference_phonon is not None:
        bands = band_calculator(reference_phonon, band_path, band_labels)
        x_axis = bands["distances"]
        connections = bands["connections"]
        for section in range(len(x_axis)):
            lines = ax.plot(x_axis[section], bands["frequencies"][section], color="black", linewidth=1.5)
            # Only label one line so the legend has one entry per data set
            if section == 0:
                lines[0].set_label(reference_label)

    # Then one colour per data set
    colour_number = 0
    for label in datasets:
        bands = band_calculator(datasets[label], band_path, band_labels)
        if x_axis is None:
            x_axis = bands["distances"]
            connections = bands["connections"]
        colour = f"C{colour_number}"    # matplotlib's default colour cycle
        for section in range(len(x_axis)):
            lines = ax.plot(x_axis[section], bands["frequencies"][section], color=colour, linewidth=1.0)
            if section == 0:
                lines[0].set_label(label)
        colour_number = colour_number + 1

    # Tick marks at the start and end of each section of the band path
    tick_positions = [x_axis[0][0]]
    tick_labels = [band_labels[0]]
    label_number = 1
    last_section = len(x_axis) - 1
    for section in range(len(x_axis)):
        tick_positions.append(x_axis[section][-1])
        if connections[section] or section == last_section:
            tick_labels.append(band_labels[label_number])
            label_number = label_number + 1
        else:
            # The path jumps here (e.g. K to U), so the tick gets both labels
            tick_labels.append(band_labels[label_number] + "|" + band_labels[label_number + 1])
            label_number = label_number + 2

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    for position in tick_positions:
        ax.axvline(position, color="grey", linewidth=0.5)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlim(tick_positions[0], tick_positions[-1])
    ax.set_ylabel("Frequency (THz)")
    ax.set_title(title)
    ax.legend()

    figure_file = Path(figure_file)
    figure_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(figure_file), dpi=200)
    plt.close(fig)
    return figure_file

def plot_run_bands(material_name, model_name, phonon, reference_phonon, band_path, band_labels, supercell, displacement, reference_label="REF"):
    """
    One run (one model, supercell and displacement) with the reference underneath.
    Saves results/<material>/<model>/sc<supercell>_d<displacement>/band_structure.png
    """
    return plot_bands(
        {model_name: phonon}, reference_phonon, band_path, band_labels,
        title=f"{material_name}: {model_name}, sc{supercell}, displacement {displacement} A",
        figure_file=get_run_folder(material_name, model_name, supercell, displacement) / "band_structure.png",
        reference_label=reference_label,
    )

def plot_band_comparison(material_name, phonons, reference_phonon, band_path, band_labels, supercell, displacement, reference_label="REF"):
    """
    Every model in "phonons" (model name -> Phonopy) on the same figure.
    Saves results/<material>/band_comparison_sc<supercell>_d<displacement>.png
    """
    return plot_bands(
        phonons, reference_phonon, band_path, band_labels,
        title=f"{material_name}: sc{supercell}, displacement {displacement} A",
        figure_file=RESULTS_FOLDER / material_name / f"band_comparison_sc{supercell}_d{displacement}.png",
        reference_label=reference_label,
    )

