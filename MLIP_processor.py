from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import time

from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.optimize import FIRE, BFGS, LBFGS, BFGSLineSearch
from ase.constraints import FixSymmetry

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath

import phonopy
from phonopy import Phonopy
from phonopy.file_IO import write_FORCE_SETS, write_FORCE_CONSTANTS, write_force_constants_to_hdf5
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_primitive, guess_primitive_matrix
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

# Settings for calculations can be changed in the config_file.txt
from MLIP_config import CONFIG, read_config


# Settings - all set in config_file.txt, read by MLIP_config.py

# Folder where the results are saved
RESULTS_FOLDER = CONFIG["RESULTS_FOLDER"]

# ASE relaxation
OPTIMIZER = CONFIG["OPTIMIZER"] 
OPTIMIZERS = {"FIRE": FIRE, "BFGS": BFGS, "LBFGS": LBFGS, "BFGSLineSearch": BFGSLineSearch}
RELAX_CELL = CONFIG["RELAX_CELL"]           # relax the cell as well as the atom positions
FMAX = CONFIG["RELAX_FMAX"]                 # relaxation stops when all forces are below this, eV/Angstrom
ENERGY_TOLERANCE = CONFIG["ENERGY_TOLERANCE"]  # ...and the energy change per step is below this, eV/atom (0 = off)
MAX_STEPS = CONFIG["MAX_STEPS"]             # stops a stuck relaxation
RELAX_SYMMETRY = CONFIG["RELAX_SYMMETRY"]   # keep the starting space group during relaxation

# Phonopy
PRIMITIVE_AXES = CONFIG["PRIMITIVE_AXES"]   # "P", "auto" or a 3x3 matrix
PRIMITIVE_TOLERANCE = CONFIG["PRIMITIVE_TOLERANCE"] # float setting for distance tolerance of atom from primitive position
IS_PLUSMINUS = CONFIG["PM"]                 # displace in both + and - directions (phonopy tag PM)
SYMMETRY = CONFIG["SYMMETRY"]               # use crystal symmetry to reduce the displacements
SYMMETRY_TOLERANCE = CONFIG["SYMMETRY_TOLERANCE"]
FC_CALCULATOR = CONFIG["FC_CALCULATOR"]     # "traditional", "symfc" or "alm"
FC_SYMMETRY = CONFIG["FC_SYMMETRY"]         # symmetrise the force constants after fitting
PHONOPY_LOG_LEVEL = CONFIG["PHONOPY_LOG_LEVEL"]  # 0 quiet, 1 normal, 2 verbose
BAND_POINTS = CONFIG["BAND_POINTS"]         # q-points per section of the band path

# Phonopy post-processing
MESH = CONFIG["MESH"]                       # q-point mesh for thermal properties
TPROP = CONFIG["TPROP"]                     # calculate thermal properties
TMIN, TMAX, TSTEP = CONFIG["TMIN"], CONFIG["TMAX"], CONFIG["TSTEP"]   # K
IRREPS = CONFIG["IRREPS"]                   # None, or (q-point, degeneracy tolerance)
SHOW_IRREPS = CONFIG["SHOW_IRREPS"]         # also print the irreducible representations
HDF5 = CONFIG["HDF5"]                       # write band structure and force constants as .hdf5
WRITE_FORCE_CONSTANTS = CONFIG["WRITE_FORCE_CONSTANTS"]   # save the force constants too

# Displacements, force sets and the config file for this run
DISPLACEMENTS_FILE = CONFIG["DISPLACEMENTS_FILE"]
FORCE_SETS_FILE = CONFIG["FORCE_SETS_FILE"]
RUN_CONFIG_FILE = "run_config.txt"

# Column names of results/summary.csv
CSV_COLUMNS = [
    "date_time", "material", "model", "supercell", "displacement_A","run", "status",
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
    with open(csv_file, "a", newline = "") as file:
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

# Folder label generation for one run
def get_run_folder(material_name, model_name, supercell, displacement):
    """
    Returns the folder for one run. Its name holds the supercell, the displacement and the
    symmetry settings and the primitive axis settings
    results/Si/MACE-MP-0-medium/sc2x2x2_d0.03_sym-on_pa-P - all symmetry settings on and primitive cell is cif file cell
    results/Si/MACE-MP-0-medium/sc2x2x2_d0.03_sym-off_relaxsym-off_pa-auto - SYMMETRY off (phonopy) RELAX_SYMMETRY off (ASE)
    , FC_SYMMETRY is on and primitive cell automatically calculated by phonopy
    RELAX_SYMMETRY and FC_SYMMETRY are only added to the name when they are off.
    Every other function that needs a run's folder or name calls this one.
    The folder is not created here.

    """
    name = f"sc{supercell}_d{displacement}_sym-{'on' if SYMMETRY else 'off'}"
    if not RELAX_SYMMETRY:
        name = name + "_relaxsym-off"
    if not FC_SYMMETRY:
        name = name + "_fcsym-off"

    if PRIMITIVE_AXES == "P":
        name = name + "_pa-P"
    elif PRIMITIVE_AXES == "auto":
        name = name + "_pa-auto"
    else:
        name = name + "_pa-matrix"
    return RESULTS_FOLDER / material_name / model_name / name


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
    Function finds the primitive cell from the input CIF file.
    AUTO - phonopy finds the primitive cell, P - the CIF cell is used as primitive cell
    and if a 9 digit array representing custom primitive cell is configured
    that is used. 
    PRIMITIVE_TOLERANCE is applied to starting cif file and is the distance an atom can be 
    from their ideal positions and still count as the same site.

    """
    cell = PhonopyAtoms(symbols = material.get_chemical_symbols(), cell = material.cell.array, 
                        scaled_positions = material.get_scaled_positions())
    if PRIMITIVE_AXES == "auto":
        matrix = guess_primitive_matrix(cell, symprec = PRIMITIVE_TOLERANCE)
    elif PRIMITIVE_AXES == "P":
        matrix = ((1,0,0),(0,1,0),(0,0,1))
    else:
        matrix = np.array(PRIMITIVE_AXES)
    primitive = get_primitive(cell, matrix, symprec = PRIMITIVE_TOLERANCE)
    return Atoms(symbols = primitive.symbols, cell = primitive.cell,
                 scaled_positions = primitive.scaled_positions, pbc = True)

def get_band_path(primitive_material):
    """
    Function that finds the high symmetry K points of the band 
    and generates the labels for phonon band plot
    """

    structure_pmg = AseAtomsAdaptor.get_structure(primitive_material)
    kpath_object = HighSymmKpath(structure_pmg)
    high_symm_points = kpath_object.kpath["kpoints"]
    band_path = []
    band_labels = []
    for section in kpath_object.kpath["path"]:
        qpoints = []
        for label in section:
            qpoints.append(list(high_symm_points[label]))
            band_labels.append(f"${label}$")
        band_path.append(qpoints)
    return band_path, band_labels

# Stage 1: relaxation

def relax_structure(material, calc, folder):
    """
    Relaxes the material with the MLIP. The material object is changed into the relaxed structure.
    Settings from the config:
      OPTIMIZER         which ASE optimiser (FIRE, BFGS, LBFGS, BFGSLineSearch)
      RELAX_CELL        True: cell and atom positions (FrechetCellFilter); False: atom positions only
      RELAX_FMAX        force convergence, eV/Angstrom
      ENERGY_TOLERANCE  energy convergence, eV/atom per step (0 = force convergence only)
      MAX_STEPS         the relaxation stops here if it has not converged
      RELAX_SYMMETRY    keep the starting space group
    Saves the relax.log, relaxed.traj and the relaxed.cif in the folder.
    Returns converged (True/False), the number of steps, and the last energy change per atom (eV).
    """
    material.calc = calc
    if RELAX_SYMMETRY:
        # Forces and cell changes that would break the starting space group are removed
        material.set_constraint(FixSymmetry(material))

    # What the optimiser moves: the atoms only, or the atoms and the cell
    if RELAX_CELL:
        optimizable = FrechetCellFilter(material)
    else:
        optimizable = material
    optimiser = OPTIMIZERS[OPTIMIZER](optimizable, logfile = str(folder / "relax.log"),
                                      trajectory = str(folder / "relaxed.traj"))

    # ASE's own run() only checks forces, so step through the optimiser and check both
    # forces and energy after every step. fmax=0 stops ASE ending the run by itself.
    number_of_atoms = len(material)
    previous_energy = None
    energy_change = float("nan")
    converged = False
    for _ in optimiser.irun(fmax = 0.0, steps = MAX_STEPS):
        # Largest force, measured the same way as ASE's fmax (includes the cell when it relaxes)
        forces = optimizable.get_forces()
        largest_force = np.sqrt((forces ** 2).sum(axis = 1).max())
        energy = material.get_potential_energy() / number_of_atoms
        if previous_energy is not None:
            energy_change = abs(energy - previous_energy)
        previous_energy = energy

        forces_converged = largest_force < FMAX
        energy_converged = ENERGY_TOLERANCE <= 0 or energy_change < ENERGY_TOLERANCE
        if forces_converged and energy_converged:
            converged = True
            break

    write(str(folder / "relaxed.cif"), material)
    return converged, optimiser.nsteps, energy_change

# Stage 2: force constant calculation

def force_constant_calculator(material, calc, supercell, displacement, folder):

    """
    Function that calculates the displacements of the cell in phonopy,
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
        supercell_matrix = supercell,
        # P as the primitive cell has already been found by get_primitive_cell function
        primitive_matrix = "P",
        symprec = SYMMETRY_TOLERANCE,
        is_symmetry = SYMMETRY,
        log_level = PHONOPY_LOG_LEVEL,
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
            cell = supercell_atoms.cell,
            scaled_positions = supercell_atoms.scaled_positions,
            pbc = True,
        )
        displaced.calc = calc
        forces.append(displaced.get_forces())

    phonon.forces = forces
    phonon.produce_force_constants(fc_calculator=FC_CALCULATOR)
    
    if FC_SYMMETRY:
        phonon.symmetrize_force_constants()    
    write_FORCE_SETS(phonon.dataset, filename = str(folder / FORCE_SETS_FILE)) 

    if WRITE_FORCE_CONSTANTS:
        if HDF5:
            write_force_constants_to_hdf5(phonon.force_constants, filename = str(folder / "force_constants.hdf5"))
        else:
            write_FORCE_CONSTANTS(phonon.force_constants, filename = str(folder / "FORCE_CONSTANTS"))
    return phonon
    

def load_run_phonon(run_folder):
    """
    Loads the saved phonons of one run folder (phonopy_disp.yaml and FORCE_SETS).
    The force constants are rebuilt with the settings the run was calculated with, read from
    its run_config.txt.
    Any of these settings missing from run_config.txt (or a run folder with no run_config.txt)
    take the current config's value, and a note says which.
    Returns None if the run folder has no saved phonons.
    """
    displacements_file = run_folder / DISPLACEMENTS_FILE
    force_sets_file = run_folder / FORCE_SETS_FILE
    if not (displacements_file.exists() and force_sets_file.exists()):
        return None
    # The settings this run was calculated with, from its saved copy of the config
    needed = ["SYMMETRY", "SYMMETRY_TOLERANCE", "FC_CALCULATOR", "FC_SYMMETRY"]
    saved_config = run_folder / RUN_CONFIG_FILE
    saved = read_config(saved_config, strict=False) if saved_config.exists() else {}
    # Checks if all settings have a value for a run - if not the guesses used are printed in terminal
    from_current = [name for name in needed if name not in saved]
    if from_current:
        print(f"note: {run_folder.parent.name}/{run_folder.name} has no saved value for "
              f"{', '.join(from_current)}, so the current config's value is used")
    settings = {name: saved.get(name, CONFIG[name]) for name in needed}

    return phonopy.load(
        str(displacements_file),
        force_sets_filename = str(force_sets_file),
        primitive_matrix = "P",
        is_symmetry = settings["SYMMETRY"],
        symprec = settings["SYMMETRY_TOLERANCE"],
        fc_calculator = settings["FC_CALCULATOR"],
        symmetrize_fc = settings["FC_SYMMETRY"],
        log_level = PHONOPY_LOG_LEVEL,
    )
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
    if HDF5:
        phonon.write_hdf5_band_structure(filename = str(folder / "band.hdf5"))
    else:
        phonon.write_yaml_band_structure(filename = str(folder / "band.yaml"))

def phonon_properties(phonon, folder):
    """
    Optional post-processing, switched on in the config:
      TPROP   thermal properties on the MESH q-point mesh, from TMIN to TMAX in steps of TSTEP (K),
              written to thermal_properties.yaml
      IRREPS  irreducible representations at a q-point, written to irreps.yaml
              (and printed if SHOW_IRREPS is True)
    Returns the list of files written.
    """
    written = []
    if TPROP:
        phonon.run_mesh(MESH)
        phonon.run_thermal_properties(t_min = TMIN, t_max = TMAX, t_step = TSTEP)
        thermal_file = folder / "thermal_properties.yaml"
        phonon.write_yaml_thermal_properties(filename = str(thermal_file))
        written.append(thermal_file.name)

    if IRREPS is not None:
        qpoint, tolerance = IRREPS
        phonon.set_irreps(qpoint, degeneracy_tolerance = tolerance)
        if SHOW_IRREPS:
            phonon.show_irreps(show_irreps = True)
        start_folder = os.getcwd()
        try:
            os.chdir(folder)
            phonon.write_yaml_irreps(show_irreps = SHOW_IRREPS)
        finally:
            os.chdir(start_folder)
        written.append("irreps.yaml")
    return written

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
    with open(csv_file, "a", newline = "") as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(CSV_COLUMNS)
        writer.writerow(row)


# Plots
def plot_bands(datasets, reference_phonon, band_path, band_labels, title, figure_file, reference_label="REF", styles = None):
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
        if styles is not None:
            style = styles[label]
        else:
            style = {"color": f"C{colour_number}", "linestyle": "-"}    # matplotlib's default colour cycle
        for section in range(len(x_axis)):
            lines = ax.plot(x_axis[section], bands["frequencies"][section], linewidth=1.0, **style)
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
        ax.axvline(position, color = "grey", linewidth = 0.5)
    ax.axhline(0, color = "grey", linewidth = 0.5)
    ax.set_xlim(tick_positions[0], tick_positions[-1])
    ax.set_ylabel("Frequency (THz)")
    ax.set_title(title)
    ax.legend()

    figure_file = Path(figure_file)
    figure_file.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(str(figure_file), dpi=200)
    plt.close(fig)
    return figure_file

def plot_run_bands(material_name, model_name, phonon, reference_phonon, band_path, band_labels, supercell, displacement, reference_label = "REF", colour = None):
    """
    One run (one model, supercell and displacement and symmetry on/off) with the reference underneath.
    Colour of MLIP is determined by dictionary in MLIP_parser
    Saves results/<material>/<model>/<run label>/band_structure.png
    """
    run_folder = get_run_folder(material_name, model_name, supercell, displacement)
    styles = {model_name: {"color": colour, "linestyle": "-"}} if colour is not None else None
    return plot_bands(
        {model_name: phonon}, reference_phonon, band_path, band_labels,
        title = f"{material_name}: {model_name}, {run_folder.name}",
        figure_file = run_folder / "band_structure.png",
        reference_label = reference_label,
        styles = styles,
    )

