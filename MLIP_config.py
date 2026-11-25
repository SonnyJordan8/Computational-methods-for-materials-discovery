"""
Reads the workflow configuration file config_file.txt.
config_file.txt sets the settings and material and MLIP used by parer, processer and runner

The file is text with one setting per line:   KEY = value

Phonopy settings use phonopy's own tag names and value formats enabling phonopy to read its own settings 
in the command line. 

"""
from pathlib import Path
import argparse
import numpy as np

# The folder this file is in. Folder settings in the config are relative to it.
CODE_FOLDER = Path(__file__).resolve().parent


# Readers
# each turns the text after = into a Python value

def read_bool(text):
    """.TRUE. or .FALSE. as phonopy reads"""
    value = text.strip().upper()
    if value == ".TRUE.":
        return True
    if value == ".FALSE.":
        return False
    raise ValueError(f"write .TRUE. or .FALSE. (phonopy's format), {text.strip()!r} was written")

def read_int(text):
    return int(text)

def read_float(text):
    return float(text)

def read_word(text):
    return text.strip()

def read_folder(text):
    """A folder either relative to the code folder or a full path."""
    path = Path(text.strip())
    return path if path.is_absolute() else CODE_FOLDER / path

def read_names(text):
    """
    Space-separated material names are put into a list from text line in config file. 
    ALL means every CIF in the CIF folder (returned as None by runner file).
    """
    names = text.split()
    if len(names) == 1 and names[0].upper() == "ALL":
        return None
    return names

def read_dim(text):
    """
    Supercell: 3 numbers (2 2 4) are the diagonal of the supercell matrix,
    or 9 numbers give the whole 3x3 matrix, row by row. Returned as a 3x3 list.
    """
    numbers = [int(n) for n in text.split()] 
    if len(numbers) == 3:
        return np.diag(numbers).tolist()
    if len(numbers) == 9:
        return np.array(numbers).reshape(3, 3).tolist()
    raise ValueError(f"DIM needs 3 or 9 whole numbers, {len(numbers)} was written.")

def read_fraction(text):
    """A fraction such as 1/2."""
    if "/" in text:
        top, bottom = text.split("/")
        return float(top) / float(bottom)
    return float(text)

def read_primitive_axes(text):
    """P (the cell is already primitive), AUTO (phonopy finds it), or 9 numbers for a 3x3 matrix."""
    value = text.strip()
    if value.upper() == "P":
        return "P"
    if value.upper() == "AUTO":
        return "auto"
    numbers = [read_fraction(n) for n in value.split()]
    if len(numbers) != 9:
        raise ValueError(f"PRIMITIVE_AXES needs P, AUTO or 9 numbers, {value!r} was written")
    return np.array(numbers).reshape(3, 3).tolist()

def read_plusminus(text):
    """True, False, or AUTO (phonopy decides from the symmetry)."""
    if text.strip().upper() == "AUTO":
        return "auto"
    return read_bool(text)

def read_fc_calculator(text):
    value = text.strip().lower()
    if value not in ("traditional", "symfc", "alm"):
        raise ValueError(f"FC_CALCULATOR must be traditional, symfc or alm, got {text!r}")
    return value

def read_choice(options):
    """Makes a reader that accepts one word from options regardless of capitalisation."""
    def reader(text):
        for option in options:
            if text.strip().lower() == option.lower():
                return option
        raise ValueError(f"must be one of {', '.join(options)}, got {text.strip()!r}")
    return reader

def read_mesh(text):
    """Three whole numbers."""
    numbers = [int(n) for n in text.split()]
    if len(numbers) != 3:
        raise ValueError(f"MESH needs 3 whole numbers, got {len(numbers)}")
    return numbers

def read_irreps(text):
    """
    A q-point (3 numbers) and optionally a degeneracy tolerance.
    Returned as (q-point, tolerance or None). 
    To skip irreps, comment the line out.
    """
    parts = text.split()
    if len(parts) == 0:
        raise ValueError("IRREPS is empty - comment the line out instead")
    if len(parts) not in (3, 4):
        raise ValueError(f"IRREPS needs a q-point (3 numbers) and optionally a tolerance, got {text.strip()!r}")
    qpoint = [read_fraction(n) for n in parts[:3]]
    tolerance = float(parts[3]) if len(parts) == 4 else None
    return qpoint, tolerance

def read_magmom(text):
    """MAGMOM: one number per atom, three per atom for non-collinear
    or empty for non-magnetic. Empty gives None."""
    numbers = [float(n) for n in text.split()]
    return numbers if numbers else None

def read_log_level(text):
    level = int(text)
    if level not in (0, 1, 2):
        raise ValueError(f"PHONOPY_LOG_LEVEL must be 0, 1 or 2, got {level}")
    return level

def read_references(text):
    """
    Comma-separated entries, each   <material> <mp-id>   or   <material> <source> <mp-id>
    e.g.   Si mp-149, alpha-3R-In2Se3 MP mp-19907
    Returned as {material: mp-id} or {material: (source, mp-id)}. Empty gives {}.
    """
    references = {}
    for entry in text.split(","):
        parts = entry.split()
        if len(parts) == 0:
            continue
        if len(parts) == 2:
            references[parts[0]] = parts[1]
        elif len(parts) == 3:
            references[parts[0]] = (parts[1], parts[2])
        else:
            raise ValueError(f"each REFERENCES entry needs '<material> <mp-id>' or "
                             f"'<material> <source> <mp-id>', got {entry.strip()!r}")
    return references


# Every setting: name -> (reader, default). The defaults are the values the code used before
# the config file existed. A name in the file that is not listed here is an error, so typos
# are caught instead of ignored.
SETTINGS = {
    # run
    "MATERIALS":                 read_names,
    "MODEL":                     read_word,
    "REFERENCES":                read_references,
    "DEVICE":                    read_word,
    "DTYPE":                     read_choice(["float64", "float32"]),
    "RUN_CALCULATIONS":          read_bool,
    "RUN_PLOTS_RUNS":            read_bool,
    "IMAGINARY_LIMIT":           read_float,
    # ASE relaxation
    "OPTIMIZER":                 read_choice(["FIRE", "BFGS", "LBFGS", "BFGSLineSearch"]),
    "RELAX_CELL":                read_bool,
    "RELAX_FMAX":                read_float,
    "ENERGY_TOLERANCE":          read_float,
    "MAX_STEPS":                 read_int,
    # phonopy pre-process
    "DIM":                       read_dim,
    "PRIMITIVE_AXES":            read_primitive_axes,
    "PRIMITIVE_TOLERANCE":       read_float,
    "DISPLACEMENT_DISTANCE":     read_float,
    "PM":                        read_plusminus,
    "MAGMOM":                    read_magmom,
    "PHONOPY_LOG_LEVEL":         read_log_level,
    # phonopy post-process
    "BAND_POINTS":               read_int,
    "MESH":                      read_mesh,
    "TPROP":                     read_bool,
    "TMIN":                      read_float,
    "TMAX":                      read_float,
    "TSTEP":                     read_float,
    "IRREPS":                    read_irreps,
    "SHOW_IRREPS":               read_bool,
    "HDF5":                      read_bool,
    "WRITE_FORCE_CONSTANTS":     read_bool,
    "RESULTS_FOLDER":            read_folder,
    "DISPLACEMENTS_FILE":        read_word,
    "FORCE_SETS_FILE":           read_word,
    # input data folders
    "MATERIALS_CIF_FOLDER":      read_folder,
    "PHONON_DB_FOLDER":          read_folder,
    "MP_REFERENCE_FOLDER":       read_folder,
    # symmetry and force constants
    "SYMMETRY":                  read_bool,
    "SYMMETRY_TOLERANCE":        read_float,
    "RELAX_SYMMETRY":            read_bool,
    "FC_CALCULATOR":             read_fc_calculator,
    "FC_SYMMETRY":               read_bool,
}

# Settings that may be left out of the file, and the value they take when they are:
OPTIONAL = {"IRREPS": None, "MAGMOM": None}

def read_config(path, strict = True):
    """
    Reads the config file at path. 
    Returns the CONFIG dictionary with every setting in SETTINGS.
    This is the dictionary read by the workflow for run settings.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    config = dict(OPTIONAL)  
    seen = set()               
    with open(path) as file:
        for line_number, line in enumerate(file, start=1):
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            if "=" not in text:
                raise ValueError(f"{path.name} line {line_number}: expected KEY = value, got {text!r}")
            key, value = text.split("=", 1)
            key = key.strip().upper()
            try:
                if key not in SETTINGS:
                    if not strict:
                        continue
                    raise ValueError(f"unknown setting {key}")
                if key in seen:
                    raise ValueError(f"{key} is set more than once")
                seen.add(key)
                config[key] = SETTINGS[key](value) 
            except ValueError as error:
                raise ValueError(f"{path.name} line {line_number}: {error}") from None

    missing = [name for name in SETTINGS if name not in seen and name not in OPTIONAL]
    if missing and strict:
        raise ValueError(f"{path.name} is missing these settings: {', '.join(missing)}")
    return config

# Settings that change the relaxed structure, the forces or the force constants. A run folder
# may only be reused by a run with the same values of all of these.
CALCULATION_SETTINGS = [
    "DTYPE",
    "OPTIMIZER", "RELAX_CELL", "RELAX_FMAX", "ENERGY_TOLERANCE", "MAX_STEPS", "RELAX_SYMMETRY",
    "DIM", "PRIMITIVE_AXES", "DISPLACEMENT_DISTANCE", "PM", "MAGMOM",
    "SYMMETRY", "SYMMETRY_TOLERANCE", "FC_CALCULATOR", "FC_SYMMETRY",
]

def changed_settings(saved_config_file):
    """
    Compares a run's saved copy of the config with the current CONFIG.
    This protectrs runs from being overwritten that have the same folder name
    but differ in config_file settings.
    """
    saved = read_config(saved_config_file, strict=False)
    return [f"{key}: {saved.get(key, 'not recorded')!r} -> {CONFIG[key]!r}"
            for key in CALCULATION_SETTINGS if saved.get(key, "not recorded") != CONFIG[key]]

def supercell_label(dim):
    """
    Short text for a supercell .
    Diagonal [[2,0,0],[0,2,0],[0,0,4]] - "2x2x4";
    any other matrix - "m" + its 9 numbers.
    """
    matrix = np.array(dim, dtype=int)
    if np.count_nonzero(matrix - np.diag(np.diagonal(matrix))) == 0:
        return "x".join(str(n) for n in np.diagonal(matrix))
    return "m" + "_".join(str(n) for n in matrix.flatten())


# Used to easily submit multiple jobs to HPC
def find_config_file():
    """
    The config file to read on the command line with --config, e.g.
    python MLIP_runner.py --config my_settings.txt
    """
    parser = argparse.ArgumentParser(description="uMLIP phonon benchmarking workflow")
    parser.add_argument("--config", default=str(CODE_FOLDER / "config_file.txt"),
                        help="settings file to read (default: config_file.txt in the code folder)")
    arguments, _ = parser.parse_known_args()
    return Path(arguments.config).resolve()


# Read once, when the module is first imported; every other module uses these
CONFIG_FILE = find_config_file()
CONFIG = read_config(CONFIG_FILE)
if CONFIG["MAGMOM"] is not None:
    print("note: MAGMOM is set, but the workflow does not use it yet")
