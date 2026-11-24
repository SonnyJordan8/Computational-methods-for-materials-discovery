"""
Reads the workflow configuration file (config_file.txt) once.
The parser, processor and runner all take their settings from the config_file.txt

The file is plain text with one setting per line:   KEY = value
Everything after # on a line is a comment. True/false is written .TRUE. / .FALSE.

Phonopy settings use phonopy's own tag names and value formats, so the same file can be read by
phonopy's own reader. 
check_with_phonopy runs that reader on the file and stops if it reads any
phonopy setting differently. 
The workflow's own settings use names that are not phonopy tags.

"""
from pathlib import Path
import argparse
import numpy as np

# The folder this file is in. Folder settings in the config are relative to it.
CODE_FOLDER = Path(__file__).resolve().parent


# Readers: each turns the text after "=" into a Python value

def read_bool(text):
    """.TRUE. or .FALSE. (any capitals), the only forms phonopy reads. Phonopy reads anything else,
    such as True, as false without a warning, so anything else is an error here."""
    value = text.strip().upper()
    if value == ".TRUE.":
        return True
    if value == ".FALSE.":
        return False
    raise ValueError(f"write .TRUE. or .FALSE. (phonopy's format), got {text.strip()!r}")

def read_int(text):
    return int(text)

def read_float(text):
    return float(text)

def read_word(text):
    return text.strip()

def read_folder(text):
    """A folder: relative to the code folder, unless it is a full path."""
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
    raise ValueError(f"DIM needs 3 or 9 whole numbers, got {len(numbers)}")

def read_fraction(text):
    """A number, or a fraction such as 1/2."""
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
        raise ValueError(f"PRIMITIVE_AXES needs P, AUTO or 9 numbers, got {value!r}")
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
    """Makes a reader that accepts one word from options (any capitalisation)."""
    def reader(text):
        for option in options:
            if text.strip().lower() == option.lower():
                return option
        raise ValueError(f"must be one of {', '.join(options)}, got {text.strip()!r}")
    return reader

def read_mesh(text):
    """Three whole numbers, e.g. 20 20 20."""
    numbers = [int(n) for n in text.split()]
    if len(numbers) != 3:
        raise ValueError(f"MESH needs 3 whole numbers, got {len(numbers)}")
    return numbers

def read_irreps(text):
    """
    A q-point (3 numbers, fractions allowed) and optionally a degeneracy tolerance, e.g. 0 0 0 1e-4.
    Returned as (q-point, tolerance or None). To skip irreps, comment the line out: phonopy stops
    on an empty IRREPS line, so an empty value is an error here too.
    """
    parts = text.split()
    if len(parts) == 0:
        raise ValueError("IRREPS is empty - comment the line out instead (phonopy stops on an empty IRREPS)")
    if len(parts) not in (3, 4):
        raise ValueError(f"IRREPS needs a q-point (3 numbers) and optionally a tolerance, got {text.strip()!r}")
    qpoint = [read_fraction(n) for n in parts[:3]]
    tolerance = float(parts[3]) if len(parts) == 4 else None
    return qpoint, tolerance

def read_magmom(text):
    """MAGMOM: one number per atom (three for non-collinear), or empty for non-magnetic. Empty gives None."""
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
# IRREPS must be commented out when unused (phonopy stops on an empty IRREPS line),
# and MAGMOM only applies to magnetic systems.
OPTIONAL = {"IRREPS": None, "MAGMOM": None}

def read_config(path, strict = True):
    """
    Reads the config file at path. Returns a dictionary with every setting in SETTINGS,
    each turned into its Python type. Every setting must be in the file, except those in
    OPTIONAL. Any line it can't read, or any missing setting, raises an error.
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

    config["_IN_FILE"] = seen
    return config

# Settings that change the relaxed structure, the forces or the force constants. A run folder
# may only be reused by a run with the same values of all of these (see changed_settings).
# The others (run switches, plotting, post-processing, folders) can change between runs.
CALCULATION_SETTINGS = [
    "DTYPE",
    "OPTIMIZER", "RELAX_CELL", "RELAX_FMAX", "ENERGY_TOLERANCE", "MAX_STEPS", "RELAX_SYMMETRY",
    "DIM", "PRIMITIVE_AXES", "DISPLACEMENT_DISTANCE", "PM", "MAGMOM",
    "SYMMETRY", "SYMMETRY_TOLERANCE", "FC_CALCULATOR", "FC_SYMMETRY",
]

def changed_settings(saved_config_file):
    """
    Compares a run's saved copy of the config with the current CONFIG.
    Returns a list of the calculation settings that differ, as "KEY: saved -> now".
    An empty list means the run was made with the same calculation settings.
    """
    saved = read_config(saved_config_file, strict=False)
    return [f"{key}: {saved.get(key, 'not recorded')!r} -> {CONFIG[key]!r}"
            for key in CALCULATION_SETTINGS if saved.get(key, "not recorded") != CONFIG[key]]

# Phonopy setting tag -> (name of the same value in phonopy's own reader, how to compare it with ours)
PHONOPY_TAGS = {
    "DIM":                   "supercell_matrix",
    "PRIMITIVE_AXES":        "primitive_matrix",
    "DISPLACEMENT_DISTANCE": "displacement_distance",
    "PM":                    "is_plusminus_displacement",
    "SYMMETRY":              "is_symmetry",
    "SYMMETRY_TOLERANCE":    "symmetry_tolerance",
    "FC_CALCULATOR":         "fc_calculator",
    "FC_SYMMETRY":           "fc_symmetry",
    "BAND_POINTS":           "band_points",
    "MESH":                  "mesh_numbers",
    "TPROP":                 "is_thermal_properties",
    "TMIN":                  "min_temperature",
    "TMAX":                  "max_temperature",
    "TSTEP":                 "temperature_step",
    "SHOW_IRREPS":           "show_irreps",
    "HDF5":                  "is_hdf5",
    "WRITE_FORCE_CONSTANTS": "write_force_constants",
    "MAGMOM":                "magnetic_moments",
    "IRREPS":                "irreps_q_point",
}


def same_value(ours, phonopys):
    """True if our value and phonopy's value are the same setting (numbers compared to 1e-9)."""
    # Empty and None both mean "not set" (phonopy reads an empty MAGMOM as [])
    if ours is None and phonopys is not None and np.size(phonopys) == 0:
        return True
    if isinstance(ours, str) or isinstance(phonopys, str):
        return str(ours).lower() == str(phonopys).lower()
    if ours is None or phonopys is None:
        return ours is None and phonopys is None
    try:
        return np.allclose(np.array(ours, dtype=float), np.array(phonopys, dtype=float), atol=1e-9)
    except (TypeError, ValueError):
        return ours == phonopys


def check_with_phonopy(path, config):
    """
    Reads the config file with phonopy's own reader and compares every phonopy setting that is in
    the file with the value read here. Stops with a list of any differences, so the file always
    means the same to phonopy as to the workflow. Phonopy's reader lives in phonopy.cui (its
    command-line code), so if a future phonopy moves it, the check is skipped with a message.
    """
    try:
        from phonopy.cui.settings import PhonopyConfParser
    except ImportError:
        print("note: phonopy's config reader was not found, so the file was not checked against it")
        return
    try:
        phonopy_settings = PhonopyConfParser(filename=str(path)).settings
    except SystemExit:
        raise ValueError(f"phonopy's own reader could not read {Path(path).name} (see its message above)") from None
    except Exception as error:
        raise ValueError(f"phonopy's own reader could not read {Path(path).name}: {type(error).__name__}: {error}") from None

    differences = []
    for tag, attribute in PHONOPY_TAGS.items():
        if tag not in config["_IN_FILE"] or not hasattr(phonopy_settings, attribute):
            continue
        ours = config[tag]
        if tag == "IRREPS":
            ours = ours[0]                         # compare the q-point; phonopy keeps the tolerance separately
        phonopys = getattr(phonopy_settings, attribute)
        if not same_value(ours, phonopys):
            differences.append(f"{tag}: workflow reads {ours!r}, phonopy reads {phonopys!r}")
    if differences:
        raise ValueError(f"{Path(path).name} is read differently by phonopy:\n  " + "\n  ".join(differences))


def supercell_label(dim):
    """
    Short text for a supercell matrix, for folder names and tables.
    Diagonal [[2,0,0],[0,2,0],[0,0,4]] -> "2x2x4"; any other matrix -> "m" + its 9 numbers.
    """
    matrix = np.array(dim, dtype=int)
    if np.count_nonzero(matrix - np.diag(np.diagonal(matrix))) == 0:
        return "x".join(str(n) for n in np.diagonal(matrix))
    return "m" + "_".join(str(n) for n in matrix.flatten())


def find_config_file():
    """
    The config file to read: given on the command line with --config, e.g.
    python MLIP_runner.py --config my_settings.txt
    otherwise config_file.txt in the code folder. parse_known_args ignores
    arguments meant for something else (e.g. Jupyter passes -f <file>).
    """
    parser = argparse.ArgumentParser(description="uMLIP phonon benchmarking workflow")
    parser.add_argument("--config", default=str(CODE_FOLDER / "config_file.txt"),
                        help="settings file to read (default: config_file.txt in the code folder)")
    arguments, _ = parser.parse_known_args()
    return Path(arguments.config).resolve()


# Read once, when the module is first imported; every other module uses these
CONFIG_FILE = find_config_file()
CONFIG = read_config(CONFIG_FILE)
check_with_phonopy(CONFIG_FILE, CONFIG)
if CONFIG["MAGMOM"] is not None:
    print("note: MAGMOM is set, but the workflow does not use it yet")
