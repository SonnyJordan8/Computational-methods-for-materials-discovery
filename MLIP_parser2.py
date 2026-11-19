from pathlib import Path
import json
import numpy as np
from ase.io import read
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import phonopy
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.file_IO import parse_FORCE_CONSTANTS
from monty.serialization import loadfn


MATERIALS_CIF_FOLDER = Path(__file__).resolve().parent / "materials_cif"

# All reference data lives in Reference_data, one folder per source
REFERENCE_DATA_FOLDER = Path(__file__).resolve().parent / "Reference_data"
# Kyoto (Togo) phonon database, set up with Kyoto_database_extractor.ipynb
PHONON_DB_FOLDER = REFERENCE_DATA_FOLDER / "phononDB"
# Materials Project phonons, set up with MaterialsProject_extractor.ipynb
MP_REFERENCE_FOLDER = REFERENCE_DATA_FOLDER / "materials_project_reference_data"

def cif_parser(folder_path):
    """
    Parses every CIF file in the given folder (materials_cif).
    Returns a dictionary of ASE Atoms objects, one for each material.
    The key is the file name without ".cif", e.g. "Si.cif" -> "Si".
    The space group symbol and number are stored in the Atoms info dictionary.
    
    """
    materials = {}
    for file in folder_path.glob("*.cif"):
        structure = read(file, format="cif")
        structure_pmg = AseAtomsAdaptor.get_structure(structure)
        finder = SpacegroupAnalyzer(structure_pmg)

        sg_symbol = finder.get_space_group_symbol()
        sg_number = finder.get_space_group_number()

        structure.info["space_group_symbol"] = sg_symbol
        structure.info["space_group_number"] = sg_number
        structure.info["reduced_formula"] = structure_pmg.composition.reduced_formula
        materials[file.stem] = structure

    if not materials:
        raise FileNotFoundError(f"No CIF files found in {folder_path}")

    return materials

# Reference data parsers
# Two sources, each in its own folder in Reference_data:
#   phononDB  - phonon_db_lookup_df.json lists every material; each has phonon.yaml + FORCE_SETS
#   MP        - phonon_mp_lookup.json gives mp-id -> material name; each material has
#               force_constants/<name>_force_constants and phonopy_settings/<name>_phonopy_settings.json

SOURCES = ("phononDB", "MP")

def phonon_db_lookup(phonon_db_folder):
    """
    Loads the phononDB lookup table, or returns None if it is not there.

    """
    lookup_file = phonon_db_folder / "phonon_db_lookup_df.json"
    if not lookup_file.exists():
        print(f"phononDB lookup table not found at {lookup_file}")
        return None
    return loadfn(lookup_file)

def mp_lookup(mp_folder):
    """
    Loads the Materials Project lookup file (mp-id -> material name), or returns {} if it is not there.

    """
    lookup_file = mp_folder / "phonon_mp_lookup.json"
    if not lookup_file.exists():
        print(f"Materials Project lookup file not found at {lookup_file}")
        return {}
    with open(lookup_file) as file:
        return json.load(file)

def phonon_db_finder(material_name, structure, lookup):
    """
    Finds the phononDB entry of a material automatically, by formula and space group.
    Returns the mp-id, or None if there is no single match.

    """
    matches = lookup[
        (lookup["formula_pretty_reduced"] == structure.info["reduced_formula"])
        &
        (lookup["spacegroup_number"] == structure.info["space_group_number"])
        ]
    if len(matches) == 1:
        return matches["material_id"].iloc[0]
    if len(matches) > 1:
        # e.g. two polymorphs with the same space group - add the right one to REFERENCES
        print(f"  {material_name}: more than one phononDB match {list(matches['material_id'])}, "
              f"add the right one to REFERENCES")
    return None

def phonon_db_parser(phonon_db_folder, lookup, mp_id):
    """
    Parses one material from phononDB as a Phonopy object, from its phonon.yaml and FORCE_SETS.
    The non-analytical correction (BORN) is not used - can be changed for future calculations
    """
    row = lookup[lookup["material_id"] == mp_id].iloc[0]
    material_phonons = phonon_db_folder / "phonon_db" / Path(row["phonon_filepath"]).parent.name
    return phonopy.load(
        str(material_phonons / "phonon.yaml"),
        force_sets_filename=str(material_phonons / "FORCE_SETS"),
        primitive_matrix="auto",
        is_nac=False
    )

def mp_parser(mp_folder, mp_name):
    """
    Parses one material from the Materials Project folder as a Phonopy object.
    The unit cell, supercell matrix and primitive matrix come from <name>_phonopy_settings.json,
    the force constants from <name>_force_constants (phonopy FORCE_CONSTANTS format).
    The non-analytical correction is not used yet, the same as phononDB.

    """
    settings_file = mp_folder / "phonopy_settings" / f"{mp_name}_phonopy_settings.json"
    force_constants_file = mp_folder / "force_constants" / f"{mp_name}_force_constants"
    for file in (settings_file, force_constants_file):
        if not file.exists():
            raise FileNotFoundError(f"{file} not found - run MaterialsProject_extractor.ipynb for {mp_name}")

    with open(settings_file) as file:
        settings = json.load(file)

    structure = Structure.from_dict(settings["structure"])
    unit_cell = PhonopyAtoms(
        symbols=[site.specie.symbol for site in structure],
        cell=structure.lattice.matrix,
        scaled_positions=structure.frac_coords,
    )
    # "P" means the unit cell is already the primitive cell
    primitive_matrix = settings.get("primitive_matrix") or "P"
    phonon = Phonopy(unit_cell, supercell_matrix=settings["supercell_matrix"], primitive_matrix=primitive_matrix)

    force_constants = parse_FORCE_CONSTANTS(str(force_constants_file))
    supercell_atoms = len(phonon.supercell)
    if force_constants.shape[1] != supercell_atoms:
        raise ValueError(f"{force_constants_file.name} is for {force_constants.shape[1]} atoms, "
                         f"but the supercell in {settings_file.name} has {supercell_atoms}")
    phonon.force_constants = force_constants
    return phonon

def find_reference(material_name, request, phonon_db_ids, mp_ids):
    """
    Works out which source an entry of REFERENCES points to.
    "request" is either an mp-id, e.g. "mp-149", which is looked for in both folders,
    or a (source, mp-id) pair, e.g. ("MP", "mp-19907"), which only looks in that source.
    Returns (source, mp-id), or None if it is not found.

    """
    available = {"phononDB": phonon_db_ids, "MP": mp_ids}

    if isinstance(request, (tuple, list)):
        source, mp_id = request
        if source not in SOURCES:
            raise ValueError(f"{material_name}: source must be one of {SOURCES}, not {source!r}")
        if mp_id in available[source]:
            return source, mp_id
        print(f"  {material_name}: {mp_id} is not in the {source} folder")
        return None

    mp_id = request
    found_in = [source for source in SOURCES if mp_id in available[source]]
    if len(found_in) == 0:
        print(f"  {material_name}: {mp_id} is not in phononDB or the MP folder")
        return None
    if len(found_in) > 1:
        print(f"  {material_name}: {mp_id} is in both {found_in}, using {found_in[0]}. "
              f"Write it as (\"MP\", \"{mp_id}\") in REFERENCES to use the other one")
    return found_in[0], mp_id

def check_reference_cell(material_name, structure, phonon):
    """
    The band path is made from the primitive cell of the CIF, but the reference bands are
    calculated in the reference's own primitive cell. Warns if the two cells look different,
    because then the same q-point labels would mean different points.

    """
    cif_primitive = SpacegroupAnalyzer(AseAtomsAdaptor.get_structure(structure)).get_primitive_standard_structure()
    reference_cell = np.array(phonon.primitive.cell)
    reference_lengths = np.linalg.norm(reference_cell, axis=1)
    reference_angles = [
        np.degrees(np.arccos(np.dot(reference_cell[i], reference_cell[j]) / (reference_lengths[i] * reference_lengths[j])))
        for i, j in ((1, 2), (0, 2), (0, 1))
    ]
    length_diff = np.abs(np.array(cif_primitive.lattice.abc) / reference_lengths - 1).max() * 100
    angle_diff = np.abs(np.array(cif_primitive.lattice.angles) - reference_angles).max()
    if length_diff > 5 or angle_diff > 2:
        print(f"  WARNING {material_name}: the reference primitive cell does not match the CIF primitive cell "
              f"(lengths differ by up to {length_diff:.1f} %, angles by up to {angle_diff:.1f} deg). "
              f"Check the band path labels before comparing.")

def reference_data_parser(materials, phonon_db_folder=PHONON_DB_FOLDER, mp_folder=MP_REFERENCE_FOLDER, requested=None):
    """
    Finds and parses the reference phonons for every material.
    "requested" is the REFERENCES dictionary from the runner: material name -> mp-id or (source, mp-id).
    Materials not in it are matched in phononDB automatically by formula and space group.
    Returns two dictionaries keyed by material name:
    references - Phonopy objects
    sources    - where each came from, e.g. "phononDB mp-149", "MP mp-19907" or "none"

    """
    if requested is None:
        requested = {}

    db_lookup = phonon_db_lookup(phonon_db_folder)
    phonon_db_ids = set(db_lookup["material_id"]) if db_lookup is not None else set()
    mp_names = mp_lookup(mp_folder)     # mp-id -> {"material": name}

    references = {}
    sources = {}
    for material_name, structure in materials.items():
        found = None
        if material_name in requested:
            found = find_reference(material_name, requested[material_name], phonon_db_ids, mp_names)
        elif db_lookup is not None:
            mp_id = phonon_db_finder(material_name, structure, db_lookup)
            if mp_id is not None:
                found = ("phononDB", mp_id)

        if found is None:
            sources[material_name] = "none"
            if material_name not in requested:
                print(f"  {material_name}: no phononDB match for {structure.info['reduced_formula']} "
                      f"space group {structure.info['space_group_number']} - add it to REFERENCES if there is one")
            continue

        source, mp_id = found
        try:
            if source == "phononDB":
                phonon = phonon_db_parser(phonon_db_folder, db_lookup, mp_id)
            else:
                phonon = mp_parser(mp_folder, mp_names[mp_id]["material"])
        except Exception as error:
            print(f"  {material_name}: could not load {source} {mp_id}: {type(error).__name__}: {error}")
            sources[material_name] = "none"
            continue

        check_reference_cell(material_name, structure, phonon)
        references[material_name] = phonon
        sources[material_name] = f"{source} {mp_id}"

    for material_name, source in sources.items():
        print(f"  reference for {material_name}: {source}")

    return references, sources


# MLIP Functions

def make_mace(device):
   """
   Creates a MACE-MP-0 medium calculator for the given device GPU or CPU.
   
   """
   try:
       from mace.calculators import mace_mp
   except ImportError:
       raise ImportError("MACE is not installed. Please check you are working in the right conda environment.")

   return mace_mp(model="medium", default_dtype="float64", device=device)


def make_prophet(device):
  
  """ 
  Creates a Prophet-OAME-MBD calculator for the given device (GPU or CPU)
  
  """
  PROPHET_WEIGHTS = Path(__file__).resolve().parent / "weights" / "prophet-oame-mbd.pt"
  PROPHET_URL = "https://huggingface.co/kairosmaterial/prophet/resolve/f9a54df874ba52b8b35e7d3f7f348b46e0135563/prophet-oame-mbd.pt" # https://creativecommons.org/licenses/by/4.0/legalcode
  if not PROPHET_WEIGHTS.exists():
    raise FileNotFoundError(f"Prophet weights not found at {PROPHET_WEIGHTS}. Download them from {PROPHET_URL}")

# KairosCalculator - the ASE calculator for Prophet
  
  try:
      from prophet import KairosCalculator
  except ImportError:
      raise ImportError("Prophet is not installed. Please check you are working in the right conda environment.") 
  
# use_kernel=False skips Prophet's GPU kernel, so it also runs on CPU - Prophet README
  return KairosCalculator(model_path = str(PROPHET_WEIGHTS), use_kernel = False, device = device)

def make_sevennet(device):
   """
   Creates a SevenNet-Omni-i12 calculator for a given device (CPU or GPU)

   """ 
   try:
       from sevenn.calculator import SevenNetCalculator
    
   except ImportError:
        raise ImportError("SevenNet is not installed. Please check you are working in the right conda environment.")
    
   return SevenNetCalculator(model = "7net-omni-i12", modal = "mpa", device = device)



# Model name -> function that builds its calculator
MODELS = {
    "MACE-MP-0-medium": make_mace,
    "Prophet-OAME-MBD": make_prophet,
    "SevenNet-Omni-i12": make_sevennet,
}
