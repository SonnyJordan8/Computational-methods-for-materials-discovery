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

# All starting material CIF files live here
MATERIALS_CIF_FOLDER = Path(__file__).resolve().parent / "materials_cif"
# All reference data is stored here - different sources have different folders within this folder
REFERENCE_DATA_FOLDER = Path(__file__).resolve().parent / "reference_data"
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

# Reference data parser
# To be ammended if / when more reference data sources are used
# Currently two sources of reference data - both live in reference_data folder

SOURCES = ("phononDB", "MP")
def phonon_db_lookup(phonon_db_folder):
    """
    Loads in the phononDB lookup file.
    Reference phonon calculations from PhononDB (Kyoto University)
    """
    lookup_file = phonon_db_folder / "phonon_db_lookup_df.json"
    if not lookup_file.exists():
        print(f"phononDB lookup table not found at {lookup_file}")
        return None
    try:
        lookup = loadfn(lookup_file)   
        lookup["material_id"]          
        return lookup
    except Exception as error:
        print(f"could not read {lookup_file.name} ({type(error).__name__}: {error}) - phononDB skipped")
        return None

def mp_lookup(mp_folder):
    """
    Loads in the Materials Project lookup .JSON file.
    Structure of the file is: "mp_ID": {"material": "material_formula"}.
    """
    lookup_file = mp_folder / "phonon_mp_lookup.json"
    if not lookup_file.exists():
        print(f"Materials Project lookup file not found at {lookup_file}")
        return {}
    try:
        lookup = loadfn(lookup_file)     # reads plain JSON too, and a monty-saved DataFrame if it is one
        return lookup
    except Exception as error:
        print(f"could not read {lookup_file.name} ({type(error).__name__}: {error}) - MP references skipped")
        return {}


def phonon_db_finder(material_name, structure, lookup):
    """
    Finds the phononDB reference data for a material automatically.
    Found by space-group and formula, returns mp-id if there is a match between phononDB and parsed CIF structure.
    """
    # Formula and space group match

    matches = []
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

def mp_finder(material_name, structure, mp_folder, mp_names):
    """
    Finds the Materials Project reference data for a material if no file is called as a reference in config file (YET TO BE ADDED)
    It is found by the name of the CIF file -> Should match the {material name}-phonondata.JSON structure of reference MP data.
    Also found by the formula and space group of the parsed CIF file against the structure and space group of the reference file.
    which is stored in {material name}_phonopy_settings.JSON file.   
    Returns the mp-id or None if there is no single match.
    """
    # Name match
    for mp_id, entry in mp_names.items():
        if entry["material"] == material_name:
            return mp_id

    # Formula and space group match
    matches = []
    for mp_id, entry in mp_names.items():
        settings_file = mp_folder / "phonopy_settings" / f"{entry['material']}_phonopy_settings.json"
        if not settings_file.exists():
            continue
        try:
            with open(settings_file) as file:
                mp_structure = Structure.from_dict(json.load(file)["structure"])
            same_formula = mp_structure.composition.reduced_formula == structure.info["reduced_formula"]
            same_space_group = SpacegroupAnalyzer(mp_structure).get_space_group_number() == structure.info["space_group_number"]
        except Exception as e:
            print(f"  could not read {settings_file.name}: {e})")
            continue
        if same_formula and same_space_group:
            matches.append(mp_id)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # e.g. the same material downloaded twice under different names - add the right one to REFERENCES
        print(f"{material_name}: more than one MP match {matches}, add the right one to REFERENCES")
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
    Parse data from the Materials Project reference data folder as a Phonopy object.
    The unit cell, supercell matric and primitive matrix of the material come from the 
    {material name}_phonopy_settings.JSON file in the phonopy_settings folder in the 
    materials_projecy_reference_data folder. The band data, DOS and force constant data
    come from the relevant folders.

    """
    settings_file = mp_folder / "structure" / f"{mp_name}.json"
    force_constants_file = mp_folder / "force_constants" / f"{mp_name}_force_constants"
    dos_file = mp_folder / "dos" / f"{mp_name}_phonondos"
    band_structure_file = mp_folder / "band_structure" / f"{mp_name}_phonondata"

    # Functionality to increase to DOS in future
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

    # This will be changed when config file is added
    primitive_matrix = settings.get("primitive_matrix") or "auto"
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
    """
    # To be increased if more reference sources found
    available = {"phononDB": phonon_db_ids, "MP": mp_ids}

    # Checking the entry is correctly formatted and that the database exists 
    if isinstance(request, (tuple, list)):
        source, mp_id = request
        if source not in SOURCES:
            raise ValueError(f"{material_name}: source must be one of {SOURCES}, not {source!r}")
        if mp_id in available[source]:
            return source, mp_id
        print(f"{material_name}: {mp_id} is not in the {source} folder")
        return None

    mp_id = request
    found_in = [source for source in SOURCES if mp_id in available[source]]
    if len(found_in) == 0:
        print(f"{material_name}: {mp_id} is not in phononDB or the MP folder")
        return None
    if len(found_in) > 1:
        print(f"{material_name}: {mp_id} is in both {found_in}, using {found_in[0]}. ")
    return found_in[0], mp_id

# Need to double check this is necessary

def check_reference_cell(material_name, structure, phonon):
    """
    The band path is made from the primitive cell of the CIF, but the reference bands are
    calculated in the reference's own primitive cell. Warns if the two cells look different,
    because then the same k-point labels would mean different points.

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
        print(f" WARNING {material_name}: the reference primitive cell does not match the CIF primitive cell "
              f"(lengths differ by up to {length_diff:.1f} %, angles by up to {angle_diff:.1f} deg). "
              f"Check the band path labels before comparing.")

def reference_data_parser(materials, phonon_db_folder=PHONON_DB_FOLDER, mp_folder=MP_REFERENCE_FOLDER, requested=None):
    """
    Finds and parses the reference phonons for every material.
    "requested" is the REFERENCES dictionary from the runner (soon to be config file)
    Materials not in it are matched automatically
    Returns two dictionaries keyed by material name:
    references - Phonopy objects
    sources    - where each came from, e.g. "phononDB mp-149", "MP mp-19907" or "none"

    """
    if requested is None:
        requested = {}

    db_lookup = phonon_db_lookup(phonon_db_folder)
    phonon_db_ids = set(db_lookup["material_id"]) if db_lookup is not None else set()
    mp_names = mp_lookup(mp_folder)     

    references = {}
    sources = {}
    for material_name, structure in materials.items():
        found = None
        if material_name in requested:
            found = find_reference(material_name, requested[material_name], phonon_db_ids, mp_names)
        else:
            # Automatic matching: the MP folder first, then phononDB
            mp_id = mp_finder(material_name, structure, mp_folder, mp_names)
            if mp_id is not None:
                found = ("MP", mp_id)
            elif db_lookup is not None:
                mp_id = phonon_db_finder(material_name, structure, db_lookup)
                if mp_id is not None:
                    found = ("phononDB", mp_id)

        if found is None:
            sources[material_name] = "none"
            if material_name not in requested:
                print(f"  {material_name}: no MP or phononDB match for {structure.info['reduced_formula']} "
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
