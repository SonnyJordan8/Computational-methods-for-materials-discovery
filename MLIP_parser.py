from pathlib import Path
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import phonopy
from monty.serialization import loadfn


MATERIALS_CIF_FOLDER = Path(__file__).resolve().parent / "materials_cif"
REFERENCE_PHONOPY_PARAMS_FOLDER = Path(__file__).resolve().parent / "reference_data"

# Kyoto (Togo) phonon database, set up with Kyoto_database_extractor.ipynb
PHONON_DB_FOLDER = Path(__file__).resolve().parent / "phononDB"

# Molecule to constituent atoms mapper
molecules = {"Si": {"Si"}, "GaAs": {"Ga", "As"}}

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

def phonon_db_finder(material_name, structure, lookup, mp_ids):

    if material_name in mp_ids:  # This is set in runner file where mp_ids matched with material is hard coded
        mp_id = mp_ids[material_name]
        if mp_id in set(lookup["material_id"]): # this is searching material_id column of lookup df 
            return mp_id
        else:
            print(f"{material_name}: {mp_id} is not in phononDB")
        return None

    # Checking the loaded phonon_db data matches the structure of the parsed cif files. 
    matches = lookup[
        (lookup["formula_pretty_reduced"] == structure.info["reduced_formula"])
        & 
        (lookup["spacegroup_number"] == structure.info["space_group_number"])
        ]
    if len(matches) == 1:
        return matches["material_id"].iloc[0]
    if len(matches) > 1:
        # e.g. two polymorphs with the same space group - add the right one to REFERENCE_MP_IDS
        print(f"  {material_name}: more than one phononDB match {list(matches['material_id'])}, "
              f"add the right one to REFERENCE_MP_IDS")
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

def reference_data_parser(materials, phonon_db_folder, reference_folder, mp_ids = None):
    """
    Parses the reference phonopy files in the given folder (reference_data).
    Returns a dictionary of Phonopy objects, one for each material.
    The key is the material name, e.g. "Si_phonopy_params.yaml" -> "Si",
    so it matches the key from cif_parser.

    """
    lookup_file = phonon_db_folder / "phonon_db_lookup_df.json"
    lookup = None

    if mp_ids is None:
        mp_ids = {}
    
    if lookup_file.exists():
        lookup = loadfn(lookup_file)
    else:
        print(f"phononDB lookup table not found at {lookup_file}, using {reference_folder} only")

    
    references = {}
    sources = {}
    for material_name, structure in materials.items():
        # 1. phononDB
        if lookup is not None:
            mp_id = phonon_db_finder(material_name, structure, lookup, mp_ids)
            if mp_id is not None:
                references[material_name] = phonon_db_parser(phonon_db_folder, lookup, mp_id)
                sources[material_name] = f"phononDB {mp_id}"
                
                continue

        # 2. reference folder
        params_file = reference_folder / f"{material_name}_phonopy_params.yaml"
        if params_file.exists():
            references[material_name] = phonopy.load(str(params_file))
            sources[material_name] = params_file.name
            continue

        # 3. no reference data
        sources[material_name] = "none"

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
  return KairosCalculator(model_path=str(PROPHET_WEIGHTS), use_kernel=False, device=device)





# Model name -> function that builds its calculator
MODELS = {
    "MACE-MP-0-medium": make_mace,
    "Prophet-OAME-MBD": make_prophet,
}