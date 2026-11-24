# MLIP_parser.py

**Role in the workflow:** gets everything *into* the program. It reads the crystal structures (CIF files), finds the reference phonon data for each material, and builds the MLIP calculators. It does no physics itself.

It is imported by `MLIP_runner.py` and `MLIP_plotter.py`. It imports `MLIP_config.py` (for its folders and `DTYPE`) but not `MLIP_processor.py`.

```
materials_cif/*.cif ──► cif_parser ──────────────► materials (dict of ASE Atoms)
                                                        │
reference_data/phononDB/ ─────────────┐                 ▼
reference_data/materials_project_     ├──► reference_data_parser ──► references (dict of Phonopy objects)
  reference_data/                     ┘                            + sources (where each came from)

MODELS["name"]["function"](device) ──► make_mace / make_prophet / make_sevennet ──► ASE calculator
```

---

## 1. What the file needs

### Python packages

| Package | Used for |
|---|---|
| `pathlib.Path`, `json` | Folder paths; reading the Materials Project settings files |
| `numpy` | The cell comparison in `check_reference_cell` |
| `ase.io.read` | Reading CIF files into ASE `Atoms` objects |
| `pymatgen` (`Structure`, `AseAtomsAdaptor`, `SpacegroupAnalyzer`) | Converting between ASE and pymatgen, space groups, reading MP structures |
| `phonopy` (`load`, `Phonopy`, `PhonopyAtoms`, `parse_FORCE_CONSTANTS`) | Loading reference phonon data as `Phonopy` objects |
| `monty.serialization.loadfn` | Loading the two lookup files (phononDB's is a pandas DataFrame) |
| `MLIP_config` (`CONFIG`) | The data folders and `DTYPE` |
| `mace-torch` | Only when `make_mace` is called |
| `prophet-mlip` (imported as `prophet`) | Only when `make_prophet` is called |
| `sevenn` | Only when `make_sevennet` is called |

The model packages are imported *inside* their functions, not at the top of the file, so the workflow runs in an environment that has only the model you are using. The install commands are in the top-level `README.md`.

### Folders and files on disk

The three folders are set in `config_file.txt` (relative to the code folder unless a full path is given) and read here from `CONFIG`.

| Constant (config setting) | Default folder | What must be in it | Needed by |
|---|---|---|---|
| `MATERIALS_CIF_FOLDER` | `materials_cif/` | One CIF per material, named `<material>.cif` (e.g. `Si.cif`) | `cif_parser` (required) |
| `PHONON_DB_FOLDER` | `reference_data/phononDB/` | `phonon_db_lookup_df.json` and `phonon_db/mp-XXXX-20180417/` folders | `reference_data_parser` (optional) |
| `MP_REFERENCE_FOLDER` | `reference_data/materials_project_reference_data/` | `phonon_mp_lookup.json`, `phonopy_settings/<name>_phonopy_settings.json`, `force_constants/<name>_force_constants` (written by `MaterialsProject_extractor.ipynb`) | `reference_data_parser` (optional) |
| (inside `make_prophet`) | `weights/` | `prophet-oame-mbd.pt` | `make_prophet` (required for Prophet) |

The extractor also writes `band_structure/<name>-phonondata.json`, `dos/<name>_phonondos.json` and `structure/<name>.cif`. The parser does not read these.

### Other module-level names

- `SOURCES = ("phononDB", "MP")`: the reference databases `find_reference` accepts by name.
- `MODELS` is the calculator registry (see section 4).

---

## 2. Structure parser

### `cif_parser(folder_path)`

**What it does:** reads **every** `.cif` file in the folder and returns a dictionary `{material name: ASE Atoms}`. The material name is the file name without `.cif`.

**Steps, for each CIF file:**
1. `ase.io.read` reads the file into an ASE `Atoms` object.
2. The structure is converted to a pymatgen `Structure`.
3. `SpacegroupAnalyzer` finds the space group symbol and number.
4. Three values are stored in the `Atoms.info` dictionary:
   - `space_group_symbol` (e.g. `Fd-3m`)
   - `space_group_number` (e.g. `227`)
   - `reduced_formula` (e.g. `Si`, `GaAs`, `MgO`)

**Relies on:** CIF files in `MATERIALS_CIF_FOLDER`; `ase`, `pymatgen`.

**Returns:** `materials`, a dictionary of ASE `Atoms` objects.

**Raises:** `FileNotFoundError` if the folder has no CIF files.

**Who uses the result:**
- The runner filters it down to the names in `MATERIALS` (config).
- `reference_data_parser` reads `info["reduced_formula"]` and `info["space_group_number"]` to match references automatically.
- The runner reads `info["space_group_symbol"]` as the "before relaxation" space group.
- `MLIP_plotter.py` takes the structure of its `MATERIAL` for the band path.

**Good to know:**
- It parses all CIFs in the folder, even ones not listed in `MATERIALS`. A broken CIF for a material you are not running will still stop the script.
- `SpacegroupAnalyzer` is used with its default tolerance (`symprec = 0.01`).
- The file name is the key for everything downstream (result folders, `REFERENCES`, the plot file's `MATERIAL`), so name CIFs carefully.
- ASE may warn that a CIF's crystal system "is not interpreted" for its space group. That is about the CIF header and is harmless.

---

## 3. Reference data parsers

Two reference sources are supported: the Kyoto phonon database (phononDB) and Materials Project phonon data downloaded with `MaterialsProject_extractor.ipynb`. `reference_data_parser` is the function the runner (and the plotter with `REFERENCE = AUTO`) calls; the others are helpers.

```
reference_data_parser
   ├── phonon_db_lookup, mp_lookup     (load the two index files)
   ├── find_reference                  (a material listed in REFERENCES: which source has that mp-id?)
   ├── mp_finder, phonon_db_finder     (a material not listed: automatic match)
   ├── mp_parser, phonon_db_parser     (load the chosen reference as a Phonopy object)
   └── check_reference_cell            (warn if its primitive cell differs from the CIF's)
```

### Background: the two databases

- **phononDB:** the WMD-group packaging of Atsushi Togo's Kyoto University phonon database (release 2018-04-17), set up with `Kyoto_database_extractor.ipynb`. 10,034 materials, each in `phonon_db/mp-XXXX-20180417/` with `phonon.yaml`, `FORCE_SETS` and (for about half) `BORN`. `phonon_db_lookup_df.json` is the index; the columns used are `material_id`, `formula_pretty_reduced`, `spacegroup_number` and `phonon_filepath`. **The DFT functional is PBEsol, not PBE.**
- **Materials Project:** one material per run of `MaterialsProject_extractor.ipynb` (`phonon_method="pheasy"`). `phonon_mp_lookup.json` maps `{"mp-id": {"material": "<name>"}}`. `phonopy_settings/<name>_phonopy_settings.json` holds the unit cell, supercell matrix, primitive matrix, Born charges and dielectric tensors; `force_constants/<name>_force_constants` holds the force constants for that supercell.

### `phonon_db_lookup(phonon_db_folder)`

Loads `phonon_db_lookup_df.json` with `loadfn` (not `json.load`: the file is a monty-saved DataFrame). Returns the DataFrame, or `None` with a printed message if the file is missing or cannot be read.

### `mp_lookup(mp_folder)`

Loads `phonon_mp_lookup.json` with `loadfn`. Returns the dictionary, or `{}` with a printed message if the file is missing or cannot be read.

### `phonon_db_finder(material_name, structure, lookup)`

**What it does:** finds the phononDB entry whose `formula_pretty_reduced` **and** `spacegroup_number` both match the CIF structure.
- exactly one match: returns its mp-id;
- more than one (e.g. two polymorphs with the same space group): prints a message asking you to add the right one to `REFERENCES`, returns `None`;
- no match: returns `None`.

**Good to know:** the match is only as good as the space group found from your CIF. A slightly distorted CIF can give a lower-symmetry space group and no match; then name the mp-id in `REFERENCES`.

### `mp_finder(material_name, structure, mp_folder, mp_names)`

**What it does:** finds the Materials Project entry for a material.
1. **Name match:** an entry of `phonon_mp_lookup.json` whose `material` is the CIF name.
2. **Formula and space group match:** otherwise, reads each entry's `phonopy_settings/<name>_phonopy_settings.json` and compares its structure's reduced formula and space group number with the CIF's.

Returns the mp-id if there is exactly one match, otherwise `None` (printing a message if there were several).

### `find_reference(material_name, request, phonon_db_ids, mp_ids)`

**What it does:** works out which source an entry of `REFERENCES` points to.
- `"<mp-id>"` (e.g. `Si mp-149` in the config): looks for the id in both sources; if it is in both, phononDB (first in `SOURCES`) is used, with a message.
- `("<source>", "<mp-id>")` (e.g. `Si MP mp-149`): only that source. An unknown source name is an error.

Returns `(source, mp_id)`, or `None` with a message if the id is not found.

### `phonon_db_parser(phonon_db_folder, lookup, mp_id)`

**What it does:** loads one phononDB material with `phonopy.load`: its `phonon.yaml`, `force_sets_filename = FORCE_SETS`, `primitive_matrix = "auto"`, `is_nac = False`.

**Good to know:**
- `is_nac=False` means `BORN` is ignored, so there is **no LO/TO splitting** in the reference. This is deliberate: the MLIP phonons have no Born charges either.
- The folder name is taken from the `phonon_filepath` column with `Path(...).parent.name`. A lookup table built on Windows can hold Windows paths, which fail on Linux; rebuild the table on each machine (see `KYOTO_database_extractor_README.md`).

### `mp_parser(mp_folder, mp_name)`

**What it does:** builds a `Phonopy` object from the Materials Project files for one material.

**Steps:**
1. Checks that `phonopy_settings/<name>_phonopy_settings.json` and `force_constants/<name>_force_constants` exist (otherwise `FileNotFoundError` telling you to run the extractor).
2. Builds the unit cell from the settings file's `structure`.
3. Creates `Phonopy` with the reference's **own** `supercell_matrix` and `primitive_matrix` (or `"auto"` if none was saved), not `DIM` / `PRIMITIVE_AXES` from the config: its force constants were calculated in its own supercell.
4. Reads the force constants with `parse_FORCE_CONSTANTS`, checks their size matches the supercell, and attaches them.

**Good to know:** the Born charges in the settings file are not used (no NAC, as for phononDB). The `dos_file` and `band_structure_file` names in this function are not used yet, and do not yet match the names the extractor writes (`_phonondos.json`, `-phonondata.json`).

### `check_reference_cell(material_name, structure, phonon)`

**What it does:** compares the reference's primitive cell with the CIF's standard primitive cell (lattice lengths and angles). If lengths differ by more than 5 % or angles by more than 2°, it prints a warning: the band path is built from the CIF's cell, so the same labels would mean different q-points for the reference.

### `reference_data_parser(materials, phonon_db_folder, mp_folder, requested=None)`

**What it does:** finds and loads the reference for every material. `requested` is `REFERENCES` from the config, e.g. `{"Si": "mp-149"}`.

**For each material:**

| Case | How the reference is chosen |
|---|---|
| Listed in `REFERENCES` | `find_reference` (no automatic matching if the id is not found) |
| Not listed | `mp_finder` first (the MP folder), then `phonon_db_finder` (phononDB) |
| Nothing found | `sources[material] = "none"`, with a message |

The chosen reference is loaded with `mp_parser` or `phonon_db_parser`. A loading error is printed and the material gets no reference; it does not stop the run. Each loaded reference goes through `check_reference_cell`.

**Returns two dictionaries:**
- `references`: `{material name: Phonopy object}`. A material with no reference is **absent** (the runner checks `if material_name in references`).
- `sources`: `{material name: label}` for every material, e.g. `"MP mp-149"`, `"phononDB mp-149"` or `"none"`. The label is used as the reference's legend entry in the plots.

**Good to know:** the phononDB lookup is optional. Without it you will see `phononDB lookup table not found` and only the Materials Project folder is used.

---

## 4. MLIP calculator functions

Each function takes a `device` (`"cpu"` or `"cuda"`) and returns an ASE calculator. Every model is used the same way afterwards (`atoms.calc = calc`), which is what makes the workflow model-agnostic.

### `make_mace(device)`

Returns the MACE-MP-0 **medium** foundation model: `mace_mp(model="medium", default_dtype=CONFIG["DTYPE"], device=device)`. `DTYPE` in the config (`float64` or `float32`) sets its precision.

**Relies on:** `mace-torch` (clear `ImportError` if missing); internet access the **first** time it runs, to download and cache the weights (on an HPC compute node without internet, run it once on a login node first).

### `make_prophet(device, dtype=CONFIG["DTYPE"])`

Returns Prophet-OAME-MBD as a `KairosCalculator(model_path=..., use_kernel=False, device=device)`.

**Steps:** checks that `weights/prophet-oame-mbd.pt` exists (otherwise `FileNotFoundError` with the pinned Hugging Face download link, `PROPHET_URL`); imports `KairosCalculator` from `prophet`; builds the calculator.

**Good to know:**
- `use_kernel=False` turns off Prophet's GPU kernel so it also runs on CPU.
- `dtype` is accepted but **not used**: no precision option has been found for `KairosCalculator` yet (open question, see `claude/deferred_features.md`).
- The weights file is placed by hand (CC BY 4.0 licence).

### `make_sevennet(device, dtype=CONFIG["DTYPE"])`

Returns SevenNet-Omni-i12: `SevenNetCalculator(model="7net-omni-i12", modal="mpa", device=device)`.

**Good to know:**
- `dtype` is accepted but **not used**: `SevenNetCalculator` has no precision option and builds its inputs in `float32`, so `DTYPE = float64` in the config does not change SevenNet (open question for the supervisor).
- SevenNet warns that no tensor-product accelerator is enabled, so it may run slowly. The run still works.

### `MODELS` dictionary

```python
MODELS = {
    "MACE-MP-0-medium":  {"function": make_mace,     "colour": "tab:blue"},
    "SevenNet-Omni-i12": {"function": make_sevennet, "colour": "tab:orange"},
    "Prophet-OAME-MBD":  {"function": make_prophet,  "colour": "tab:green"},
}
```

Maps a model name to the function that builds it and the colour it is drawn in on every plot. The runner does `calc = MODELS[MODEL]["function"](DEVICE)`; the runner's band plot and `MLIP_plotter.py` take `MODELS[model]["colour"]`.

The key is also the **folder name** for results (`results/<material>/<model name>/...`), so avoid characters that are not allowed in folder names.

**To add a new model:**
1. Write a `make_<model>(device, dtype=CONFIG["DTYPE"])` function that returns an ASE calculator.
2. Add one line to `MODELS` with its function and a colour not used by the other models.
3. Set `MODEL` in `config_file.txt` to the new key.

Nothing in `MLIP_processor.py` needs to change.

---

## 5. Summary table

| Function | Called by | Relies on | Returns |
|---|---|---|---|
| `cif_parser` | runner, plotter | `materials_cif/*.cif`, ase, pymatgen | dict of ASE `Atoms` |
| `phonon_db_lookup` | `reference_data_parser` | `phonon_db_lookup_df.json`, monty | DataFrame or `None` |
| `mp_lookup` | `reference_data_parser` | `phonon_mp_lookup.json`, monty | dict or `{}` |
| `phonon_db_finder` | `reference_data_parser` | `Atoms.info` from `cif_parser`, phononDB lookup | mp-id or `None` |
| `mp_finder` | `reference_data_parser` | MP lookup, `phonopy_settings/*.json` | mp-id or `None` |
| `find_reference` | `reference_data_parser` | `REFERENCES` entry, both id sets | `(source, mp-id)` or `None` |
| `phonon_db_parser` | `reference_data_parser` | `phonon.yaml` + `FORCE_SETS`, phonopy | `Phonopy` object |
| `mp_parser` | `reference_data_parser`, plotter | `<name>_phonopy_settings.json` + `<name>_force_constants`, phonopy | `Phonopy` object |
| `check_reference_cell` | `reference_data_parser`, plotter | CIF structure, reference `Phonopy` | prints a warning if the cells differ |
| `reference_data_parser` | runner, plotter (`AUTO`) | all of the above | `references`, `sources` |
| `make_mace` | runner via `MODELS` | `mace-torch`, internet on first use, `DTYPE` | ASE calculator |
| `make_prophet` | runner via `MODELS` | `prophet-mlip`, `weights/prophet-oame-mbd.pt` | ASE calculator |
| `make_sevennet` | runner via `MODELS` | `sevenn` | ASE calculator |
