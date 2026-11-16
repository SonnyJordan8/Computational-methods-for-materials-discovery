# MLIP_parser.py

**Role in the workflow:** gets everything *into* the program. It reads the crystal structures (CIF files), finds the reference phonon data for each material, and builds the MLIP calculators. It does no physics itself.

It is imported by `MLIP_runner.py`. It does not import `MLIP_processor.py`.

```
materials_cif/*.cif ──► cif_parser ──────────► materials (dict of ASE Atoms)
                                                    │
phononDB/ + reference_data/ ──► reference_data_parser ──► references (dict of Phonopy objects)

MODELS["name"](device) ──► make_mace / make_prophet ──► ASE calculator
```

---

## 1. What the file needs

### Python packages

| Package | Used for |
|---|---|
| `pathlib.Path` | Building folder paths that work on Windows and Linux |
| `ase.io.read` | Reading CIF files into ASE `Atoms` objects |
| `pymatgen` (`AseAtomsAdaptor`, `SpacegroupAnalyzer`) | Converting ASE to pymatgen and finding the space group |
| `phonopy` | Loading reference phonon data as `Phonopy` objects |
| `monty.serialization.loadfn` | Loading the phononDB lookup table (a JSON file that becomes a pandas DataFrame) |
| `mace-torch` | Only when `make_mace` is called |
| `prophet` | Only when `make_prophet` is called |

`mace` and `prophet` are imported *inside* their functions, not at the top of the file. This means you can run the workflow with MACE in an environment where Prophet is not installed, and the other way round.

### Folders and files on disk

All paths are relative to the folder that contains `MLIP_parser.py`.

| Constant | Folder | What must be in it | Needed by |
|---|---|---|---|
| `MATERIALS_CIF_FOLDER` | `materials_cif/` | One CIF per material, named `<material>.cif` (e.g. `Si.cif`) | `cif_parser` (required) |
| `PHONON_DB_FOLDER` | `phononDB/` | `phonon_db_lookup_df.json` and `phonon_db/mp-XXXX-20180417/` folders | `reference_data_parser` (optional) |
| `REFERENCE_PHONOPY_PARAMS_FOLDER` | `reference_data/` | `<material>_phonopy_params.yaml` files | `reference_data_parser` (optional fallback) |
| (inside `make_prophet`) | `weights/` | `prophet-oame-mbd.pt` | `make_prophet` (required for Prophet) |

### Other module-level names

- `molecules = {"Si": {"Si"}, "GaAs": {"Ga", "As"}}` maps a material to its elements. **It is not used anywhere** in the three scripts at the moment.
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

**Relies on:**
- CIF files in `materials_cif/`.
- `ase`, `pymatgen`.

**Returns:** `materials`, a dictionary of ASE `Atoms` objects.

**Raises:** `FileNotFoundError` if the folder has no CIF files.

**Who uses the result:**
- The runner filters it down to the names in `MATERIALS`.
- `reference_data_parser` reads `info["reduced_formula"]` and `info["space_group_number"]` to match against phononDB.
- The runner reads `info["space_group_symbol"]` as the "before relaxation" space group.

**Good to know:**
- It parses all CIFs in the folder, even ones not listed in `MATERIALS`. A broken CIF for a material you are not running will still stop the script.
- `SpacegroupAnalyzer` is used with its default tolerance (`symprec = 0.01`).
- The file name is the key for everything downstream (result folders, reference files, `REFERENCE_MP_IDS`), so name CIFs carefully.

---

## 3. Reference data parsers

Three functions work together. `reference_data_parser` is the one the runner calls; the other two are helpers.

```
reference_data_parser
   ├── phonon_db_finder   (which phononDB entry matches this material?)
   └── phonon_db_parser   (load that entry as a Phonopy object)
```

### Background: the Kyoto phonon database (phononDB)

- Source: WMD-group packaging of Atsushi Togo's Kyoto University phonon database (release 2018-04-17), set up locally with `Kyoto_database_extractor.ipynb`.
- 10,034 materials, each in a folder `phonon_db/mp-XXXX-20180417/`, keyed by Materials Project ID.
- Each folder holds `phonon.yaml`, `FORCE_SETS`, and (for about half the materials) `BORN`.
- `phonon_db_lookup_df.json` is the index. The columns this script uses are `material_id`, `formula_pretty_reduced`, `spacegroup_number` and `phonon_filepath`.
- **The DFT functional is PBEsol, not PBE**, with no van der Waals correction. Keep this in mind when calling it "the reference".
- **GaAs is not in this release.** For GaAs the workflow falls through to the `reference_data/` folder, or to no reference at all.

### `phonon_db_finder(material_name, structure, lookup, mp_ids)`

**What it does:** decides which phononDB entry (mp-id) belongs to a material. Returns the mp-id as a string, or `None` if there is no usable match.

**Two routes:**

1. **Manual route.** If the material name is a key in `mp_ids` (this is `REFERENCE_MP_IDS` from the runner):
   - if that mp-id is in the lookup table, return it;
   - if not, print `"<material>: <mp-id> is not in phononDB"` and return `None`. It does **not** go on to try the automatic route.
2. **Automatic route.** Otherwise, find the rows of the lookup table where **both** of these match the CIF structure:
   - `formula_pretty_reduced` equals `structure.info["reduced_formula"]`
   - `spacegroup_number` equals `structure.info["space_group_number"]`

   Then:
   - exactly one match: return its mp-id;
   - more than one match (e.g. two polymorphs with the same space group): print a message asking you to add the right one to `REFERENCE_MP_IDS`, return `None`;
   - no match: return `None` silently.

**Relies on:**
- `cif_parser` having filled in `structure.info` (formula and space group number).
- The lookup DataFrame loaded by `reference_data_parser`.
- `REFERENCE_MP_IDS` in the runner for the manual route.

**Good to know:** the match is only as good as the space group found from your CIF. If the CIF is slightly distorted and pymatgen finds a lower-symmetry space group, the automatic match will fail and you need the manual route.

### `phonon_db_parser(phonon_db_folder, lookup, mp_id)`

**What it does:** loads one phononDB material as a `Phonopy` object.

**Steps:**
1. Finds the row of the lookup table with this mp-id.
2. Builds the folder path: `phononDB/phonon_db/<folder name taken from phonon_filepath>`.
3. Calls `phonopy.load` with:
   - `phonon.yaml` (cell, supercell and displacement information)
   - `force_sets_filename = FORCE_SETS` (the DFT forces)
   - `primitive_matrix = "auto"` (phonopy works out the primitive cell itself)
   - `is_nac = False` (no non-analytical correction)

**Relies on:**
- The unpacked material folder existing on disk with `phonon.yaml` and `FORCE_SETS`.
- `phonopy`.

**Returns:** a `Phonopy` object with force constants ready for band structure or q-point calculations.

**Good to know:**
- `is_nac=False` means the `BORN` file is ignored, so there is **no LO/TO splitting** in the reference for polar materials (MgO, GaAs). This is deliberate for now, because the MLIP phonons have no Born charges either, so the two are compared like-for-like.
- When Born charges are added later, pass `born_filename=<folder>/BORN` explicitly. phonopy will not find the `BORN` file on its own in this layout.

### `reference_data_parser(materials, phonon_db_folder, reference_folder, mp_ids=None)`

**What it does:** finds reference phonon data for every material, trying sources in a fixed order. This is the function the runner calls.

**Order of preference, for each material:**

| Order | Source | Condition | Label stored in `sources` |
|---|---|---|---|
| 1 | phononDB | the lookup file exists **and** `phonon_db_finder` returns an mp-id | `"phononDB mp-XXXX"` |
| 2 | `reference_data/<material>_phonopy_params.yaml` | the file exists | the file name |
| 3 | nothing | neither of the above | `"none"` |

**Steps:**
1. Looks for `phononDB/phonon_db_lookup_df.json`. If it exists it is loaded with `loadfn`; if not, a message is printed and only the `reference_data/` folder is used.
2. Loops over the materials and applies the order above.
3. Prints one line per material saying where its reference came from.

**Relies on:**
- `materials` from `cif_parser` (with the `info` values filled in).
- `phonon_db_finder` and `phonon_db_parser`.
- The phononDB folder (optional) and the `reference_data/` folder (optional).
- `monty`, `phonopy`.

**Returns two dictionaries:**
- `references`: `{material name: Phonopy object}`. A material with no reference data is **absent** from this dictionary (the runner checks `if material_name in references`).
- `sources`: `{material name: text label}` for every material.

**Good to know:**
- A `<material>_phonopy_params.yaml` in `reference_data/` must contain force constants (or forces) so that `phonopy.load` can use it directly.
- The two sources can use different DFT settings (phononDB is PBEsol). The `sources` dictionary is how you tell which one a result was compared with. At the moment `sources` is printed but **not written to `summary.csv`**.

---

## 4. MLIP calculator functions

Each function takes a `device` (`"cpu"` or `"cuda"`) and returns an ASE calculator. Every model is used the same way afterwards (`atoms.calc = calc`), which is what makes the workflow model-agnostic.

### `make_mace(device)`

**What it does:** returns the MACE-MP-0 **medium** foundation model as an ASE calculator.

**Settings:** `model="medium"`, `default_dtype="float64"`, `device=device`.

**Relies on:**
- The `mace-torch` package (clear `ImportError` if missing).
- Internet access the **first** time it runs: `mace_mp` downloads the model weights and caches them. On an HPC compute node without internet, run it once on a login node first.

**Good to know:** `float64` is used because phonons come from small differences in forces between displaced structures, and `float32` rounding noise can be comparable to those differences.

### `make_prophet(device)`

**What it does:** returns the Prophet-OAME-MBD model as an ASE calculator (`KairosCalculator`).

**Steps:**
1. Checks that `weights/prophet-oame-mbd.pt` exists. If not, raises `FileNotFoundError` with the download URL. It does not download the file for you.
2. Imports `KairosCalculator` from the `prophet` package.
3. Returns `KairosCalculator(model_path=..., use_kernel=False, device=device)`.

**Relies on:**
- The weights file placed by hand in `weights/` (CC BY 4.0 licence).
- The `prophet` package (clear `ImportError` if missing).

**Good to know:** `use_kernel=False` turns off Prophet's GPU kernel so the model also runs on CPU.

### `MODELS` dictionary

```python
MODELS = {
    "MACE-MP-0-medium": make_mace,
    "Prophet-OAME-MBD": make_prophet,
}
```

Maps a model name to the function that builds it. The runner does `calc = MODELS[MODEL](DEVICE)`.

The key is also used as the **folder name** for results (`results/<material>/<model name>/...`) and as the legend label in the plots, so avoid characters that are not allowed in folder names.

**To add a new model:**
1. Write a `make_<model>(device)` function that returns an ASE calculator.
2. Add one line to `MODELS`.
3. Set `MODEL` in the runner to the new key.

Nothing in `MLIP_processor.py` needs to change.

---

## 5. Summary table

| Function | Called by | Relies on | Returns |
|---|---|---|---|
| `cif_parser` | runner | `materials_cif/*.cif`, ase, pymatgen | dict of ASE `Atoms` |
| `phonon_db_finder` | `reference_data_parser` | `Atoms.info` from `cif_parser`, lookup table, `REFERENCE_MP_IDS` | mp-id or `None` |
| `phonon_db_parser` | `reference_data_parser` | `phononDB/phonon_db/<mp-id folder>/phonon.yaml` + `FORCE_SETS`, phonopy | `Phonopy` object |
| `reference_data_parser` | runner | the two functions above, `phonon_db_lookup_df.json`, `reference_data/*.yaml`, monty | `references`, `sources` |
| `make_mace` | runner via `MODELS` | `mace-torch`, internet on first use | ASE calculator |
| `make_prophet` | runner via `MODELS` | `prophet`, `weights/prophet-oame-mbd.pt` | ASE calculator |
