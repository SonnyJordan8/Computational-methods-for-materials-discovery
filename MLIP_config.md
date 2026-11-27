# MLIP_config.py

**Role in the workflow:** reads the settings file (`config_file.txt` by default) and turns it into the `CONFIG` dictionary that every other script uses. It also decides which settings make a run folder safe to reuse. It does no physics itself.

It is imported by `MLIP_parser.py`, `MLIP_processor.py`, `MLIP_runner.py` and `MLIP_plotter.py`. It imports nothing from the other workflow scripts.

```
python MLIP_runner.py --config my_settings.txt
                │
find_config_file ──► CONFIG_FILE ──► read_config ──► CONFIG (dict, read once at import)
                                                        │
run folder/run_config.txt ──► changed_settings ──► list of settings that differ
```

---

## 1. What the file needs

### Python packages

| Package | Used for |
|---|---|
| `pathlib.Path` | Folder paths that work on Windows and Linux |
| `argparse` | Reading `--config` from the command line |
| `numpy` | Building 3×3 matrices for `DIM` and `PRIMITIVE_AXES` |

### Files on disk

| File | Where | Needed |
|---|---|---|
| `config_file.txt` | the code folder (the folder containing `MLIP_config.py`) | yes, unless another file is given with `--config` |

### Names other scripts import

| Name | What it is |
|---|---|
| `CODE_FOLDER` | The folder containing `MLIP_config.py`. Relative folder settings are relative to this, not to where you run Python from. |
| `CONFIG` | The settings dictionary, read once when the module is first imported. |
| `CONFIG_FILE` | The full path of the settings file that was read. |
| `SETTINGS` | Every allowed setting and the function that reads it. |
| `OPTIONAL` | Settings that may be left out, and the value they take then. |
| `CALCULATION_SETTINGS` | Settings that must match before a run folder is reused. |
| `read_config`, `changed_settings`, `supercell_label` | Functions, see below. |

---

## 2. The settings file

Plain text, one setting per line:

```
KEY = value      # anything after # is a comment
```

- Keys are not case-sensitive (`dim` and `DIM` are the same).
- Blank lines and comment-only lines are ignored.
- **Every setting in `SETTINGS` must be present**, except those in `OPTIONAL` (`IRREPS`, `MAGMOM`). A missing one stops the run.
- **An unknown key stops the run**, so a misspelt setting is caught rather than ignored.
- **A key written twice stops the run.**
- Phonopy settings use phonopy's own tag names and value formats (e.g. `.TRUE.` / `.FALSE.`), so the same values can be used with phonopy on the command line.

### Value formats

| Format | Settings | Examples |
|---|---|---|
| `.TRUE.` / `.FALSE.` (phonopy style; `True` is an error) | `RUN_CALCULATIONS`, `RUN_PLOTS_RUNS`, `RELAX_CELL`, `TPROP`, `SHOW_IRREPS`, `HDF5`, `WRITE_FORCE_CONSTANTS`, `SYMMETRY`, `RELAX_SYMMETRY`, `FC_SYMMETRY` | `RELAX_CELL = .TRUE.` |
| whole number | `MAX_STEPS`, `BAND_POINTS` | `MAX_STEPS = 500` |
| decimal number | `IMAGINARY_LIMIT`, `RELAX_FMAX`, `ENERGY_TOLERANCE`, `PRIMITIVE_TOLERANCE`, `DISPLACEMENT_DISTANCE`, `TMIN`, `TMAX`, `TSTEP`, `SYMMETRY_TOLERANCE` | `RELAX_FMAX = 1e-4` |
| one word, kept as written | `MODEL`, `DEVICE`, `DISPLACEMENTS_FILE`, `FORCE_SETS_FILE` | `MODEL = MACE-MP-0-medium` |
| one word from a fixed list (any capitalisation) | `DTYPE`: `float64`, `float32`<br>`OPTIMIZER`: `FIRE`, `BFGS`, `LBFGS`, `BFGSLineSearch`<br>`FC_CALCULATOR`: `traditional`, `symfc`, `alm` | `OPTIMIZER = fire` |
| folder (relative to the code folder, or a full path) | `RESULTS_FOLDER`, `MATERIALS_CIF_FOLDER`, `PHONON_DB_FOLDER`, `MP_REFERENCE_FOLDER` | `RESULTS_FOLDER = results` |
| material names, space-separated, or `ALL` | `MATERIALS` | `MATERIALS = Si GaAs MgO` |
| `<material> <mp-id>` or `<material> <source> <mp-id>`, comma-separated; may be empty | `REFERENCES` | `REFERENCES = Si mp-149, alpha-3R-In2Se3 MP mp-19907` |
| 3 whole numbers (diagonal) or 9 (full matrix, row by row) | `DIM` | `DIM = 2 2 2` |
| `P`, `AUTO`, or 9 numbers (fractions allowed) | `PRIMITIVE_AXES` | `PRIMITIVE_AXES = 0 1/2 1/2 1/2 0 1/2 1/2 1/2 0` |
| `.TRUE.`, `.FALSE.` or `AUTO` | `PM` | `PM = AUTO` |
| 3 whole numbers | `MESH` | `MESH = 20 20 20` |
| q-point (3 numbers, fractions allowed) and optional degeneracy tolerance | `IRREPS` (optional) | `IRREPS = 0 0 0 1e-4` |
| 0, 1 or 2 | `PHONOPY_LOG_LEVEL` | `PHONOPY_LOG_LEVEL = 1` |
| numbers, or empty | `MAGMOM` (optional) | `MAGMOM =` |

### What each value becomes in `CONFIG`

| Setting | Python value |
|---|---|
| `MATERIALS = ALL` | `None` (the runner then uses every CIF) |
| `MATERIALS = Si GaAs` | `["Si", "GaAs"]` |
| `DIM = 2 2 4` | `[[2, 0, 0], [0, 2, 0], [0, 0, 4]]` |
| `PRIMITIVE_AXES = P` / `AUTO` | `"P"` / `"auto"` |
| `PM = AUTO` | `"auto"` |
| `REFERENCES = Si mp-149, X MP mp-1` | `{"Si": "mp-149", "X": ("MP", "mp-1")}` |
| `IRREPS = 0 0 0` | `([0.0, 0.0, 0.0], None)` |
| `IRREPS` left out | `None` |
| `MAGMOM =` (empty) or left out | `None` |
| a folder | a `Path` |

---

## 3. Functions

### Value readers (`read_bool`, `read_int`, `read_float`, `read_word`, `read_folder`, `read_names`, `read_dim`, `read_fraction`, `read_primitive_axes`, `read_plusminus`, `read_fc_calculator`, `read_choice`, `read_mesh`, `read_irreps`, `read_magmom`, `read_log_level`, `read_references`)

Each takes the text after `=` and returns a Python value, or raises `ValueError` with a message saying what was expected. `read_config` adds the file name and line number to the message.

`read_choice(options)` is different: it *makes* a reader that accepts one word from `options`, ignoring capitalisation, and returns the word as spelt in `options`.

### `read_config(path, strict=True)`

**What it does:** reads a settings file and returns the settings dictionary.

**Steps:**
1. Starts from `OPTIONAL` (`IRREPS = None`, `MAGMOM = None`).
2. For each line: removes the comment, skips blank lines, splits at the first `=`, upper-cases the key, and passes the value to that key's reader.
3. Checks that every non-optional setting was given.

**`strict`:**

| | `strict=True` (default) | `strict=False` |
|---|---|---|
| Key not in `SETTINGS` | error | skipped |
| Setting missing from the file | error | left out of the result |
| Value that can't be read | error | error |
| Key written twice | error | error |

`strict=True` is used for the main settings file. `strict=False` is used for the `run_config.txt` copies saved in run folders, so runs made by an older version of the workflow (with settings since added or removed) can still be read. See `design_decisions.md` (24 Nov 2026).

**Raises:** `FileNotFoundError` if the file doesn't exist; `ValueError` (with file name and line number) for anything it can't read.

### `changed_settings(saved_config_file)`

**What it does:** compares a run folder's saved `run_config.txt` with the current `CONFIG`, for the settings in `CALCULATION_SETTINGS` only. Returns a list of differences as text, e.g. `"DIM: [[2,0,0],...] -> [[3,0,0],...]"`. An empty list means the folder can be reused.

A setting the saved copy doesn't have is shown as `'not recorded'` and counts as different, so the run stops rather than assuming it was the same.

**Used by:** the runner, before it writes into an existing run folder.

### `CALCULATION_SETTINGS`

The settings that change the relaxed structure, the forces or the force constants:

`DTYPE`, `OPTIMIZER`, `RELAX_CELL`, `RELAX_FMAX`, `ENERGY_TOLERANCE`, `MAX_STEPS`, `RELAX_SYMMETRY`, `DIM`, `PRIMITIVE_AXES`, `DISPLACEMENT_DISTANCE`, `PM`, `MAGMOM`, `SYMMETRY`, `SYMMETRY_TOLERANCE`, `FC_CALCULATOR`, `FC_SYMMETRY`.

Plot-only settings (`BAND_POINTS`, `MESH`, `TPROP`, ...) are not in the list, so changing them does not block reusing a folder. `MODEL` is not in the list because each model has its own results folder.

### `supercell_label(dim)`

**What it does:** turns a supercell matrix into short text for folder names.
- Diagonal `[[2,0,0],[0,2,0],[0,0,4]]` → `"2x2x4"`
- Anything else → `"m"` followed by the 9 numbers joined with `_`, e.g. `"m0_2_2_2_0_2_2_2_0"`

### `find_config_file()`

**What it does:** returns the settings file to read: the one given with `--config` on the command line, otherwise `config_file.txt` in the code folder. Unknown command-line arguments are ignored (`parse_known_args`), so other scripts can add their own.

```
python MLIP_runner.py                              # uses config_file.txt
python MLIP_runner.py --config Si_float32.txt      # uses Si_float32.txt
```

This is what makes it easy to submit several HPC jobs with different settings from the same code.

---

## 4. Good to know

- **`CONFIG` is read once, when `MLIP_config` is first imported.** Any script that imports it (including `MLIP_plotter.py`) needs a complete, valid settings file, even if it only uses a few settings. Editing the file during a run has no effect on that run.
- **`--config` works for every script**, because every script imports `MLIP_config`.
- **Relative folders are relative to the code folder**, not to the folder you run Python from.
- **`MAGMOM` is read but not used yet.** If it is set, a note is printed at import.
- **`DTYPE` only affects MACE at the moment.** SevenNet and Prophet run in their own precision whatever `DTYPE` says, but `run_config.txt` still records the config value. See `deferred_features.md` (24 Nov 2026).
- **There is no setting for a model's channel or task** (e.g. SevenNet-Omni's `mpa` channel). It is fixed inside the model's set-up function, so it is not recorded in `run_config.txt`.
- The comment above `SETTINGS` says each entry is `(reader, default)`, but there are no defaults: each entry is just the reader, and every setting must be in the file.

---

## 5. Summary table

| Function | Called by | Returns |
|---|---|---|
| `find_config_file` | this module, at import | `Path` of the settings file |
| `read_config` | this module (main file, strict); `changed_settings` and `load_run_phonon` (saved copies, lenient) | settings dictionary |
| `changed_settings` | runner | list of differences as text |
| `supercell_label` | other scripts, for run folder names | text such as `"2x2x2"` |
| readers (`read_*`) | `read_config`, via `SETTINGS` | one Python value |
