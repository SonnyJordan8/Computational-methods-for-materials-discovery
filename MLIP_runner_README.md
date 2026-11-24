# MLIP_runner.py

**Role in the workflow:** the script you actually run. It reads its settings from the config file (through `MLIP_config.py`), calls the parser to load the structures, the references and the model, then calls the processor functions in order for each material. It plots each run against its reference; plots comparing several runs are made separately with `MLIP_plotter.py`.

```
python MLIP_runner.py                                  uses config_file.txt next to the scripts
python MLIP_runner.py --config my_settings.txt         uses another config file
```

```
MLIP_runner.py
   ├── imports from MLIP_config.py     (CONFIG, CONFIG_FILE, supercell_label, changed_settings)
   ├── imports from MLIP_parser.py     (folders, MODELS, cif_parser, reference_data_parser)
   └── imports from MLIP_processor.py  (run folder, relaxation, force constants, bands, properties, CSV, plot, timing)
```

## 1. What the file needs

- `MLIP_config.py`, `MLIP_parser.py` and `MLIP_processor.py` in the same folder.
- A config file: `config_file.txt` next to the scripts, or another one given with `--config`.
- `materials_cif/` with a CIF for every name in `MATERIALS`.
- Optional, for the reference comparison: `reference_data/phononDB/` (Kyoto phononDB) and/or `reference_data/materials_project_reference_data/` (filled by `MaterialsProject_extractor.ipynb`).
- The conda environment for the chosen `MODEL` active (see the top-level `README.md`).
- Standard library `datetime` and `shutil`, plus `numpy`.

## 2. Settings

**No settings are changed in this file any more.** They are all in the config file, read once by `MLIP_config.py` when it is imported. The runner copies the ones it uses into constants:

| Constant in the runner | Config setting | Meaning |
|---|---|---|
| `MODEL` | `MODEL` | Which MLIP. Must be a key of `MODELS` in the parser file (checked at the start, with the list of choices if not), and must match the active environment |
| `DEVICE` | `DEVICE` | `cpu` or `cuda`, passed to the model's `make_...` function |
| `DTYPE` | `DTYPE` | Precision; used by `make_mace` only (see the parser README) |
| `RUN_CALCULATIONS` | `RUN_CALCULATIONS` | `.FALSE.` loads the structures and references, then stops |
| `RUN_PLOTS_RUNS` | `RUN_PLOTS_RUNS` | Plot each run against its reference after it is saved |
| `IMAGINARY_LIMIT` | `IMAGINARY_LIMIT` | A lowest frequency below this (THz) counts as an imaginary mode |
| `MATERIALS` | `MATERIALS` | Which CIFs to run; `ALL` runs every CIF in the folder |
| `REFERENCES` | `REFERENCES` | Chosen references, e.g. `Si mp-149` (see Step 2) |
| `SUPERCELL_MATRIX` | `DIM` | The 3×3 supercell matrix, passed to `force_constant_calculator` |
| `SUPERCELL` | from `DIM` | The label (`2x2x2`) from `supercell_label`, used in folder names, the CSVs and the logs |
| `DISPLACEMENT` | `DISPLACEMENT_DISTANCE` | Finite displacement in Å |

All the other settings (relaxation, phonopy, output) are read by the processor itself. The top-level `README.md` lists every setting.

**Running several models:** change `MODEL` in the config (or keep one config file per model and pass it with `--config`), activate that model's environment, and run again. Each model gets its own folders; `MLIP_plotter.py` then puts them on one figure.

## 3. Parse section (always runs)

### Step 1: load the structures

```python
materials = cif_parser(MATERIALS_CIF_FOLDER)
```

- Reads every CIF in `materials_cif/`.
- If `MATERIALS` is not `ALL`: checks every name has a CIF (otherwise `ValueError` listing what is available), then keeps only those, in the order given.

### Step 2: load the reference data

```python
references, reference_sources = reference_data_parser(
    materials, PHONON_DB_FOLDER, MP_REFERENCE_FOLDER, requested=REFERENCES
)
```

- A material listed in `REFERENCES` uses that entry (`Si mp-149` searches both databases; `Si MP mp-149` or `Si phononDB mp-149` names the source).
- Any other material is matched automatically: the Materials Project folder first, then phononDB, then no reference.
- `references` holds a `Phonopy` object for each material that has one; `reference_sources` says where it came from (e.g. `phononDB mp-149`, `MP mp-149`, `none`) and is used as the reference's legend label in the band plot.

How each reference `Phonopy` object is built is explained in section 5.

## 4. Calculations section (`if RUN_CALCULATIONS:`)

### Step 3: load the model (once)

```python
calc = MODELS[MODEL]["function"](DEVICE)
```

`MODELS` entries now hold the set-up function and the model's plot colour, so the function is `MODELS[MODEL]["function"]`. The model is loaded once, timed (material `all`, stage `model load`) and reused for every material.

### Step 4: loop over the materials

Each material is run inside a `try / except`. `stage` is updated before each step, so an error message says where it failed.

| Order | `stage` | What happens | Functions | Values produced |
|---|---|---|---|---|
| 1 | `set up` | Build and create the run folder; **overwrite check**; copy the config into the folder as `run_config.txt`; read the CIF space group; reduce to the primitive cell | `get_run_folder`, `changed_settings`, `get_primitive_cell` | `folder`, `space_group_before`, `atoms` |
| 2 | `relaxation` | Relax; measure leftover force and volume | `relax_structure` | `converged`, `steps`, `energy_change`, `relax_time`, `max_force`, `volume_per_atom` |
| 3 | `symmetry check` | Space group of the relaxed cell, compared with the CIF | `get_space_group` | `space_group_after`, `symmetry_kept` |
| 4 | `force constants` | Displacements, forces, force constants | `force_constant_calculator` | `phonon`, `forces_time` |
| 5 | `band structure` | Band path, bands, save, lowest frequency, Γ optical frequency | `get_band_path`, `band_calculator`, `save_bands`, `lowest_band_frequency`, `gamma_optical_frequency` | `bands`, `lowest_frequency`, `gamma_optical`, `imaginary_modes`, `bands_time` |
| 6 | `phonon properties` | Thermal properties and/or irreps, if switched on (timed only if something was written) | `phonon_properties` | files listed in the terminal |
| 7 | `reference comparison` | Γ optical frequency of the reference and the percentage difference | `gamma_optical_frequency` | `ref_gamma_optical`, `gamma_diff` |
| 8 | `saving results` | Print a summary; add a row to `summary.csv` | `results_table` | one CSV row |
| after | (own `try`) | If `RUN_PLOTS_RUNS`: this run against its reference | `plot_run_bands` | `band_structure.png` |

**Details worth knowing:**

1. **Set up.**
   - The folder name (`sc2x2x2_d0.03_sym-on`, plus `_relaxsym-off` / `_fcsym-off` when those are off) is built only by `get_run_folder`, so runs with different symmetry settings never share a folder.
   - **Overwrite check:** if the folder already has a `run_config.txt`, `changed_settings` compares its calculation settings (`CALCULATION_SETTINGS` in `MLIP_config.py`) with the current config. If any differ, the material stops with "`<folder>` already holds a run with different settings (...)", listing each change, and the old run is left untouched. If none differ, the run is repeated in the same folder. Output-only settings (plots, HDF5, materials list, ...) are not compared.
   - `shutil.copy` saves the config **exactly as written, comments included**, as `run_config.txt`. This is the record of how the run was calculated; `load_run_phonon` reads it later.
   - `get_primitive_cell` returns a new `Atoms`, so the CIF structure in `materials` is left untouched.

2. **Relaxation.** `relax_structure` changes `atoms` in place and returns three values. The terminal line now also gives the last energy change per atom. `max_force` is the largest force component on any atom (eV/Å); with `RELAX_SYMMETRY` on it is the symmetry-averaged force, because the `FixSymmetry` constraint stays on the structure.

3. **Symmetry check.** Compares the space group symbols; a change prints a warning but the run carries on.

4. **Force constants.** Uses the relaxed `atoms` and `SUPERCELL_MATRIX` (the matrix, not the label).

5. **Band structure.** The band path is found from the **relaxed MLIP primitive cell** (`get_band_path(atoms)` after relaxation). `MLIP_plotter.py` uses the unrelaxed CIF primitive cell instead (section 5).

6. **Phonon properties.** Only does something if `TPROP` is on or `IRREPS` is set.

7. **Reference comparison.** Only for materials in `references`:

   ```
   gamma_diff = (gamma_optical - ref_gamma_optical) / ref_gamma_optical * 100
   ```

   Negative means the MLIP is softer than the reference. With no reference both cells are left blank.

8. **Saving results.** The list passed to `results_table` must be in the order of `CSV_COLUMNS` in the processor.

9. **Band plot.** Runs after "saved to", with its own `try / except`, so a plotting error cannot lose the results. Draws the run in its `MODELS` colour against the reference (black, labelled with `reference_sources`), titled with the run folder name.

### If a material crashes

```python
except Exception as error:
    status = f"CRASHED in {stage}: {type(error).__name__}: {error}"
```

- A short row (date, material, model, supercell, displacement, status) goes into `summary.csv`, and the loop moves on to the next material.
- Files written before the crash (including `run_config.txt`) stay in the run folder. Because `run_config.txt` is copied before the calculation, a crashed run with the same settings can simply be rerun.
- The stage that crashed gets no row in `performance_log.csv`.

## 5. How the reference is built and plotted

The reference and the MLIP run are compared on the same band path, but they are **not** built from the same cell. This is what happens for each.

### The reference `Phonopy` object

| Source | Unit cell | Supercell and primitive cell | Force constants |
|---|---|---|---|
| Materials Project (`mp_parser`) | The **starting unit cell stored by MP**, from `structure` in `phonopy_settings/<material>_phonopy_settings.json` | MP's own `supercell_matrix`, and its `primitive_matrix` (or `auto` if none is stored), from the same JSON | Read **directly** from `force_constants/<material>_force_constants` and given to phonopy as they are. No forces or `FORCE_SETS` are involved: MP provides the force constants, and they are not converted into force sets. The parser checks that they are for the same number of atoms as the supercell. |
| Kyoto phononDB (`phonon_db_parser`) | The unit cell in the material's `phonon.yaml` | The supercell in `phonon.yaml`; `primitive_matrix = auto` | **Fitted by phonopy from that material's `FORCE_SETS`** when it is loaded. The Born charges are not used (`is_nac = False`). |

In both cases:
- The reference is used **as published**: it is not relaxed with the MLIP, and the config's `DIM`, `PRIMITIVE_AXES`, `SYMMETRY` and `FC_*` settings are **not** applied to it, because its force constants belong to its own supercell.
- phonopy calculates the reference bands in the reference's **own primitive cell** (from its starting unit cell and primitive matrix).

### The MLIP run

- Unit cell: the CIF in `materials_cif/`, reduced to its standard primitive cell and then **relaxed with the MLIP**.
- Supercell and primitive cell: `DIM` and `PRIMITIVE_AXES` from the config.
- Force constants: calculated from the MLIP forces on the displaced supercells; the forces are saved as `FORCE_SETS`. When `MLIP_plotter.py` reloads a run, `load_run_phonon` **refits the force constants from `FORCE_SETS`** with the settings in the run's `run_config.txt`.

### The shared band path

- The band path (`get_band_path`) is a list of q-points in **fractional coordinates** of a primitive cell: the relaxed MLIP primitive cell in the runner, the unrelaxed CIF primitive cell in `MLIP_plotter.py`.
- The same fractional q-points are then used for the reference, in the reference's primitive cell. This is only correct if the two primitive cells have the same shape and axis order; otherwise a label such as `L` would point to a different q-point in the reference.
- `check_reference_cell` (parser) compares the CIF primitive cell with the reference's primitive cell when the reference is loaded, and prints a **WARNING** if the lengths differ by more than 5 % or the angles by more than 2°. For cubic materials such as Si the cells match; check the warning for rhombohedral and hexagonal materials.
- The Γ optical frequency comparison does not depend on the path, so it is not affected by this.

## 6. What a run produces

**Printed to the terminal:**
- which config file was used; the materials; the reference source of each (and any cell-mismatch warnings);
- model load time;
- for each material: atoms in the primitive cell, displaced supercells, convergence (steps, max force, last energy change), space group before and after, volume per atom, Γ optical frequency, difference from the reference, lowest frequency, stage times, where it was saved, phonon property files, and the band plot.

**Written to disk:**

```
results/
├── summary.csv                                   one row per material per run (appended)
├── performance_log.csv                           one row per stage per run (appended)
└── <material>/<MODEL>/sc<supercell>_d<disp>_sym-on[_relaxsym-off][_fcsym-off]/
    ├── run_config.txt                            copy of the config used
    ├── relax.log, relaxed.traj, relaxed.cif      stage 1
    ├── phonopy_disp.yaml, FORCE_SETS             stage 2
    ├── force_constants.hdf5 / FORCE_CONSTANTS    stage 2, if WRITE_FORCE_CONSTANTS
    ├── band.hdf5 / band.yaml                     stage 3 (HDF5 on / off)
    ├── thermal_properties.yaml, irreps.yaml      stage 4, if switched on
    └── band_structure.png                        this run against its reference
```

The runner no longer makes comparison figures of several models (`RUN_PLOTS`, `phonon_loader`, `plot_band_comparison` and `phonopy_params.yaml` are gone). Use `MLIP_plotter.py`.

## 7. Which function each step relies on

| Runner step | Function | File | Needs on disk |
|---|---|---|---|
| Settings | `CONFIG`, `find_config_file` | config | `config_file.txt` or `--config` file |
| Load structures | `cif_parser` | parser | `materials_cif/*.cif` |
| Load references | `reference_data_parser` (→ `mp_parser`, `phonon_db_parser`, `check_reference_cell`) | parser | `reference_data/phononDB/`, `reference_data/materials_project_reference_data/` |
| Load model | `MODELS[MODEL]["function"]` → `make_mace` / `make_sevennet` / `make_prophet` | parser | model weights |
| Stage timing | `start_timer`, `performance_log` | processor | `results/performance_log.csv` |
| Run folder | `get_run_folder` | processor | |
| Overwrite check | `changed_settings` | config | `run_config.txt` in the run folder |
| Primitive cell | `get_primitive_cell` | processor | |
| Relaxation | `relax_structure` | processor | run folder |
| Symmetry check | `get_space_group` | processor | |
| Force constants | `force_constant_calculator` | processor | run folder |
| Band path | `get_band_path` | processor | |
| Bands | `band_calculator`, `save_bands` | processor | run folder |
| Checks | `lowest_band_frequency`, `gamma_optical_frequency` | processor | |
| Properties | `phonon_properties` | processor | run folder |
| CSV row | `results_table` | processor | `results/summary.csv` |
| Band plot | `plot_run_bands` → `plot_bands` | processor | run folder |

## 8. Performance log

Up to five stages are timed: the model load (once per run, material `all`), and the relaxation, force constants, band structure and (when it writes something) phonon properties of each material. Each adds a row to `results/performance_log.csv`; the functions are described in section 10 of `MLIP_processor_README.md`. `summary.csv` keeps its four time columns (`load_s`, `relax_s`, `forces_s`, `bands_s`).
