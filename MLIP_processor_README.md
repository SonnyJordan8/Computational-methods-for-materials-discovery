# MLIP_processor.py

**Role in the workflow:** does the calculations and writes the outputs. Every function here works on objects that are handed to it (an ASE `Atoms`, a calculator, a `Phonopy` object). It never chooses a model or a material itself; the runner does that.

It is imported by `MLIP_runner.py` and `MLIP_plotter.py`. It imports `MLIP_config.py` (for its settings) but not `MLIP_parser.py`, so it does not know which MLIP it is using. Any ASE calculator works.

```
Atoms (from CIF)
   │  get_primitive_cell
   ▼
primitive Atoms ──► relax_structure ──► relaxed Atoms ──► get_space_group (symmetry check)
                                             │
                                             ▼
                               force_constant_calculator ──► Phonopy object
                                             │
              get_band_path ──► band_calculator ──► bands dict ──► lowest_band_frequency
                                             │                     gamma_optical_frequency
                                             │                     save_bands
                                             ├──► phonon_properties (thermal properties, irreps)
                                             ├──► results_table ──► results/summary.csv
                                             └──► plot_run_bands ──► plot_bands ──► band_structure.png

load_run_phonon (saved run + its run_config.txt) ──► Phonopy object        (used by MLIP_plotter.py)

start_timer ──► (one stage runs) ──► performance_log ──► results/performance_log.csv
```

---

## 1. What the file needs

### Python packages

| Package | Used for |
|---|---|
| `numpy` | Forces, lowest and highest frequencies |
| `csv` | Writing `summary.csv` and `performance_log.csv` |
| `os`, `time` | Timing each stage, counting cores, writing `irreps.yaml` in the run folder |
| `matplotlib` | Band structure figures |
| `ase` (`Atoms`, `FrechetCellFilter`, `write`, `FIRE`, `BFGS`, `LBFGS`, `BFGSLineSearch`, `FixSymmetry`) | Relaxation and turning phonopy supercells back into ASE structures |
| `pymatgen` (`AseAtomsAdaptor`, `SpacegroupAnalyzer`, `HighSymmKpath`) | Space group, primitive cell and band path |
| `phonopy` (`load`, `Phonopy`, `PhonopyAtoms`, `write_FORCE_SETS`, `write_FORCE_CONSTANTS`, `write_force_constants_to_hdf5`, `get_band_qpoints_and_path_connections`) | Displacements, force constants, band structure, saving and reloading runs |
| `MLIP_config` (`CONFIG`, `read_config`) | All settings; reading a run's saved `run_config.txt` |

### Settings at the top of the file

All are read from `config_file.txt` through `CONFIG`; nothing is hard-coded except `RUN_CONFIG_FILE`. See the top-level `README.md` for what each config setting means.

| Constant | Config setting | Used by |
|---|---|---|
| `RESULTS_FOLDER` | `RESULTS_FOLDER` | every output |
| `OPTIMIZER`, `RELAX_CELL`, `FMAX`, `ENERGY_TOLERANCE`, `MAX_STEPS`, `RELAX_SYMMETRY` | `OPTIMIZER`, `RELAX_CELL`, `RELAX_FMAX`, `ENERGY_TOLERANCE`, `MAX_STEPS`, `RELAX_SYMMETRY` | `relax_structure`; `RELAX_SYMMETRY` also `get_run_folder` |
| `OPTIMIZERS` | (none) | maps the `OPTIMIZER` name to the ASE class |
| `PRIMITIVE_AXES`, `IS_PLUSMINUS`, `SYMMETRY`, `SYMMETRY_TOLERANCE`, `FC_CALCULATOR`, `FC_SYMMETRY`, `PHONOPY_LOG_LEVEL` | `PRIMITIVE_AXES`, `PM`, `SYMMETRY`, `SYMMETRY_TOLERANCE`, `FC_CALCULATOR`, `FC_SYMMETRY`, `PHONOPY_LOG_LEVEL` | `force_constant_calculator`; `SYMMETRY`, `FC_SYMMETRY` also `get_run_folder` |
| `BAND_POINTS` | `BAND_POINTS` | `band_calculator` |
| `MESH`, `TPROP`, `TMIN`, `TMAX`, `TSTEP`, `IRREPS`, `SHOW_IRREPS` | same names | `phonon_properties` |
| `HDF5`, `WRITE_FORCE_CONSTANTS` | same names | `force_constant_calculator`, `save_bands` |
| `DISPLACEMENTS_FILE`, `FORCE_SETS_FILE` | same names (`phonopy_disp.yaml`, `FORCE_SETS`) | saving and reloading runs |
| `RUN_CONFIG_FILE` | (fixed: `run_config.txt`) | the copy of the config saved in each run folder (written by the runner, read by `load_run_phonon`) |
| `CSV_COLUMNS` | (none) | header row of `results/summary.csv` (22 names) |
| `PERFORMANCE_COLUMNS` | (none) | header row of `results/performance_log.csv` (12 names) |

### Folder layout this file creates

```
results/
├── summary.csv
├── performance_log.csv
└── <material>/
    ├── comparison_<plot file name>.png             written by MLIP_plotter.py
    └── <model>/
        └── sc<supercell>_d<displacement>_sym-on[...]/   see get_run_folder
            ├── run_config.txt                       copy of the config (written by the runner)
            ├── relax.log, relaxed.traj, relaxed.cif stage 1
            ├── phonopy_disp.yaml, FORCE_SETS        stage 2
            ├── force_constants.hdf5                 stage 2, if WRITE_FORCE_CONSTANTS (FORCE_CONSTANTS if HDF5 is off)
            ├── band.hdf5                            stage 3 (band.yaml if HDF5 is off)
            ├── thermal_properties.yaml              stage 4, if TPROP
            ├── irreps.yaml                          stage 4, if IRREPS is set
            └── band_structure.png                   the runner's plot of this run
```

---

## 2. Folder and symmetry helpers

### `get_run_folder(material_name, model_name, supercell, displacement)`

**What it does:** builds the path for one run. It is the **only** place a run folder's name is built; every other function that needs a run's folder or name calls it. It does not create the folder.

**The name:** `sc<supercell>_d<displacement>_sym-on` or `_sym-off` (`SYMMETRY`), then `_relaxsym-off` if `RELAX_SYMMETRY` is off and `_fcsym-off` if `FC_SYMMETRY` is off. For example:

| Settings | Folder |
|---|---|
| all symmetry on | `results/Si/MACE-MP-0-medium/sc2x2x2_d0.03_sym-on` |
| `SYMMETRY` off | `.../sc2x2x2_d0.03_sym-off` |
| `RELAX_SYMMETRY` and `FC_SYMMETRY` off | `.../sc2x2x2_d0.03_sym-on_relaxsym-off_fcsym-off` |

`supercell` is the label from `supercell_label` in `MLIP_config.py` (`2x2x2`, or `m` + nine numbers for a non-diagonal matrix).

**Good to know:** runs with different symmetry settings get different folders and never overwrite each other. Runs that differ in another calculation setting (e.g. `PM`, `FC_CALCULATOR`) share a folder name; the runner's overwrite check stops the second one instead of overwriting.

### `get_space_group(material)`

Returns `(symbol, number)` of the space group of an ASE structure, e.g. `("Fd-3m", 227)`, using pymatgen's `SpacegroupAnalyzer` with its default tolerance (`symprec = 0.01`). Used for the symmetry check after relaxation.

### `get_primitive_cell(material)`

Takes any ASE structure (usually the conventional cell from a CIF) and returns the **standard primitive cell** as a new ASE `Atoms` object (e.g. 8-atom conventional Si becomes 2-atom primitive Si), using pymatgen's `get_primitive_standard_structure()`.

**Why it matters:** fewer atoms means fewer displaced supercells; and the standard primitive cell follows the Setyawan–Curtarolo convention that `HighSymmKpath` uses in `get_band_path`, so the band path coordinates are correct for it.

### `get_band_path(primitive_material)`

**What it does:** finds the high-symmetry path through the Brillouin zone with pymatgen's `HighSymmKpath` (Setyawan–Curtarolo) and returns it in the form phonopy needs.

**Returns:**
- `band_path`: a list of continuous sections, each a list of `[qx, qy, qz]` fractional coordinates;
- `band_labels`: one flat list of labels in the same order, with Greek labels such as `\Gamma` wrapped in `$...$` for matplotlib.

**Used by:** `band_calculator`, `plot_bands`.

---

## 3. Stage 1: relaxation

### `relax_structure(material, calc, folder)`

**What it does:** relaxes the structure with the MLIP, so the phonons are calculated at that model's own equilibrium geometry. `material` is changed in place.

**Steps:**
1. Attaches the calculator.
2. If `RELAX_SYMMETRY`: attaches ASE's `FixSymmetry` constraint, which removes force and cell changes that would break the starting space group. It uses ASE's own default tolerance (`symprec = 0.01`), not `SYMMETRY_TOLERANCE`.
3. If `RELAX_CELL`: wraps the structure in `FrechetCellFilter`, so the cell shape and volume relax as well as the atom positions; otherwise only the positions relax.
4. Runs the optimiser chosen by `OPTIMIZER` (`OPTIMIZERS[OPTIMIZER]`) one step at a time (`irun` with `fmax=0`, so ASE does not stop by itself). After each step it checks:
   - the largest force (measured as ASE does, including the cell when it relaxes) is below `FMAX` (`RELAX_FMAX`), **and**
   - the energy change per atom since the last step is below `ENERGY_TOLERANCE` (`ENERGY_TOLERANCE = 0` switches this check off).

   Both true: converged, stop. Otherwise it stops after `MAX_STEPS`.
5. Writes the final structure to `relaxed.cif`.

**Returns:** `(converged, steps, energy_change)`: `True`/`False`, the number of optimiser steps, and the last energy change per atom (eV).

**Files written:** `relax.log`, `relaxed.traj`, `relaxed.cif`.

**Good to know:** the `FixSymmetry` constraint is left on the structure after relaxation (Sonny's choice, 23 Nov, to discuss with the supervisor). So when `RELAX_SYMMETRY` is on, the runner's `max_force` is the symmetry-averaged force, and pymatgen prints a warning about unsupported constraints during the symmetry check. Neither changes the phonons.

---

## 4. Stage 2: force constants

### `force_constant_calculator(material, calc, supercell, displacement, folder)`

**What it does:** the finite-displacement phonon calculation. phonopy decides which atoms to move, the MLIP calculates the forces, and phonopy turns the forces into force constants.

**Steps:**
1. Converts the relaxed ASE structure to a `PhonopyAtoms` object.
2. Creates `Phonopy` with `supercell_matrix = supercell` (the 3×3 matrix from `DIM`), `primitive_matrix = PRIMITIVE_AXES`, `symprec = SYMMETRY_TOLERANCE`, `is_symmetry = SYMMETRY`, `log_level = PHONOPY_LOG_LEVEL`.
3. `generate_displacements(distance=displacement, is_plusminus=IS_PLUSMINUS)`.
4. Prints the number of displaced supercells and saves `phonopy_disp.yaml` (structure and displacements, no forces) **before** the forces, so it survives a crash in the force loop.
5. For each displaced supercell: rebuilds it as an ASE `Atoms`, attaches the calculator, stores `get_forces()`.
6. `produce_force_constants(fc_calculator=FC_CALCULATOR)`; then `symmetrize_force_constants()` if `FC_SYMMETRY`.
7. Writes the forces to `FORCE_SETS` (`write_FORCE_SETS`).
8. If `WRITE_FORCE_CONSTANTS`: writes `force_constants.hdf5` (`HDF5` on) or `FORCE_CONSTANTS` (off).

**Relies on:** the **relaxed primitive cell** from `relax_structure` (leftover forces would contaminate the force constants); an ASE calculator; `supercell` (matrix) and `displacement` from the runner.

**Returns:** the `Phonopy` object, with force constants.

**Good to know:**
- `supercell` must be the 3×3 matrix (`SUPERCELL_MATRIX` in the runner), not the `2x2x2` label; the label gives "supercell_matrix shape has to be (3,) or (3, 3)".
- `FC_CALCULATOR = symfc` or `alm` needs the `symfc` or `alm` package installed in that environment.
- With `PRIMITIVE_AXES = AUTO`, phonopy can reorder the axes of rhombohedral and hexagonal cells relative to pymatgen's, which misplaces band labels. There is no warning for this in the current code.

### `load_run_phonon(run_folder)`

**What it does:** rebuilds a saved run as a `Phonopy` object from its `phonopy_disp.yaml` and `FORCE_SETS`, with the force constants refitted **using the settings the run was calculated with**.

**Steps:**
1. Returns `None` if either file is missing.
2. Reads the run's `run_config.txt` with `read_config(..., strict=False)` (so copies written by older versions of the workflow still read).
3. Takes `PRIMITIVE_AXES`, `SYMMETRY`, `SYMMETRY_TOLERANCE`, `FC_CALCULATOR` and `FC_SYMMETRY` from it. Any missing (or no `run_config.txt` at all) take the current config's value, with a note naming them.
4. `phonopy.load(...)` with those settings and `log_level = PHONOPY_LOG_LEVEL`.

**Used by:** `MLIP_plotter.py`. Changing the config later does not change how an old run is rebuilt.

---

## 5. Stage 3: band structure

### `band_calculator(phonon, band_path, band_labels)`

**What it does:** calculates the frequencies along the band path for any `Phonopy` object (an MLIP run or a reference).

**Steps:** `get_band_qpoints_and_path_connections` fills in `BAND_POINTS` q-points per section and works out which sections join; `phonon.run_band_structure(...)` calculates the frequencies.

**Returns a dictionary:** `"distances"` (x-axis positions per section), `"frequencies"` (per section, shape (q-points, bands), THz), `"connections"` (whether each section joins the next).

### `save_bands(phonon, folder)`

Writes the band structure from the last `band_calculator` call on this `phonon`: `band.hdf5` if `HDF5`, otherwise `band.yaml`. (Its docstring still says `band.yaml` only.)

### `lowest_band_frequency(bands)`

Returns the single lowest frequency (THz) anywhere on the band path. The runner compares it with `IMAGINARY_LIMIT` to flag imaginary modes.

### `gamma_optical_frequency(phonon)`

Calculates the frequencies at Γ and returns the **highest**, in THz. Used for the single-number comparison with the reference (`gamma_percent_diff`). Does not need `band_calculator` first.

---

## 6. Stage 4: phonon properties

### `phonon_properties(phonon, folder)`

**What it does:** the optional post-processing switched on in the config.
- `TPROP`: `run_mesh(MESH)`, then `run_thermal_properties(TMIN, TMAX, TSTEP)` (free energy, entropy, heat capacity), written to `thermal_properties.yaml`.
- `IRREPS` set (a q-point and optional degeneracy tolerance): irreducible representations at that q-point, written to `irreps.yaml` (and printed if `SHOW_IRREPS`). phonopy always writes this file into the current folder, so the function changes into the run folder while it writes.

**Returns:** the list of files written (empty if both are off).

**Good to know:** an even `MESH` such as `20 20 20` makes phonopy print a `MeshSymmetryFallbackWarning`: the shifted mesh loses the symmetry reduction, so it is slower, but the thermal properties are still right.

---

## 7. Results table

### `results_table(row)`

**What it does:** appends one row to `results/summary.csv`, writing the `CSV_COLUMNS` header first if the file is new.

**Relies on:** the runner passing the values **in the same order as `CSV_COLUMNS`**. Nothing checks this.

| Column | Meaning |
|---|---|
| `date_time` | When the row was written |
| `material`, `model`, `supercell`, `displacement_A` | What was run (`supercell` is the label, e.g. `2x2x2`) |
| `status` | `OK`, or `CRASHED in <stage>: <error>` |
| `relax_converged`, `relax_steps`, `max_force_eV_per_A` | Relaxation quality |
| `space_group_before`, `space_group_after`, `symmetry_kept` | Symmetry check |
| `volume_per_atom_A3` | Relaxed volume per atom |
| `gamma_optical_thz`, `ref_gamma_optical_thz`, `gamma_percent_diff` | Γ-point comparison with the reference (blank if no reference) |
| `lowest_freq_thz`, `imaginary_modes` | Dynamical stability check |
| `load_s`, `relax_s`, `forces_s`, `bands_s` | Wall-clock time of each stage (from `performance_log`) |

**Good to know:** the symmetry settings are not in the table, so runs with symmetry on and off give rows that look the same. A `run` column with the folder name is on the fix list.

---

## 8. Plots

### `plot_bands(datasets, reference_phonon, band_path, band_labels, title, figure_file, reference_label="REF", styles=None)`

**What it does:** draws the band structure of every `Phonopy` object in `datasets` (`{label: Phonopy}`) on one figure, with the reference underneath, and saves it.

**Steps:**
1. **Reference first** (if not `None`): bands from `band_calculator`, drawn in black, line width 1.5. Its x-axis positions are reused for everything else.
2. **Each data set:** bands drawn with line width 1.0, in the colour and line style from `styles[label]` if `styles` is given, otherwise matplotlib's colour cycle (`C0`, `C1`, ...) with solid lines. If there was no reference, the first data set's x-axis is used.
3. Only the first line of each data set is labelled, so the legend has one entry each.
4. **Ticks:** at the start and end of every section; where the path jumps (e.g. K to U), the tick gets a combined label `K|U`.
5. Grey lines at the ticks and at zero, y-axis label, title, legend; saved at 200 dpi.

**Returns:** the path of the saved figure.

**Used by:** `plot_run_bands` (runner) and `MLIP_plotter.py`, which passes `styles` to give each model its `MODELS` colour and each run of a model its own line style.

### `plot_run_bands(material_name, model_name, phonon, reference_phonon, band_path, band_labels, supercell, displacement, reference_label="REF", colour=None)`

**What it does:** one run against the reference. Calls `plot_bands` with `{model_name: phonon}`, the run folder's name as the title, and `colour` as a solid-line style (matplotlib's first colour if `colour` is not given). The runner passes `MODELS[MODEL]["colour"]`.

**Saves:** `band_structure.png` in the run folder.

---

## 9. Summary table

| Function | Called by | Relies on | Returns / writes |
|---|---|---|---|
| `get_run_folder` | runner, `plot_run_bands`, plotter | `RESULTS_FOLDER`, symmetry settings | folder path (not created) |
| `get_space_group` | runner | pymatgen | `(symbol, number)` |
| `get_primitive_cell` | runner, plotter | pymatgen | primitive ASE `Atoms` |
| `get_band_path` | runner, plotter | pymatgen, a standard primitive cell | `band_path`, `band_labels` |
| `relax_structure` | runner | calculator, relaxation settings, existing folder | `(converged, steps, energy_change)`; `relax.log`, `relaxed.traj`, `relaxed.cif`; changes `material` in place |
| `force_constant_calculator` | runner | relaxed primitive cell, calculator, phonopy settings | `Phonopy` object; `phonopy_disp.yaml`, `FORCE_SETS`, force constants file |
| `load_run_phonon` | plotter | `phonopy_disp.yaml`, `FORCE_SETS`, `run_config.txt` | `Phonopy` object or `None` |
| `band_calculator` | runner, `plot_bands` | `Phonopy` with force constants, `get_band_path` output, `BAND_POINTS` | `bands` dictionary |
| `save_bands` | runner | `band_calculator` run first | `band.hdf5` / `band.yaml` |
| `phonon_properties` | runner | `Phonopy` with force constants, TPROP/IRREPS settings | list of files written |
| `lowest_band_frequency` | runner | `bands` dictionary | float, THz |
| `gamma_optical_frequency` | runner | `Phonopy` with force constants | float, THz |
| `results_table` | runner | `CSV_COLUMNS`, row in the right order | appends to `summary.csv` |
| `plot_bands` | `plot_run_bands`, plotter | `band_calculator` | PNG figure path |
| `plot_run_bands` | runner | `plot_bands`, `get_run_folder` | `band_structure.png` path |
| `start_timer` | runner | `time` | `(wall time, CPU time)` at this moment |
| `get_cores_available` | `performance_log` | `os` | number of cores |
| `performance_log` | runner | `start_timer` value, `PERFORMANCE_COLUMNS` | wall time in seconds; appends to `performance_log.csv` |

---

## 10. Performance log

The runner wraps each stage in a timer, so every stage of every run gets one row in `results/performance_log.csv`.

```python
timer = start_timer()
converged, steps, energy_change = relax_structure(atoms, calc, folder)
relax_time = performance_log(material_name, MODEL, DEVICE, SUPERCELL, DISPLACEMENT, stage, timer)
```

### `start_timer()`

Returns the wall-clock time (`time.time()`) and this process's CPU time (`time.process_time()`) as a pair. Call it just before a stage and pass the result to `performance_log` afterwards.

### `get_cores_available()`

Returns the number of CPU cores this process may use: on Linux `os.sched_getaffinity` (the cores given to the job on a supercomputer, not the whole node); on Windows and Mac `os.cpu_count` (every core in the machine).

### `performance_log(material_name, model_name, device, supercell, displacement, stage, timer)`

Works out the stage's wall time and CPU time, `cores_used` = CPU time ÷ wall time, and `cpu_usage_percent` = cores used ÷ cores available × 100; appends one row to `results/performance_log.csv` (header first if new); returns the wall time in seconds.

| Column | Meaning |
|---|---|
| `date_time` | When the row was written (the end of the stage) |
| `material` | Material name, or `all` for the model load |
| `model`, `device`, `supercell`, `displacement_A` | What was run |
| `stage` | `model load`, `relaxation`, `force constants`, `band structure` or `phonon properties` |
| `wall_time_s` | Time taken |
| `cpu_time_s` | Compute time, summed over all cores |
| `cores_used` | CPU time ÷ wall time |
| `cores_available` | From `get_cores_available` |
| `cpu_usage_percent` | Cores used ÷ cores available × 100 |
