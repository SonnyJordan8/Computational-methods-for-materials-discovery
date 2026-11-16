# MLIP_processor.py

**Role in the workflow:** does the calculations and writes the outputs. Every function here works on objects that are handed to it (an ASE `Atoms`, a calculator, a `Phonopy` object). It never chooses a model or a material itself; the runner does that.

It is imported by `MLIP_runner.py`. It does not import `MLIP_parser.py`, so it does not know which MLIP it is using. Any ASE calculator works.

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
                                             ▼                     save_bands
                                       results_table ──► results/summary.csv

phonon_loader ──► plot_band_comparison ──► results/<material>/band_comparison_*.png
```

---

## 1. What the file needs

### Python packages

| Package | Used for |
|---|---|
| `numpy` | Finding the lowest and highest frequencies |
| `csv` | Writing `summary.csv` |
| `matplotlib` | The band comparison figure |
| `ase` (`Atoms`, `FrechetCellFilter`, `FIRE`, `write`) | Relaxation and turning phonopy supercells back into ASE structures |
| `pymatgen` (`AseAtomsAdaptor`, `SpacegroupAnalyzer`, `HighSymmKpath`) | Space group, primitive cell and band path |
| `phonopy` (`Phonopy`, `PhonopyAtoms`, `get_band_qpoints_and_path_connections`) | Displacements, force constants, band structure |

**Imported but not used at the moment** (safe to delete): `ase.build.bulk`, `ase.io.read`, `ase.dft.kpoints.BandPath`, `pymatgen.io.vasp.inputs.Kpoints`, `read_crystal_structure`, `write_crystal_structure`.

### Settings at the top of the file

These are the same for every run. Settings that change between runs (model, supercell, displacement) live in the runner.

| Constant | Value | Meaning | Source noted in code |
|---|---|---|---|
| `RESULTS_FOLDER` | `results/` next to this file | Where everything is saved | |
| `IS_PLUSMINUS` | `True` | Displace each atom in both + and - directions | Orb-v3 paper, Appendix G |
| `FMAX` | `0.005` eV/Å | Relaxation stops when the largest force is below this | Loew et al. 2025 |
| `MAX_STEPS` | `500` | Stops a relaxation that is stuck | Loew et al. 2025 / MatRIS paper |
| `BAND_POINTS` | `51` | q-points per section of the band path | phonopy documentation |
| `CSV_COLUMNS` | 22 names | Header row of `results/summary.csv` | |

### Folder layout this file creates

```
results/
├── summary.csv
└── <material>/
    ├── band_comparison_sc<supercell>_d<displacement>.png
    └── <model>/
        └── sc<supercell>_d<displacement>/
            ├── relax.log
            ├── relaxed.traj
            ├── relaxed.cif
            ├── phonopy_disp.yaml
            ├── phonopy_params.yaml
            └── band.yaml
```

---

## 2. Folder and symmetry helpers

### `get_run_folder(material_name, model_name, supercell, displacement)`

**What it does:** builds the path for one run, e.g. `results/Si/MACE-MP-0-medium/sc2_d0.03`. It only returns the path. It does not create the folder; the runner does that with `folder.mkdir(...)`.

**Relies on:** `RESULTS_FOLDER`.

**Good to know:** `phonon_loader` rebuilds the same `sc<supercell>_d<displacement>` name to find saved results, so the two must stay in step if you ever change the naming.

### `get_space_group(material)`

**What it does:** returns `(symbol, number)` of the space group of an ASE structure, e.g. `("Fd-3m", 227)`.

**Relies on:** pymatgen `SpacegroupAnalyzer` with its default tolerance (`symprec = 0.01`).

**Used for:** the symmetry check after relaxation. If the MLIP has distorted the cell, the space group will have changed.

**Good to know:** this is the same calculation `cif_parser` does in the parser file, so the "before" and "after" space groups are found the same way.

### `get_primitive_cell(material)`

**What it does:** takes any ASE structure (usually the conventional cell from a CIF) and returns the **standard primitive cell** as a new ASE `Atoms` object. For example, 8-atom conventional Si becomes 2-atom primitive Si.

**Relies on:** pymatgen `get_primitive_standard_structure()`.

**Why it matters:**
- Fewer atoms means fewer displaced supercells and a faster calculation.
- The "standard" primitive cell follows the Setyawan-Curtarolo convention, which is the same convention `HighSymmKpath` uses in `get_band_path`. The band path coordinates are only correct for a cell in this setting.

**Good to know:** the returned `Atoms` is a new object. The `info` dictionary from `cif_parser` (space group, formula) is **not** carried over, which is why the runner reads `space_group_symbol` from the original structure.

### `get_band_path(primitive_material)`

**What it does:** finds the high-symmetry path through the Brillouin zone for the structure and returns it in the form phonopy needs.

**Steps:**
1. `HighSymmKpath` works out the high-symmetry points and the path for this lattice type.
2. The path comes as a list of sections. Each section is a continuous run of labels, e.g. `["\\Gamma", "X", "W", "K", "\\Gamma", "L", "U", "W", "L", "K"]` and then `["U", "X"]` for FCC.
3. For each label the fractional q-point coordinates are looked up.
4. Labels that start with a backslash (such as `\Gamma`) are wrapped in `$...$` so matplotlib draws the Greek letter.

**Relies on:** pymatgen `HighSymmKpath`; a structure that is already the standard primitive cell (from `get_primitive_cell`).

**Returns:**
- `band_path`: a list of sections, each a list of `[qx, qy, qz]` coordinates.
- `band_labels`: one flat list of labels in the same order.

**Used by:** `band_calculator` and `plot_band_comparison`.

**Good to know:** the q-points are in fractional (reduced) coordinates, so the same path can be reused for the reference data and for every model even though each one has a slightly different lattice constant.

---

## 3. Stage 1: relaxation

### `relax_structure(material, calc, folder)`

**What it does:** relaxes the atomic positions **and** the cell with the MLIP, so the phonons are calculated at that model's own equilibrium geometry.

**Steps:**
1. Attaches the calculator: `material.calc = calc`.
2. Wraps the structure in `FrechetCellFilter`, which lets the optimiser change the cell shape and volume as well as the atom positions.
3. Runs the `FIRE` optimiser until the largest force is below `FMAX` or `MAX_STEPS` is reached.
4. Writes the final structure to `relaxed.cif`.

**Relies on:**
- An ASE calculator (from `MODELS` in the parser file).
- A run folder that already exists.
- `FMAX`, `MAX_STEPS`.

**Returns:** `(converged, steps)`: `True`/`False`, and the number of optimiser steps taken.

**Files written:** `relax.log`, `relaxed.traj`, `relaxed.cif`.

**Good to know:**
- **The `material` object is changed in place.** After the call, `atoms` in the runner *is* the relaxed structure, and it still has the calculator attached (the runner uses this to read the leftover forces).
- With the cell filter, `fmax` applies to the atomic forces and to the stress (converted to force-like units), so both must be small for convergence.
- If the relaxation does not converge, the workflow carries on and records `relax_converged = False` in the CSV. It does not stop.
- The docstring says `relax.traj` but the file is actually saved as `relaxed.traj`.
- `relax.log` is appended to if you re-run the same settings, while `relaxed.traj` is overwritten.

---

## 4. Stage 2: force constants

### `force_constant_calculator(material, calc, supercell, displacement, folder)`

**What it does:** the finite-displacement phonon calculation. phonopy decides which atoms to move, the MLIP calculates the forces, and phonopy turns the forces into force constants.

**Steps:**
1. Converts the relaxed ASE structure to a `PhonopyAtoms` object (symbols, cell, fractional positions).
2. Creates a `Phonopy` object with:
   - `supercell_matrix` = `supercell` × `supercell` × `supercell` (diagonal);
   - `primitive_matrix="P"`, which means "the cell I gave you is already primitive, do not change it".
3. `generate_displacements(distance=displacement, is_plusminus=IS_PLUSMINUS)`: phonopy uses the crystal symmetry to find the smallest set of displacements needed.
4. Prints the number of displaced supercells and saves `phonopy_disp.yaml`.
5. For each displaced supercell: rebuilds it as an ASE `Atoms`, attaches the calculator, and stores `get_forces()`.
6. Gives the forces to phonopy and calls `produce_force_constants()`.
7. Saves `phonopy_params.yaml` including the force constants.

**Relies on:**
- The **relaxed primitive cell** from `relax_structure`. If the structure is not relaxed, leftover forces contaminate the force constants.
- An ASE calculator.
- `supercell` and `displacement` from the runner (`SUPERCELL`, `DISPLACEMENT`).
- `IS_PLUSMINUS`.

**Returns:** the `Phonopy` object, with force constants.

**Files written:**
- `phonopy_disp.yaml`: structure and displacements only (written *before* the forces, so it survives a crash in the force loop).
- `phonopy_params.yaml`: structure, displacements, forces and force constants. This is the file `phonon_loader` reads for the plots.

**Good to know:**
- The cost of this stage grows with the number of displaced supercells × atoms per supercell. Si with a 2×2×2 supercell needs only 2 displacements (1 displacement × plus/minus); low-symmetry structures need many more.
- The number of atoms in each supercell is `atoms in primitive cell × supercell³`.
- The supercell must be large enough that the force constants have decayed by the cell edge. Supercell size is one of the things the project is testing.
- No Born charges are set here, so there is no non-analytical correction and no LO/TO splitting in the MLIP result.

---

## 5. Stage 3: band structure

### `band_calculator(phonon, band_path, band_labels)`

**What it does:** calculates the phonon frequencies along the band path for any `Phonopy` object (an MLIP result or the reference).

**Steps:**
1. `get_band_qpoints_and_path_connections` fills in `BAND_POINTS` q-points along each segment of the path and works out which segments join on to the next.
2. `phonon.run_band_structure(...)` calculates the frequencies.
3. Picks out the parts needed from phonopy's band structure dictionary.

**Relies on:**
- A `Phonopy` object that has force constants (from `force_constant_calculator`, `phonon_loader`, or the reference parser).
- `band_path` and `band_labels` from `get_band_path`.
- `BAND_POINTS`.

**Returns a dictionary:**
- `"distances"`: list of arrays, the x-axis positions for each segment.
- `"frequencies"`: list of arrays, shape (q-points, bands), in THz.
- `"connections"`: list of `True`/`False`, whether each segment joins the next.

**Good to know:**
- The band structure is also stored inside the `phonon` object, which is why `save_bands` can be called straight afterwards.
- Imaginary modes appear as **negative** frequencies (phonopy's convention).

### `save_bands(phonon, folder)`

**What it does:** writes the band structure to `band.yaml` in the run folder.

**Relies on:** `band_calculator` having been run on the same `phonon` object first. Otherwise there is no band structure to write.

### `lowest_band_frequency(bands)`

**What it does:** returns the single lowest frequency (THz) anywhere on the band path.

**Relies on:** the `bands` dictionary from `band_calculator`.

**Used for:** detecting imaginary modes. The runner compares the value with `IMAGINARY_LIMIT` (-0.05 THz), which is defined **in the runner**, not in this file.

**Good to know:** it only checks q-points on the band path. An instability somewhere else in the Brillouin zone would be missed; a full q-point mesh would be needed to catch it.

### `gamma_optical_frequency(phonon)`

**What it does:** calculates the frequencies at the Γ point (q = 0, 0, 0) and returns the **highest** one, in THz.

**Relies on:** a `Phonopy` object with force constants. It does not need `band_calculator` to have been run.

**Used for:** the single-number comparison with the reference (`gamma_percent_diff` in the CSV). The runner calls it once on the MLIP result and once on the reference.

**Good to know:**
- For Si this is the triply degenerate optical mode (about 15.5 THz experimentally).
- With no non-analytical correction, for polar materials (GaAs, MgO) this is the **TO** frequency, not the LO. The reference is loaded with `is_nac=False`, so both sides are consistent.
- For a material with many atoms it is simply the top of the spectrum at Γ, which may be less meaningful than for Si.

---

## 6. Results table

### `results_table(row)`

**What it does:** appends one row to `results/summary.csv`. If the file does not exist yet, it writes the `CSV_COLUMNS` header first.

**Relies on:**
- `RESULTS_FOLDER`, `CSV_COLUMNS`.
- The runner passing the values **in the same order as `CSV_COLUMNS`**. Nothing checks this.

**The columns:**

| Column | Meaning |
|---|---|
| `date_time` | When the row was written |
| `material`, `model`, `supercell`, `displacement_A` | What was run |
| `status` | `OK`, or `CRASHED in <stage>: <error>` |
| `relax_converged`, `relax_steps`, `max_force_eV_per_A` | Relaxation quality |
| `space_group_before`, `space_group_after`, `symmetry_kept` | Symmetry check |
| `volume_per_atom_A3` | Relaxed volume per atom |
| `gamma_optical_thz`, `ref_gamma_optical_thz`, `gamma_percent_diff` | Γ-point comparison with the reference (blank if no reference) |
| `lowest_freq_thz`, `imaginary_modes` | Dynamical stability check |
| `load_s`, `relax_s`, `forces_s`, `bands_s` | Wall-clock time of each stage |

**Good to know:**
- Rows are **appended**, never replaced. Re-running the same settings adds a second row; use `date_time` to tell them apart.
- A crashed run writes a short row with only the first 6 columns.
- The header is only written when the file is new. If you add a column to `CSV_COLUMNS`, delete or rename the old `summary.csv`, otherwise new rows will not line up with the old header.

---

## 7. Comparison plot

### `phonon_loader(material_name, supercell, displacement)`

**What it does:** finds every model that has saved results for this material at this supercell and displacement, and loads each one as a `Phonopy` object.

**Steps:**
1. Looks in `results/<material>/`. If the folder does not exist, returns an empty dictionary.
2. For each sub-folder (one per model, in alphabetical order), looks for `sc<supercell>_d<displacement>/phonopy_params.yaml`.
3. Loads each file found with `phonopy.load`.

**Relies on:**
- `phonopy_params.yaml` files written by `force_constant_calculator` in earlier runs.
- The folder naming from `get_run_folder`.

**Returns:** `{model name: Phonopy object}`. The model name is the folder name.

**Good to know:** this is what allows the plot to show several models at once even though the runner only calculates one model per run. Run MACE, then run Prophet, and the second run's plot contains both.

### `plot_band_comparison(material_name, phonons, reference_phonon, band_path, band_labels, supercell, displacement)`

**What it does:** draws the band structure of every model on one figure, with the reference in black underneath, and saves it as a PNG.

**Steps:**
1. **Reference first** (if there is one): calculates its bands with `band_calculator` and plots them in black, line width 1.5. Its x-axis positions are stored and reused for everything else.
2. **Each model:** calculates its bands and plots them in matplotlib's default colours (`C0`, `C1`, ...), line width 1.0. If there was no reference, the first model's x-axis is used.
3. Only the first line of each data set is given a label, so the legend has one entry per data set.
4. **Tick marks:** one at the start and end of every segment. Where the path jumps (for example K to U in FCC), the tick gets a combined label such as `K|U`.
5. Adds grey vertical lines at the ticks, a horizontal line at zero, the y-axis label, a title and the legend.
6. Saves at 200 dpi and closes the figure.

**Relies on:**
- `phonons` from `phonon_loader`.
- `reference_phonon` from `reference_data_parser` in the parser file (or `None`).
- `band_path`, `band_labels` from `get_band_path`.
- `band_calculator`.
- The folder `results/<material>/` already existing (it does if `phonon_loader` found anything).

**Returns:** the path of the saved figure, `results/<material>/band_comparison_sc<supercell>_d<displacement>.png`.

**Good to know:**
- **One shared x-axis.** Each model relaxes to a slightly different lattice constant, so the true path lengths differ slightly. The plot uses the first data set's distances for all of them so the high-symmetry points line up. The frequencies are not changed.
- Negative frequencies on the plot are imaginary modes.
- `supercell` and `displacement` are only used for the title and file name. They describe the MLIP runs; the reference data has its own supercell and displacement.
- The reference line may be PBEsol (phononDB) or whatever is in `reference_data/`. The legend just says "Reference".

---

## 8. Summary table

| Function | Called by | Relies on | Returns / writes |
|---|---|---|---|
| `get_run_folder` | runner | `RESULTS_FOLDER` | folder path (not created) |
| `get_space_group` | runner | pymatgen | `(symbol, number)` |
| `get_primitive_cell` | runner | pymatgen | primitive ASE `Atoms` |
| `get_band_path` | runner | pymatgen, a standard primitive cell | `band_path`, `band_labels` |
| `relax_structure` | runner | calculator, `FMAX`, `MAX_STEPS`, existing folder | `(converged, steps)`; `relax.log`, `relaxed.traj`, `relaxed.cif`; changes `material` in place |
| `force_constant_calculator` | runner | relaxed primitive cell, calculator, `IS_PLUSMINUS` | `Phonopy` object; `phonopy_disp.yaml`, `phonopy_params.yaml` |
| `band_calculator` | runner, `plot_band_comparison` | `Phonopy` with force constants, `get_band_path` output, `BAND_POINTS` | `bands` dictionary |
| `save_bands` | runner | `band_calculator` run first | `band.yaml` |
| `lowest_band_frequency` | runner | `bands` dictionary | float, THz |
| `gamma_optical_frequency` | runner | `Phonopy` with force constants | float, THz |
| `results_table` | runner | `CSV_COLUMNS`, row in the right order | appends to `summary.csv` |
| `phonon_loader` | runner | saved `phonopy_params.yaml` files | dict of `Phonopy` objects |
| `plot_band_comparison` | runner | `phonon_loader`, `band_calculator`, reference from parser | PNG figure path |

---

## 9. Planned addition: performance log

**Not in the file yet.** Designed in the conversation of 15 Nov 2026; add this section's functions when they are pasted in.

| Function | What it does | Relies on |
|---|---|---|
| `start_timer()` | Returns the wall-clock time and CPU time at this moment | `time` |
| `get_cores_available()` | Number of cores the process is allowed to use | `os.sched_getaffinity` on Linux, `os.cpu_count` on Windows/Mac |
| `performance_log(material_name, model_name, device, supercell, displacement, stage, timer)` | Appends one row per stage to `results/performance_log.csv` and returns the wall time | `start_timer`, `get_cores_available`, `RESULTS_FOLDER`, `PERFORMANCE_COLUMNS` |

Columns: `date_time, material, model, device, supercell, displacement_A, stage, wall_time_s, cpu_time_s, cores_used, cores_available, cpu_usage_percent`.

- `cores_used` = CPU time ÷ wall time (average number of busy cores).
- `cpu_usage_percent` = cores used ÷ cores available × 100.
- On a GPU run the CPU columns will look low, because CPU time does not see GPU work.
