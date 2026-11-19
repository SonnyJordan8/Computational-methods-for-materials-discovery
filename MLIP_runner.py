from datetime import datetime
import numpy as np

from MLIP_parser import MATERIALS_CIF_FOLDER, PHONON_DB_FOLDER, MP_REFERENCE_FOLDER, MODELS
from MLIP_parser import cif_parser, reference_data_parser
from MLIP_processor import get_run_folder, get_space_group, get_primitive_cell, get_band_path
from MLIP_processor import relax_structure, force_constant_calculator
from MLIP_processor import band_calculator, save_bands, lowest_band_frequency, gamma_optical_frequency
from MLIP_processor import results_table, phonon_loader, plot_band_comparison
from MLIP_processor import start_timer, performance_log
from MLIP_processor import results_table, phonon_loader
from MLIP_processor import plot_run_bands, plot_band_comparison

# Settings
# Change these before each run.
MODEL = "MACE-MP-0-medium"   # one of the keys of MODELS
DEVICE = "cpu"
SUPERCELL = 1
DISPLACEMENT = 0.03
RUN_CALCULATIONS = False      # relax and calculate phonons with MODEL

RUN_PLOTS_RUNS = True                    # one plot per saved run at SUPERCELL and DISPLACEMENT, every model
RUN_PLOTS_MLIP_COMPARISON = True        # every model with saved results at SUPERCELL and DISPLACEMENT on one plot

IMAGINARY_LIMIT = -0.05   # THz, a frequency below this counts as an imaginary mode - Batatia et al. 2025

# Materials in materials_cif that are being calculated
MATERIALS = ["K3MnCrO8"]
# phononDB id for any material that can't be matched automatically by formula + space group,
# e.g. {"Si": "mp-149"}. Materials not listed here are matched automatically
REFERENCES = {}

# Parse
# Materials come from the CIF files, reference data from the phononDB folder or Phonopy_params.yaml files
materials = cif_parser(MATERIALS_CIF_FOLDER)
if MATERIALS is not None:
    missing = [name for name in MATERIALS if name not in materials]
    if missing:
        raise ValueError(f"No CIF file in materials_cif for {missing}. Available: {list(materials.keys())}")
    materials = {name: materials[name] for name in MATERIALS}
print(f"materials: {list(materials.keys())}")

# Reference data comes form three sources currently
references, reference_sources = reference_data_parser(
    materials, PHONON_DB_FOLDER, MP_REFERENCE_FOLDER, requested=REFERENCES
)

# Calculations
if RUN_CALCULATIONS:
    # Load the model once and time it
    timer = start_timer()
    calc = MODELS[MODEL](DEVICE)
    load_time = performance_log("all", MODEL, DEVICE, SUPERCELL, DISPLACEMENT, "model load", timer)
    print(f"loaded {MODEL} in {load_time:.1f} s")

    # Run each material. If one fails, record which stage failed and why, then carry on with the next.
    for material_name in materials:
        print(f"\n===== {material_name} | {MODEL} =====")
        stage = "set up"
        try:
            # Each run gets its own folder, e.g. results/Si/MACE-MP-0-medium/sc3_d0.03
            folder = get_run_folder(material_name, model_name=MODEL, supercell=SUPERCELL, displacement=DISPLACEMENT)
            folder.mkdir(parents=True, exist_ok=True)

            # Space group of the CIF structure, found by the parser
            space_group_before = materials[material_name].info["space_group_symbol"]
            # Reduce the CIF structure to its primitive cell
            atoms = get_primitive_cell(materials[material_name])
            print(f"atoms in primitive cell: {len(atoms)}")

            # Stage 1: relaxation
            stage = "relaxation"
            timer = start_timer()
            converged, steps = relax_structure(atoms, calc, folder)
            relax_time = performance_log(material_name, MODEL, DEVICE, SUPERCELL, DISPLACEMENT, stage, timer)
            # Largest force left on any atom after relaxation
            max_force = float(np.abs(atoms.get_forces()).max())
            volume_per_atom = atoms.get_volume() / len(atoms)

            # Check the space group is the same after relaxation
            stage = "symmetry check"
            space_group_after, space_group_number = get_space_group(atoms)
            symmetry_kept = space_group_after == space_group_before
            if not symmetry_kept:
                print(f"WARNING: space group changed from {space_group_before} to {space_group_after}")

            # Stage 2: force constants
            stage = "force constants"
            timer = start_timer()
            phonon = force_constant_calculator(atoms, calc, supercell=SUPERCELL, displacement=DISPLACEMENT, folder=folder)
            forces_time = performance_log(material_name, MODEL, DEVICE, SUPERCELL, DISPLACEMENT, stage, timer)

            # Stage 3: band structure
            stage = "band structure"
            timer = start_timer()
            band_path, band_labels = get_band_path(atoms)
            bands = band_calculator(phonon, band_path, band_labels)
            save_bands(phonon, folder)
            lowest_frequency = lowest_band_frequency(bands)
            gamma_optical = gamma_optical_frequency(phonon)
            bands_time = performance_log(material_name, MODEL, DEVICE, SUPERCELL, DISPLACEMENT, stage, timer)
            # Imaginary modes show up as negative frequencies
            imaginary_modes = lowest_frequency < IMAGINARY_LIMIT

            # Compare with the reference data, as a percentage difference
            stage = "reference comparison"
            ref_gamma_optical = ""
            gamma_diff = ""
            if material_name in references:
                ref_gamma_optical = gamma_optical_frequency(references[material_name])
                gamma_diff = (gamma_optical - ref_gamma_optical) / ref_gamma_optical * 100

            # Print the results
            print(f"relaxation converged: {converged} ({steps} steps, max force {max_force:.5f} eV/A)")
            print(f"space group: {space_group_before} -> {space_group_after}")
            print(f"volume per atom: {volume_per_atom:.3f} A^3")
            print(f"Gamma optical frequency: {gamma_optical:.2f} THz")
            if gamma_diff != "":
                print(f"difference from reference: {gamma_diff:.1f} %")
            print(f"lowest frequency: {lowest_frequency:.4f} THz (imaginary modes: {imaginary_modes})")
            print(f"times: relax {relax_time:.1f} s, forces {forces_time:.1f} s, bands {bands_time:.1f} s")

            # Add one row to the results table, in the same order as CSV_COLUMNS
            stage = "saving results"
            results_table([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), material_name, MODEL, SUPERCELL, DISPLACEMENT, "OK",
                converged, steps, max_force,
                space_group_before, space_group_after, symmetry_kept,
                volume_per_atom,
                gamma_optical, ref_gamma_optical, gamma_diff,
                lowest_frequency, imaginary_modes,
                round(load_time, 2), round(relax_time, 2), round(forces_time, 2), round(bands_time, 2),
            ])
            print(f"  saved to {folder}")

        except Exception as error:
            status = f"CRASHED in {stage}: {type(error).__name__}: {error}"
            print(f"{status}")
            results_table([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), material_name, MODEL,
                SUPERCELL, DISPLACEMENT, status,
            ])

# Plots
# All plots use SUPERCELL and DISPLACEMENT from the settings, and the saved results of every
# earlier run, so they can be remade with RUN_CALCULATIONS = False.
if RUN_PLOTS_RUNS or RUN_PLOTS_MLIP_COMPARISON:
    for material_name in materials:
        print(f"\n===== plots: {material_name} =====")
        try:
            reference_phonon = references.get(material_name)
            reference_label = reference_sources[material_name]

            # Band path from the primitive cell of the CIF structure, shared by every data set
            band_path, band_labels = get_band_path(get_primitive_cell(materials[material_name]))

            # Every model with saved results at this supercell and displacement
            phonons = phonon_loader(material_name, SUPERCELL, DISPLACEMENT)
            if len(phonons) == 0:
                print(f"no saved results at sc{SUPERCELL}_d{DISPLACEMENT}")

            # 1. One plot per run
            if RUN_PLOTS_RUNS:
                for model_name, phonon in phonons.items():
                    figure_file = plot_run_bands(material_name, model_name, phonon, reference_phonon,
                                                 band_path, band_labels, SUPERCELL, DISPLACEMENT, reference_label)
                    print(f"run plot ({model_name}): {figure_file}")

            # 2. All models on one plot
            if RUN_PLOTS_MLIP_COMPARISON and len(phonons) > 0:
                figure_file = plot_band_comparison(material_name, phonons, reference_phonon, band_path, band_labels,
                                                   SUPERCELL, DISPLACEMENT, reference_label)
                print(f"model comparison {list(phonons.keys())}: {figure_file}")

            
        except Exception as error:
            print(f"{material_name}: plot CRASHED: {type(error).__name__}: {error}")

