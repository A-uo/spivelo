# spivelo
A Generative Spatial RNA Velocity Inference Method for Resolving Tissue-Level Cell Fate Dynamics

SpiVelo combines single-cell and spatial transcriptomic information to analyze RNA dynamics in tissue. This source distribution contains two components: **SV-Integration**, which learns shared transcriptional programs and a soft cell-to-location mapping, and **SV-Velo**, which fits a hierarchical RNA kinetic model using Pyro variational inference.

The executable entry point is `Spivelo_test.ipynb`. The description below follows that notebook's actual workflow.

## Contents

```text
code/
├── README.md
├── environment.yml
├── Spivelo_test.ipynb
├── SV-Integration/
│   ├── mapping_utils.py       # Preprocessing and mapping interface
│   ├── mapping_optimizer.py   # Shared-program mapping optimization
│   ├── program_selection.py   # Program-selection utilities
│   ├── utils.py               # Projection and annotation utilities
│   └── ...                    # Simulation, analysis, and plotting utilities
└── SV-velo/
    ├── _hvgk_model.py          # HVGK high-level interface
    ├── _hvgk_pyro_module.py    # Generative model
    ├── _kinetics.py            # RNA kinetic calculations
    ├── _pyro_infra.py          # AutoNormal guide and training infrastructure
    ├── _validation.py         # Input validation
    ├── _benchmark.py          # Evaluation utilities
    └── ...                    # Initialization, diagnostics, and experiments
```

## Environment setup

Install Conda or Miniforge, open a terminal in this `code` directory, and run:

```bash
conda env create -f environment.yml
conda activate spivelo
python -m ipykernel install --user --name spivelo --display-name "Python (SpiVelo)"
jupyter lab Spivelo_test.ipynb
```

Select **Python (SpiVelo)** as the notebook kernel. The core scientific-library versions in `environment.yml` match the existing development environment. A fresh installation of the complete YAML has not been validated; JupyterLab and Leiden clustering dependencies are included for notebook execution and program selection.

The notebook automatically uses an NVIDIA GPU if `torch.cuda.is_available()` is true, otherwise it uses the CPU. Check availability with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

A GPU installation also requires a compatible NVIDIA driver and PyTorch build. Use the official PyTorch installation instructions for your platform if the default installation does not provide CUDA support. Full-batch training and the dense cell-to-location mapping can require substantial memory for large inputs.

## Input data

Supply two AnnData `.h5ad` files:

| Input                   | Required contents                                            |
| ----------------------- | ------------------------------------------------------------ |
| Single-cell reference   | Unique cell and gene names; nonnegative expression in `X` or a selected layer; raw integer `layers['spliced']` and `layers['unspliced']`; complete cell-type annotations in `obs[CELL_TYPE_KEY]` |
| Spatial transcriptomics | Unique location and gene names; nonnegative expression in `X` or a selected layer; finite two-dimensional coordinates in `obsm[SPATIAL_KEY]` |

The inputs must share at least two nonzero genes. Resolve duplicated gene names explicitly before running. 

## Configure and run

Edit the first configuration cell of `Spivelo_test.ipynb`:

```python
SC_PATH = ROOT.parent / 'dataset' / 'reference_sc_raw.h5ad'
SP_PATH = ROOT.parent / 'dataset' / 'spatial_expression.h5ad'
CELL_TYPE_KEY = 'cell_type'
SPATIAL_KEY = 'spatial'   # Or 'X_spatial', depending on the input file.
N_PROGRAMS = 8
INTEGRATION_EPOCHS = 1000
VELOCITY_EPOCHS = 200
OUTPUT = ROOT / 'sv_integration_velocity_output'
```

These filenames are placeholders: replace them with actual files. `SC_PATH` and `SP_PATH` are initially `None`, so the notebook intentionally stops until inputs have been supplied. Optional `SC_EXPRESSION_LAYER` and `SP_EXPRESSION_LAYER` select expression layers for integration. `TRAINING_GENES` can specify the genes used to learn the mapping.

Choose `N_PROGRAMS` for the dataset; approximately 1.1–1.2 times the number of Leiden clusters at resolution 1.0 is an empirical starting point rather than a universal optimum. The notebook suggests `(cluster_count * 1.2)`.

Run the cells from top to bottom:

1. Load and validate the reference and spatial data.
2. Learn shared programs and the soft mapping `M`, with reference cells as rows and spatial locations as columns. Each row sums to one.
3. Fit HVGK to the reference's raw spliced/unspliced counts, using expression-based initialization. The demonstration explicitly disables spatial training regularization.
4. Obtain joint posterior means for RNA velocity and biological transcript abundance.
5. Project these quantities to spatial locations using mapping-mass-normalized weights.
6. Construct an expression neighbor graph, calculate the scVelo velocity graph and spatial embedding, and plot spatial velocity streamlines.

## Outputs

By default, results are written to `code/sv_integration_velocity_output/`:

| Output                              | Description                                                  |
| ----------------------------------- | ------------------------------------------------------------ |
| `sv_integration_mapping.h5ad`       | Cell-to-location mapping and integration results             |
| `sv_velo_model/`                    | Saved HVGK model                                             |
| `reference_rna_velocity.h5ad`       | Reference-level posterior mean velocity and biological expression |
| `spatial_rna_velocity.h5ad`         | Spatially projected expression, velocity, mapping mass, and inferred composition |
| `spatial_rna_velocity.png` / `.pdf` | Spatial velocity stream plot                                 |

The spatial result contains `layers['Ms']`, `layers['Mu']`, `layers['velocity']`, `obsm['X_spatial']`, `obsm['mapped_cell_type_composition']`, and `obs['mapped_cell_type']`. 

