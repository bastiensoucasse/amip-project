# AMIP Project

**Members** Iantsa Provost, Lilian Rebiere, Bastien Soucasse, and Alexey Zhukov.

**Paper** [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155).

**Remotes** [GitHub](https://github.com/bastiensoucasse/amip-minimal), [GitLab (CREMI, Université de Bordeaux)](https://gitlab.emi.u-bordeaux.fr/bsoucasse/amip-project)

## Report

The report can be found in the `report` subfolder, containing the latest compiled PDF and the sources.

- `references.bib` is the references source file.
- `report.pdf` is tha latest compiled report.
- `report.tex` is the LaTeX main source file.

The `images` subfolder contains the images used in the `report.tex` source file.

To compile from the sources, you need a LaTeX compiler such as [TeXLive](https://www.tug.org/texlive).

## Implementation

The actual implementation sources are in the `src` subfolder.

- `datasets.py` is the custom dataset implementation file.
- `environment.py` is the global parameters and variables file.
- `models.py` is the Image Transformer Network (and its blocks) and the Loss Network implementation file.
- `sr_dataset.ipynb` is the custom dataset experiments file.
- `sr.ipynb` is the model experiments file.
- `test.py` is the testing script implementation file.
- `train.py` is the training script implementation file.
- `utils.py` is an utilitary file.

The `models` subfolder contains the saved models after training (used for testing).

[PyTorch](https://pytorch.org) was used for the implementation.
