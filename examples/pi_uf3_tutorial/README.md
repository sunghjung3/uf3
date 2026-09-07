# PI-UF3 tutorial

`pi_uf3_hea5_tutorial.ipynb` trains and evaluates pseudo-interaction UF3 (PI-UF3) models on the
five-element refractory alloy dataset of Byggmästar, Nordlund and Djurabekova, from download to an
exported UF3 model. It is the notebook distributed with the PI-UF3 paper.

Create an environment with Jupyter, then open the notebook; its first cells install this fork:

```bash
mamba create -n pi_uf3 -c conda-forge python=3.11 numpy=1.26 scipy pandas pytables numba ase matplotlib jupyter pytest git pip
conda activate pi_uf3
jupyter notebook pi_uf3_hea5_tutorial.ipynb
```

To run it without the browser:

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 pi_uf3_hea5_tutorial.ipynb
```

Part A (two-body terms, full dataset) takes about 12 minutes on 8 cores and Part B (two- and
three-body terms on a subset) about 35 minutes. Everything the notebook writes goes to `pi_uf3_work/` next to it.
