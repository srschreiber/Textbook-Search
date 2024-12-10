
# Setup Guide


## Setting Up Your Conda Environment
```
conda create -n e-book-search-dev python=3.10
```

Note: If you need to remove your conda environment, you can do so by running the following command:
```
conda env remove -n e-book-search-dev
```

## Activating Your Conda Environment
```
conda activate e-book-search-dev
```

## Setting Up Pyserini
To set up Pyserini, follow the instructions [here](https://github.com/castorini/pyserini/blob/master/docs/installation.md#pypi-installation-walkthrough) using e-book-search as the conda environment name.

```
conda install -c anaconda wget -y
conda install -c conda-forge openjdk=21 maven -y
conda install -c conda-forge lightgbm -y
conda install -c anaconda nmslib -y
conda install -c pytorch faiss-cpu -y
pip install pyserini==0.43.0
pip install tqdm
# spacy and nltk are required for the indexing process
conda install -c conda-forge spacy -y
conda install -c conda-forge nltk -y
# downgrade numpy to 1.23.5
pip install numpy==1.23.5
```

## Running the Program
Run app.py to start the program.

You can change the variabler `mode` to cranfield or default to run the program in different modes.

cranfield mode will evaluate the model on the Cranfield dataset.

default mode will run the program against the sample source document, data/text.txt