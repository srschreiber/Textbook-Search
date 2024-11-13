conda create -n searchproj python=3.10 -y
conda activate searchproj

# Inside the new environment...
conda install -c anaconda wget -y
conda install -c conda-forge openjdk=21 maven -y

# If you want the optional dependencies, otherwise skip
conda install -c conda-forge lightgbm -y
conda install -c anaconda nmslib -y
conda install -c pytorch faiss-cpu -y

# Good idea to always explicitly specify the latest version, found here: https://pypi.org/project/pyserini/
pip install pyserini==0.43.0
# If you want the optional dependencies, otherwise skip; the temperamental packages are already installed at this point
# so should be smooth...
pip install numpy==1.26.4
