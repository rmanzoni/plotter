# plotter

tested in the following environment, set up with `conda`

```
conda create -n alt_root python=3.7 root -c conda-forge
conda activate alt_root
conda config --env --add channels conda-forge
conda install -n alt_root root_numpy
conda install -n alt_root tensorflow -c conda-forge
conda install -n alt_root keras -c conda-forge
conda install -n alt_root matplotlib
conda install -n alt_root scikit-learn
python -m pip install rootpy --user
python -m pip install modin --user
```

and `conda info` returns


```

     active environment : alt_root
    active env location : /Users/manzoni/opt/anaconda2/envs/alt_root
            shell level : 2
       user config file : /Users/manzoni/.condarc
 populated config files : /Users/manzoni/.condarc
                          /Users/manzoni/opt/anaconda2/envs/alt_root/.condarc
          conda version : 4.7.12
    conda-build version : 3.18.11
         python version : 2.7.15.final.0
       virtual packages :
       base environment : /Users/manzoni/opt/anaconda2  (writable)
           channel URLs : https://conda.anaconda.org/conda-forge/osx-64
                          https://conda.anaconda.org/conda-forge/noarch
                          https://repo.anaconda.com/pkgs/main/osx-64
                          https://repo.anaconda.com/pkgs/main/noarch
                          https://repo.anaconda.com/pkgs/r/osx-64
                          https://repo.anaconda.com/pkgs/r/noarch
                          https://conda.anaconda.org/nlesc/osx-64
                          https://conda.anaconda.org/nlesc/noarch
          package cache : /Users/manzoni/opt/anaconda2/pkgs
                          /Users/manzoni/.conda/pkgs
       envs directories : /Users/manzoni/opt/anaconda2/envs
                          /Users/manzoni/.conda/envs
               platform : osx-64
             user-agent : conda/4.7.12 requests/2.22.0 CPython/2.7.15 Darwin/19.0.0 OSX/10.15.1
                UID:GID : 503:20
             netrc file : None
           offline mode : False
```

list of packages

```
# packages in environment at /Users/manzoni/opt/anaconda2/envs/alt_root:
#
# Name                    Version                   Build  Channel
_tflow_select             2.3.0                       mkl
absl-py                   0.8.1                    py37_0    conda-forge
afterimage                1.21              h044d061_1002    conda-forge
appnope                   0.1.0                 py37_1000    conda-forge
astor                     0.7.1                      py_0    conda-forge
attrs                     19.3.0                     py_0    conda-forge
awkward                   0.12.14                    py_0    conda-forge
backcall                  0.1.0                      py_0    conda-forge
binutils                  1.0.1                         0    conda-forge
bleach                    3.1.0                      py_0    conda-forge
bzip2                     1.0.8                h01d97ff_1    conda-forge
c-ares                    1.15.0            h01d97ff_1001    conda-forge
c-compiler                1.0.1                h1de35cc_0    conda-forge
ca-certificates           2019.9.11            hecc5488_0    conda-forge
cachetools                3.1.1                      py_0    conda-forge
cairo                     1.16.0            he1c11cd_1002    conda-forge
cctools                   895                           1
certifi                   2019.9.11                py37_0    conda-forge
cffi                      1.13.2           py37h33e799b_0    conda-forge
cfitsio                   3.470                h389770f_2    conda-forge
chardet                   3.0.4                 py37_1003    conda-forge
clang                     4.0.1                         1
clang_osx-64              4.0.1               h1ce6c1d_17    conda-forge
clangxx                   4.0.1                         1
clangxx_osx-64            4.0.1               h22b1bf0_17    conda-forge
compiler-rt               4.0.1                hcfea43d_1
compilers                 1.0.1                         0    conda-forge
cryptography              2.8              py37hafa8578_0    conda-forge
curl                      7.65.3               h22ea746_0    conda-forge
cxx-compiler              1.0.1                h04f5b5a_0    conda-forge
cycler                    0.10.0                     py_2    conda-forge
davix                     0.7.5                h7232a33_0    conda-forge
decorator                 4.4.1                      py_0    conda-forge
defusedxml                0.6.0                      py_0    conda-forge
entrypoints               0.3                   py37_1000    conda-forge
fftw                      3.3.8           mpi_mpich_h6e18f22_1009    conda-forge
fontconfig                2.13.1            h6b1039f_1001    conda-forge
fortran-compiler          1.0.1                h4f947d3_0    conda-forge
freetype                  2.10.0               h24853df_1    conda-forge
fribidi                   1.0.5             h01d97ff_1002    conda-forge
gast                      0.2.2                      py_0    conda-forge
gdk-pixbuf                2.36.12           h284f8de_1003    conda-forge
gettext                   0.19.8.1          h46ab8bc_1002    conda-forge
gfortran_osx-64           4.8.5                h22b1bf0_8    conda-forge
giflib                    5.1.7                h01d97ff_1    conda-forge
glew                      2.0.0             h0a44026_1002    conda-forge
glib                      2.58.3            h9d45998_1002    conda-forge
gobject-introspection     1.58.2          py37h93883a9_1002    conda-forge
google-pasta              0.1.8                      py_0    conda-forge
graphite2                 1.3.13            h2098e52_1000    conda-forge
graphviz                  2.40.1               h69955ae_1    conda-forge
grpcio                    1.23.0           py37h6ef0057_0    conda-forge
gsl                       2.5                  ha2d443c_1    conda-forge
h5py                      2.10.0          nompi_py37h106b333_100    conda-forge
harfbuzz                  2.4.0                hd8d2a14_3    conda-forge
hdf5                      1.10.5          nompi_h0cbb7df_1103    conda-forge
icu                       64.2                 h6de7cb9_1    conda-forge
idna                      2.8                   py37_1000    conda-forge
importlib_metadata        0.23                     py37_0    conda-forge
ipykernel                 5.1.3            py37h5ca1d4c_0    conda-forge
ipyparallel               6.2.4                    py37_0    conda-forge
ipython                   7.9.0            py37h5ca1d4c_0    conda-forge
ipython_genutils          0.2.0                      py_1    conda-forge
jedi                      0.15.1                   py37_0    conda-forge
jinja2                    2.10.3                     py_0    conda-forge
joblib                    0.14.0                     py_0    conda-forge
jpeg                      9c                h1de35cc_1001    conda-forge
jsonschema                3.1.1                    py37_0    conda-forge
jupyter_client            5.3.3                    py37_1    conda-forge
jupyter_core              4.5.0                      py_0    conda-forge
keras                     2.3.1                    py37_0    conda-forge
keras-applications        1.0.8                      py_1    conda-forge
keras-preprocessing       1.1.0                      py_0    conda-forge
kiwisolver                1.1.0            py37h770b8ee_0    conda-forge
krb5                      1.16.3            hcfa6398_1001    conda-forge
ld64                      274.2                         1
libblas                   3.8.0               14_openblas    conda-forge
libcblas                  3.8.0               14_openblas    conda-forge
libcroco                  0.6.13               hc484408_0    conda-forge
libcurl                   7.65.3               h16faf7d_0    conda-forge
libcxx                    4.0.1                hcfea43d_1    conda-forge
libcxxabi                 4.0.1                hcfea43d_1
libedit                   3.1.20170329      hcfe32e1_1001    conda-forge
libffi                    3.2.1             h6de7cb9_1006    conda-forge
libgfortran               3.0.1                         0    conda-forge
libgpuarray               0.7.6             h1de35cc_1003    conda-forge
libiconv                  1.15              h01d97ff_1005    conda-forge
liblapack                 3.8.0               14_openblas    conda-forge
libopenblas               0.3.7                hd44dcd8_1    conda-forge
libpng                    1.6.37               h2573ce8_0    conda-forge
libprotobuf               3.9.2                hd9629dc_0
librsvg                   2.44.15              h90c2430_0    conda-forge
libsodium                 1.0.17               h01d97ff_0    conda-forge
libssh2                   1.8.2                hcdc9a53_2    conda-forge
libtiff                   4.0.10            hd08fb8f_1003    conda-forge
libxml2                   2.9.10               h53d96d6_0    conda-forge
llvm                      4.0.1                         1
llvm-lto-tapi             4.0.1                         1    conda-forge
llvm-openmp               9.0.0                h40edb58_0    conda-forge
lz4                       2.2.1            py37he1520b0_0    conda-forge
lz4-c                     1.8.3             h6de7cb9_1001    conda-forge
mako                      1.1.0                      py_0    conda-forge
markdown                  3.1.1                      py_0    conda-forge
markupsafe                1.1.1            py37h0b31af3_0    conda-forge
matplotlib                2.2.4                    py37_1    conda-forge
matplotlib-base           2.2.4            py37h31f9439_1    conda-forge
metakernel                0.24.3                     py_0    conda-forge
mistune                   0.8.4           py37h0b31af3_1000    conda-forge
more-itertools            7.2.0                      py_0    conda-forge
mpi                       1.0                       mpich    conda-forge
mpich                     3.2.1             ha90c164_1014    conda-forge
nbconvert                 5.6.1                    py37_0    conda-forge
nbformat                  4.4.0                      py_1    conda-forge
ncurses                   6.1               h0a44026_1002    conda-forge
notebook                  6.0.1                    py37_0    conda-forge
numpy                     1.17.3           py37hde6bac1_0    conda-forge
openssl                   1.1.1d               h0b31af3_0    conda-forge
opt_einsum                3.1.0                      py_0    conda-forge
pandas                    0.25.1           py37h86efe34_0    conda-forge
pandoc                    2.7.3                         0    conda-forge
pandocfilters             1.4.2                      py_1    conda-forge
pango                     1.42.4               h6691c8e_1    conda-forge
parso                     0.5.1                      py_0    conda-forge
pcre                      8.43                 h0a44026_0
pexpect                   4.7.0                    py37_0    conda-forge
pickleshare               0.7.5                 py37_1000    conda-forge
pip                       19.3.1                   py37_0    conda-forge
pixman                    0.38.0            h01d97ff_1003    conda-forge
portalocker               1.5.1                    py37_0    conda-forge
prometheus_client         0.7.1                      py_0    conda-forge
prompt_toolkit            2.0.10                     py_0    conda-forge
protobuf                  3.9.2            py37h0a44026_0
ptyprocess                0.6.0                   py_1001    conda-forge
pycparser                 2.19                     py37_1    conda-forge
pydot                     1.4.1                 py37_1001    conda-forge
pygments                  2.4.2                      py_0    conda-forge
pygpu                     0.7.6           py37h3b54f70_1000    conda-forge
pyopenssl                 19.0.0                   py37_0    conda-forge
pyparsing                 2.4.5                      py_0    conda-forge
pyrsistent                0.15.5           py37h0b31af3_0    conda-forge
pysocks                   1.7.1                    py37_0    conda-forge
pythia8                   8.240            py37h6de7cb9_2    conda-forge
python                    3.7.3                h93065d6_1    conda-forge
python-dateutil           2.8.1                      py_0    conda-forge
python-xxhash             1.4.1            py37h0b31af3_0    conda-forge
pytz                      2019.3                     py_0    conda-forge
pyyaml                    5.1.2            py37h0b31af3_0    conda-forge
pyzmq                     18.1.0           py37hee98d25_0    conda-forge
qt                        5.9.7                h8cf7e54_3    conda-forge
readline                  8.0                  hcfe32e1_0    conda-forge
requests                  2.22.0                   py37_1    conda-forge
root                      6.18.00         py37h500fca7_17    conda-forge
root_numpy                4.8.0            py37haf112f3_2    conda-forge
root_pandas               0.7.0                      py_0    conda-forge
scikit-learn              0.21.3           py37hd4ffd6c_0    conda-forge
scipy                     1.3.1            py37hab3da7d_2    conda-forge
send2trash                1.5.0                      py_0    conda-forge
setuptools                41.6.0                   py37_1    conda-forge
six                       1.13.0                   py37_0    conda-forge
sqlite                    3.30.1               h93121df_0    conda-forge
tbb                       2019.8               h04f5b5a_0
tbb-devel                 2019.8               h04f5b5a_0
tensorboard               2.0.0              pyhb230dea_0
tensorflow                2.0.0           mkl_py37hda344b4_0
tensorflow-base           2.0.0           mkl_py37h66b1bf0_0
tensorflow-estimator      2.0.0              pyh2649769_0
termcolor                 1.1.0                      py_2    conda-forge
terminado                 0.8.2                    py37_0    conda-forge
testpath                  0.4.4                      py_0    conda-forge
theano                    1.0.4           py37h0a44026_1000    conda-forge
tk                        8.6.9             h2573ce8_1003    conda-forge
tornado                   6.0.3            py37h0b31af3_0    conda-forge
traitlets                 4.3.3                    py37_0    conda-forge
uproot                    3.10.10                  py37_0    conda-forge
uproot-base               3.10.10                  py37_0    conda-forge
uproot-methods            0.7.1                      py_0    conda-forge
urllib3                   1.25.7                   py37_0    conda-forge
vdt                       0.4.3                h6de7cb9_0    conda-forge
wcwidth                   0.1.7                      py_1    conda-forge
webencodings              0.5.1                      py_1    conda-forge
werkzeug                  0.16.0                     py_0    conda-forge
wheel                     0.33.6                   py37_0    conda-forge
wrapt                     1.11.2           py37h0b31af3_0    conda-forge
xrootd                    4.9.1            py37h02158b6_1    conda-forge
xz                        5.2.4             h1de35cc_1001    conda-forge
yaml                      0.1.7             h1de35cc_1001    conda-forge
zeromq                    4.3.2                h6de7cb9_2    conda-forge
zipp                      0.6.0                      py_0    conda-forge
zlib                      1.2.11            h0b31af3_1006    conda-forge
zstd                      1.4.0                ha9f0a20_0    conda-forge
```



# Limits

```
combineCards.py disp1=datacard_hnl_m_12_lxy_lt_0p5_hnl_m_10_v2_1p0Em06_majorana.txt disp2=datacard_hnl_m_12_lxy_0p5_to_2p0_hnl_m_10_v2_1p0Em06_majorana.txt disp3=datacard_hnl_m_12_lxy_mt_2p0_hnl_m_10_v2_1p0Em06_majorana.txt > datacard_hnl_m_12_combined_hnl_m_10_v2_1p0Em06_majorana.txt

combine -M AsymptoticLimits datacard_hnl_m_12_combined_hnl_m_10_v2_1p0Em06_majorana.txt --run blind

combine -M HybridNew --testStat=LHC --frequentist -d datacard_hnl_m_12_combined_hnl_m_10_v2_1p0Em06_majorana.txt -T 1000 -C 0.95 --rMin 0 --rMax 50  
```
