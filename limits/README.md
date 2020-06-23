# Limits Analysis Tool

## Installation
```
cmsrel CMSSW_10_2_13
cd CMSSW_10_2_13/src
cmsenv
```
```
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
source env_standalone.sh 
make -j 8; make 
```

```
git clone git@github.com:rmanzoni/plotter.git WHNL_code
cd plotter/limits
```

## Running the limits
The production and plotting of the limits are done via the LimitsLauncher.py script.

This script  monitors the  
  1. combination the datacards
  2. production of the limits
  3. the plotting of the results

via three scripts: combine_datacards.py, produce_limits.py and limits_plotter.py

For the second task, there is the possibility to run the production in parallel, by submitting jobs on the batch, via submitter.sh. 
In that case, wait that all the jobs are finished before running the plotting tool.

Usage:
```
  1. update the user's decision board
  2. python LimitsLauncher.py
```


