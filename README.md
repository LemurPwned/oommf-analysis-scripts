# oommf-simulations

### REQUIREMENTS 
* python 3 or newer 
* all packages from requirements.txt installed for python

### HOW TO USE 
* download all the files from the directory, including those ending with .json
* run **python3** Fourier.py -h to view help menu
* minimal parameters required are: directory, so please run at least **python3** Fourier.py -d "DIR"

### WHAT ARE .JSON FILES?
Both json files are necessary for operation of the script. 

"interface.json" specifies the interface for the Fourier module

"default_param_set.json" is used as the default parameter set. It helps to avoid constant specification of 
the repeating parameters through multiple analysis runs. 

For example you would like to have the parameters specified in the file except for time step, then it's sufficient to use:

**python3** Fourier.py -d DIR -ts 1e-11 - that should overwrite the parameter from default parameter set
