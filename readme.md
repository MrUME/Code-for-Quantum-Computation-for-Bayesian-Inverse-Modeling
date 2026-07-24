## Code

This repository contains the implementation, details, data and code of the quantum Bayesian inversion framework.

### Introduction

1. **`main.py`**  
   Main program for the quantum Bayesian inversion framework. It integrates the complete workflow, including:
   - Coarse bound refinement
   - QMSA-based MAP search
   - QAE-based posterior uncertainty quantification
   * results can be found on "results" folder

2. **`sub_acceler.py`**  
   Acceleration demonstration for QMSA and QAE algorithms.
   * results can be found on "results_acceler" folder

3. **`sub_qcircuit.py`**  
   This code is used to generate quantum circuits for QMSA search and QAE-based evidence or moment estimation.
   * results can be found on "results_quantum" folder
   
4. **`sub_qresult_interpret.py`**  
   This code converts and post-processes measurement results obtained from  noisy quantum computers.
   * results can be found on "results_quantum" folder

5. **`sub_plot.py`**  
   Code for generating Figure 3.
   * figure can be found on "results" folder

6. **`other code`**  
   * `utils.py` contains utility functions for main.py.
   * `Forwardmodel/UQ.py` performs the forward simulations used for the uncertainty intervals in panel D of Fig. 3.
   * Folder: `Additional code for gottingen` contains the original data and code for generating Figure 2.

   
   
   