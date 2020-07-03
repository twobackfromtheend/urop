This folder contains the scripts used to generate the plots used in the paper.

`protocols.py` served as a record of the optimised protocols, and I used this file to save protocol details into a form
ingestable by other scripts (through `optimised_protocols/save.py`'s `save_protocol()`).

`values_extrema.py` was used to identify the extrema in the protocol's Omega and Delta (to calculate ramp times).
 

The `plots_creation_#.py` files generate plots. They generally took in the protocol saved through 
`optimised_protocols/saver.py`, evolved the system, and then plotted the required data from the evolved system.
However, certain statistics involving the instantaneous eigenstates took a long time to calculate (solving the full
Hamiltonian to calculate these dressed states). To avoid recalculating these whenever needed, I implemented some 
functions within `plots_creation_3.py` to save these states (for 100 timesteps out of the total 3000 timesteps - to 
generate reasonably smooth curves while saving computational time). 

Details for each of the plots_creation files are as follows:

1. Protocol & Fidelity plots
   1. Fidelity with EEE and GGG  (with zoomed plots near t=T)
   2. Fidelity with target state (i.e. EGEG and GEGE for ALT GHZ)
   3. With diagonal and off-diagonal terms. Single figure with all 6 plots.
   4. With diagonal and off-diagonal terms. One figure per plot, for a total of 6 individual figures.
2. Energy spectrums (Full and zoomed in)
3. Time-dependent eigenenergies
   - Also contains `_save_time_dependent_eigenstates` and `_load_time_dependent_energies` functions which save the 
   evolved system at different timesteps (as implemented here, saves 1 of every 30 timesteps for a total of 100 timesteps).
4. N_pc and populations plots
   1. Plots Npc and eigenstate populations plot (in a single 2-subplots figure for each protocol)
   2. Unused. I was wanting to solely plot the eigenstate populations plot (the coloured rectangle with each row 
   representing an eigenstate), but did not complete this.
5. Magnetisation
6. Kinks
7. Fermionic energy
8. Eigenstate stats: N_pc and Entropy
9. Magnetisation, Entropy, and N_pc in a single uniform layout and format. 9_b contains the 
entanglement entropy ramp plots
10. Entanglement entropy plots on logged axes (used in paper)


     
