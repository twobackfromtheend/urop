### Basic classes

`qubit_system.qubit_system_classes`:
#### `StaticQubitSystem`
- Plots energy levels.
- `N: int, V: float, geometry: Geometry, Omega: float, Delta: np.ndarray`
- Generate energy level plot with `.plot()`
- Finer control over generated plot can be achieved with `.plot_detuning_energy_levels()` which takes more arguments.
- The hamiltonian can be retrieved (as a `Qobj`) with `.get_hamiltonian(detuning)`
   

 #### `EvolvingQubitSystem`
- Solves systems for time-varying fields, generates plots.
- `N: int, V: float, geometry: BaseGeometry, Omega: Callable[[float], float], Delta: Callable[[float], float], t_list: np.ndarray, ghz_state: BaseGHZState`
- `Omega` and `Delta` have to be functions that return one value when called with `t`. (i.e. $\Omega(t)$ = `Omega(t)`)
- These functions can be generated through `get_hamiltonian_coeff_interpolation(x, y, kind)`. `kind` is usually `linear` or `previous`. 
- `.solve()` populates the `.solve_result` attribute using QuTiP's `sesolve`.
- `.get_hamiltonian()` returns the hamiltonian in a format that `sesolve` accepts: [time-independent part, (Omega terms, Omega function), (Delta terms, Delta function)].
- Generate plot with `.plot()` (`solve()` has to be called before): 
  - This plots the three subplots of 
    1. Omega(t) and Delta(t)
    2. Overlap with GHZ state over time
    3. Basis state populations
  - `.plot()` takes a couple of arguments. `plot_others_as_sum` in particular has to be set to `True` for large N, as plotting would otherwise take prohibitively long as there will be a large number of basis states.

    
### Helper classes

#### `qubit_system.utils.ghz_states`
- `StandardGHZState`: Superposition of eeee... and gggg...
- `AlternatingGHZState`: Superposition of egeg... and gege...
- The state tensor can be retrieved (as a `Qobj`) with `.get_state_tensor(symmetric = True)`

#### `qubit_system.geometry`
- `RegularLattice1D(spacing=1)`
- `RegularLattice2D((ROWS, COLUMNS), spacing=1)`
- `RegularLattice3D((ROWS, COLUMNS, DEPTH), spacing=1)`
   
### `quimb` implementation

`qubit_system.qubit_system_classes` and `qubit_system.utils.ghz_states` have counterparts with `_quimb` appended 
(`qubit_system_classes_quimb` and `ghz_states_quimb`) whose functionality should mirror that of the QuTiP implementation.

The `quimb` implementation:
- Uses sparse matrix to take advantage of the sparse Hamiltonians.
- Caches the various parts of the hamiltonian generated (i.e. the time-independent part, the Omega part, and the Delta part) as these are reused.

`qubit_system_classes_quimb.EvolvingQubitSystem`:
- `Omega` and `Delta` have to be `np.array`s that are 1 shorter than `t_list`, as these are interpolated using "previous"
- The `ghz_state` passed as a parameter has to be from `ghz_states_quimb`.
 


## Demonstrations

1. (Unnumbered) First draft versions of energy-level diagrams with toy values, evolution from protocol from Harvard paper with N = 4.
2. Effect of Dimentionality on Energy Level Crossings: Energy level diagrams for N = 8 for increasing dimensions.
3. Achieving GHZ States in Evolving Systems: Exploration of Manual, RL, GRAPE for N = 2, 4, 6, 8 (Incomplete)
   - `demonstration_optimal_control.ipynb` was used to explore optimal control using QuTiP's GRAPE and CRAB 
4. Investigating T_min: Manual optimisation for N = 4, 8 for decreasing protocol duration t.
   - Incomplete as Manual was only bettered by Krotov which was initialised using the manually-optimised protocol
5. Investigating basis states populations over time
   - See the non-notebook `demonstration_5.py`
6. Investigating solve speeds for varying QuTiP solve options
   - Iterating through `atol`, `rtol`, `order`, `min_step` individually and combined resulted in simulation discrepancies (indicated by calculated end-fidelity differences) before any non-negligible difference in solve time.
7. `quimb` exploration
   - Playing around with `quimb` functions, `entropy_subsys` calculations on various states.
8. `quimb` with real values
   - Solving takes a shorter time compared to the QuTiP solve (when comparing with the same discretised protocols)
9. Evaluating average fidelity achieved over multiple runs for the 2 GHZ state types, and for the 3 dimensions.
