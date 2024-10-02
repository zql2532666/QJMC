# QJMC
A toy implementation of the Quantum Jump Method in python.

# Algorithm
The Quantum Jump Method simulates quantum dissipation by propagating the initial state vector $|\psi(0)\rangle$ instead of density matrix. The algorithm works as follows for each time step $dt$ in a trajectory,

1. For each time step $dt$, calculate the probability $dP$ of a jump happening. Where $dP = dt\sum_{i}\langle \psi(t)|L_i^TL_i | \psi(t)\rangle$

2. Generate a random number $r1$ uniformly. Compare this random number with the jump probability. 

3. if $dP>r1$, then
    - choose a jump operator $L_i$, according to the probability $p_i = dt\frac{\langle \psi(t)|L_i^TL_i | \psi(t)\rangle}{dP} $
    - Evolve the state vector using the jump operator - $|\psi(t+dt)\rangle = L_i |\psi(t)\rangle$
4. else:
    - Propagate the state vector according to the non-unitary Schrodinger Equation, 
        $$i\hbar\frac{d}{d t}|\psi\rangle = H_{eff} |\psi\rangle $$
      Where $H_{eff} = H - \frac{i\hbar}{2}\sum_{i}L_i^TL_i$.
      
5. Normalize the new state vector $|\psi(t+dt)\rangle$ since the above steps are non-unitary. The norm of the state vector was not preserved.

The ensemble average of the N trajectories simulated would be taken as the solution. 

# Reference
- A Wave Function approach to dissipative processes: https://arxiv.org/abs/0805.4002