import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def simulateOneTraj(H, Ls, initTime, finalTime, initPsi, hbar=1, dt=1e-3):
    '''
    Simulates one tracjectory in QJMC, for the given Hamiltonian,
    Jump opreators and intial state vector.

    Parameters
        ----------
        H     : numpy.array
            The matrix representing the system Hamiltonian operator
        Ls    : List(numpy.array)
            An array of matrices representing the Jump operators
        initTime    : float
            initial time
        finalTime   : float
            final time of the simulation
        initPsi     : numpy.array
            initial state vector
        hbar        : float
            The reduced Planck's constant. Set to 1 by default
        dt   : float
            Desired time step

    Returns
        ----------
        times : numpy.array
            numpy array of the calculated times
        sol   : numpy.array
            nested numpy array of the calculated density matrix. It has the shape (d**2, n),
            where d is the dimension of the Hilbert space of the system,
            n is the number of time steps.
    '''

    assert H.shape[0] == len(initPsi), "Dimensions of H, rho and L must match"
    assert np.array_equal(H, H.T.conj()), "H must be Hermitian"

    H_eff = H - 0.5j * np.sum(np.matmul(np.transpose(np.conj(Ls), axes=[0, 2, 1]), Ls), axis=0)

    nSteps = int((finalTime - initTime)/dt)
    times = np.linspace(initTime, finalTime, nSteps+1)

    solut = [np.outer(initPsi, initPsi.conj()).flatten()]
    psiPrev = initPsi

    for time in times[:-1]:
        # dPs = np.abs(np.array(dt * np.real([np.trace(L @ solut[-1].reshape((dim, dim)) @ L.conj().T) for L in Ls]))) 
        dPs = dt * np.real(np.trace(np.matmul(np.matmul(np.outer(psiPrev.conj(), psiPrev), np.transpose(np.conj(Ls), axes=[0, 2, 1])), Ls), axis1=1, axis2=2))
        r1 = np.random.uniform()
        jump_prob = np.sum(dPs)
        psi = None

        # a jump takes place
        if jump_prob > r1: 
            Q = dPs / jump_prob
            k = np.random.choice(np.arange(len(Ls)), size=1, p=Q)[0]  # sample a jump operator
            L = Ls[k]
            psi = L.dot(psiPrev)  # propagate the state using the jump operator

        # no jump
        else:
            # propagate the state using effective hamiltonian, with rk4 steps
            k1 = (-1j * H_eff).dot(psiPrev)
            k2 = (-1j * H_eff).dot(psiPrev + dt*k1/2)
            k3 = (-1j * H_eff).dot(psiPrev + dt*k2/2)
            k4 = (-1j * H_eff).dot(psiPrev + dt*k2/2)

            psi = psiPrev + dt * (k1 + 2*k2 + 2*k3 + k4)/6 

        # normalize psi
        psi = psi / np.linalg.norm(psi)
        psiPrev = psi

        # calculate density matrix and append to solution
        rho = np.outer(psi, psi.conj()).flatten()
        solut.append(rho)    

    return times, np.array(solut).T


def qjmc(H, Ls, initTime, finalTime, initPsi, hbar=1, dt=1e-3, n_traj=500, use_multicore=True, n_cores=1):
    '''
    Run QJMC for n_traj trajectories, for the given Hamiltonian,
    Jump opreators and intial state vector. 

    Parameters
        ----------
        H     : numpy.array
            The matrix representing the system Hamiltonian operator
        Ls    : List(numpy.array)
            An array of matrices representing the Jump operators
        initTime    : float
            initial time
        finalTime   : float
            final time of the simulation
        initPsi     : numpy.array
            initial state vector in Energy eigenbasis
        hbar        : float
            The reduced Planck's constant. Set to 1 by default
        dt   : float
            Desired time step
        n_traj  :int
            Number of trajectories to simulate
        use_multicore: bool
            Use multiprocesses or not for the simulation
        n_cores :int
            Number of cores to use. ignored if use_multicore=False
    Returns
        ----------
        ts : numpy.array
            numpy array of the calculated times
        sol: numpy.array
            Ensemble average of the n_traj trajectories. 
            nested numpy array of the ensemble-averaged density matrix. It has the shape (d**2, n),
            where d is the dimension of the Hilbert space of the system,
            n is the number of time steps.
        variance: numpy.array
            the variance of the ensemble averaged trajectory.
    '''
    num_core_avail = multiprocessing.cpu_count()
    assert n_cores <= num_core_avail, "insufficient cores."
    
    # to keep track of the culumative sum of the trajectories and trajectory ^ 2
    traj_cumulative = np.zeros((len(initPsi) ** 2, 1+int((finalTime - initTime)/dt)), dtype=complex)
    traj_square_cumulative = np.zeros((len(initPsi) ** 2, 1+int((finalTime - initTime)/dt)), dtype=complex)

    if use_multicore:
        # use multiple processes to simulate trajectories
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = [executor.submit(simulateOneTraj, H, Ls, initTime, finalTime, initPsi, dt=dt) for _ in range(n_traj)]
            for future in futures:
                ts, sol = future.result()
                traj_cumulative = np.add(traj_cumulative, sol)
                traj_square_cumulative = np.add(traj_square_cumulative, sol**2)
    else:
        for n in range(n_traj):
            ts, sol = simulateOneTraj(H, Ls, initTime, finalTime, initPsi, dt=dt)
            traj_cumulative = np.add(traj_cumulative, sol)
            traj_square_cumulative = np.add(traj_square_cumulative, sol**2)


    traj_avg = traj_cumulative / n_traj     # compute ensemble average of the trajectories
    traj_avg_sqr = traj_square_cumulative / n_traj      # compute ensemble average^2 of the trajectories
    variance = np.sqrt((traj_avg_sqr - np.abs(traj_avg)**2)/ (n_traj - (n_traj!=1)))    # compute variance
    return ts, traj_avg, variance
