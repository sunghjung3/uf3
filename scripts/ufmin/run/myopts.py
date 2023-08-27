import warnings

import numpy as np
from numpy.linalg import eigh

from ase.optimize.optimize import Optimizer


class GD(Optimizer):
    # default parameters
    defaults = {**Optimizer.defaults, 'lr': 0.05}

    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 maxstep=None, master=None, lr=None):
        """Gradient descent optimizer (hand-written by sung)

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.2 Å).

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        lr: float
            Learning rate of gradient descent (default value is 0.05 Å^2/eV).
        """
        if maxstep is None:
            self.maxstep = self.defaults['maxstep']
        else:
            self.maxstep = maxstep

        if self.maxstep > 1.0:
            warnings.warn('You are using a *very* large value for '
                          'the maximum step size: %.1f Å' % maxstep)

        if lr is None:
            self.lr = self.defaults['lr']
        else:
            self.lr = lr

        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master)

    def todict(self):
        d = Optimizer.todict(self)
        if hasattr(self, 'maxstep'):
            d.update(maxstep=self.maxstep)
        return d

    def initialize(self):
        self.r0 = None
        self.f0 = None

    def read(self):
        self.r0, self.f0, self.maxstep = self.load()

    def step(self, f=None):
        atoms = self.atoms

        if f is None:
            f = atoms.get_forces()

        r = atoms.get_positions()
        dr = self.lr * f
        steplengths = (dr**2).sum(1)**0.5
        dr = self.determine_step(dr, steplengths)
        atoms.set_positions(r + dr)
        self.r0 = r.copy()
        self.f0 = f.copy()
        self.dump((self.r0, self.f0, self.maxstep))

    def determine_step(self, dr, steplengths):
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the eigendirection.
        """
        maxsteplength = np.max(steplengths)
        if maxsteplength >= self.maxstep:
            scale = self.maxstep / maxsteplength
            dr *= scale

        return dr


class MGD(Optimizer):
    # default parameters
    defaults = {**Optimizer.defaults, 'lr': 0.05, 'rho': 0.25}

    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 maxstep=None, master=None, lr=None, rho=None):
        """Gradient descent optimizer with momentum (hand-written by sung)

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.2 Å).

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        lr: float
            Learning rate of gradient descent (default value is 0.05 Å^2/eV).

        rho: float
            Momentum hyperparameter for this optimizer (default value is 0.25).
        """
        if maxstep is None:
            self.maxstep = self.defaults['maxstep']
        else:
            self.maxstep = maxstep

        if self.maxstep > 1.0:
            warnings.warn('You are using a *very* large value for '
                          'the maximum step size: %.1f Å' % maxstep)

        if lr is None:
            self.lr = self.defaults['lr']
        else:
            self.lr = lr
        if rho is None:
            self.rho = self.defaults['rho']
        else:
            self.rho = rho

        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master)

    def todict(self):
        d = Optimizer.todict(self)
        if hasattr(self, 'maxstep'):
            d.update(maxstep=self.maxstep)
        return d

    def initialize(self):
        self.dr_1 = 0.0  # dr from the previous iteration
        self.r0 = None
        self.f0 = None

    def read(self):
        self.dr_1, self.r0, self.f0, self.maxstep = self.load()

    def step(self, f=None):
        atoms = self.atoms

        if f is None:
            f = atoms.get_forces()

        r = atoms.get_positions()
        dr = self.lr * f + self.rho * self.dr_1
        steplengths = (dr**2).sum(1)**0.5
        dr = self.determine_step(dr, steplengths)
        atoms.set_positions(r + dr)
        self.r0 = r.copy()
        self.f0 = f.copy()
        self.dr_1 = dr.copy()
        self.dump((self.dr_1, self.r0, self.f0, self.maxstep))

    def determine_step(self, dr, steplengths):
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the eigendirection.
        """
        maxsteplength = np.max(steplengths)
        if maxsteplength >= self.maxstep:
            scale = self.maxstep / maxsteplength
            # FUTURE: Log this properly
            # msg = '\n** scale step by {:.3f} to be shorter than {}'.format(
            #     scale, self.maxstep
            # )
            # print(msg, flush=True)

            dr *= scale

        return dr
