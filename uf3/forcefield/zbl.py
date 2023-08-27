import numpy as np

import ase
import ase.data as ase_data


class SwitchingFunction:
    """
    Switching function for the ZBL potential.
    
    Reference: https://docs.lammps.org/pair_gromacs.html
    """
    def __init__(self, r1, rc,
                 E_rc, dE_rc, d2E_rc,
                 *args, **kwargs):
        self.r1 = r1
        self.rc = rc

        self.A = ( -3*dE_rc + (rc-r1)*d2E_rc ) / (rc-r1)**2
        self.B = ( 2*dE_rc - (rc-r1)*d2E_rc ) / (rc-r1)**3
        self.C = -1*E_rc + 1/2*dE_rc*(rc-r1) - 1/12*d2E_rc*(rc-r1)**2

    def __call__(self, r):
        return (r < self.r1) * self.C + \
            (r >= self.r1) * (r < self.rc) * (self.A/3*(r-self.r1)**3 + self.B/4*(r-self.r1)**4 + self.C) + \
            (r >= self.rc) * 0.0

    def d(self, r):
        return (r < self.r1) * 0.0 + \
            (r >= self.r1) * (r < self.rc) * (self.A*(r-self.r1)**2 + self.B*(r-self.r1)**3) + \
            (r >= self.rc) * 0.0

    def d2(self, r):
        return (r < self.r1) * 0.0 + \
            (r >= self.r1) * (r < self.rc) * (2*self.A*(r-self.r1) + 3*self.B*(r-self.r1)**2) + \
            (r >= self.rc) * 0.0


class ZBL:
    """
    LAMMPS-style Ziegler-Biersack-Littmark potential.

    Reference: https://docs.lammps.org/pair_zbl.html
    """
    def __init__(self, z1, z2,):
        self.z1 = z1
        self.z2 = z2
        c = 299792458  # speed of light (m/s)
        e = 1.602176634e-19  # elementary charge (C)

        # z1 * z2 * e^2/(4*pi*epsilon_0) (eV C^2 /Å)
        self.prefactor = 1000 * e * c**2 * self.z1 * self.z2 

        # 1/a
        self.a_inv = (self.z1**0.23 + self.z2**0.23) / 0.46850

        # constants for phi(x)
        self.a1 = 0.18175
        self.b1 = -3.19980
        self.a2 = 0.50986
        self.b2 = -0.94229
        self.a3 = 0.28022
        self.b3 = -0.40290
        self.a4 = 0.02817
        self.b4 = -0.20162

    def phi(self, x):
        return self.a1 * np.exp(self.b1 * x) + \
            self.a2 * np.exp(self.b2 * x) + \
            self.a3 * np.exp(self.b3 * x) + \
            self.a4 * np.exp(self.b4 * x)
    
    def dphi(self, x):
        return self.a1 * self.b1 * np.exp(self.b1 * x) + \
            self.a2 * self.b2 * np.exp(self.b2 * x) + \
            self.a3 * self.b3 * np.exp(self.b3 * x) + \
            self.a4 * self.b4 * np.exp(self.b4 * x)
    
    def d2phi(self, x):
        return self.a1 * self.b1**2 * np.exp(self.b1 * x) + \
            self.a2 * self.b2**2 * np.exp(self.b2 * x) + \
            self.a3 * self.b3**2 * np.exp(self.b3 * x) + \
            self.a4 * self.b4**2 * np.exp(self.b4 * x)

    def __call__(self, r):
        return self.prefactor / r * self.phi(self.a_inv * r)
    
    def d(self, r):
        return self.prefactor * (
            -1 * self.phi(self.a_inv * r) / r**2 + \
            self.a_inv * self.dphi(self.a_inv * r) / r
            )

    def d2(self, r):
        return self.prefactor * (
            2 * self.phi(self.a_inv * r) / r**3 - \
            2 * self.a_inv * self.dphi(self.a_inv * r) / r**2 + \
            self.a_inv**2 * self.d2phi(self.a_inv * r) / r
            )
    

class SwitchingZBL:
    def __init__(self, z1, z2, r1, rc, scale=1.0):
        self.r1 = r1
        self.rc = rc
        self.zbl = ZBL(z1, z2)
        self.switch = SwitchingFunction(r1, rc,
                                        self.zbl(rc),
                                        self.zbl.d(rc),
                                        self.zbl.d2(rc))
        self.scale = scale
        
    def __call__(self, r):
        return self.scale * ( (r < self.rc) * self.zbl(r) + self.switch(r) )
    
    def d(self, r):
        return self.scale * ( (r < self.rc) * self.zbl.d(r) + self.switch.d(r) )

    def d2(self, r):
        return self.scale * ( (r < self.rc) * self.zbl.d2(r) + self.switch.d2(r) )


class LJSwitchingZBL(SwitchingZBL):
    def __init__(self, z1, z2, scale=1.0):
        approximate_bond_length = \
            ase_data.covalent_radii[z1] + ase_data.covalent_radii[z2]
        sigma = approximate_bond_length * 2**(-1/6)  # sigma from LJ
        r1 = sigma
        rc = approximate_bond_length
        super().__init__(z1, z2, r1, rc, scale=scale)


if __name__ == "__main__":
    # test ZBL
    z1 = 78
    z2 = 78
    r1 = 1.9
    rc = 2
    zbl = SwitchingZBL(z1, z2, r1, rc)

    # plot
    import matplotlib.pyplot as plt
    r = np.linspace(0.1, 3, 1000)
    plt.plot(r, zbl(r), label="V")
    #plt.plot(r, zbl.d(r), label="dV/dr")
    #plt.plot(r, zbl.d2(r), label="d2V/dr2")
    plt.ylim([-0.25, 20])
    plt.xlim([1.8, 2.0])
    plt.legend()
    plt.show()