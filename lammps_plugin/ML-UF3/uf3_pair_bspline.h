/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.
   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// De Boor's algorithm @
// https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/de-Boor.html
// For values outside the domain, it exhibits undefined behavior.
// Uses fixed B-Spline degree 3.

#include "pointers.h"

#include "uf3_bspline_basis2.h"
#include "uf3_bspline_basis3.h"

#include <vector>

#ifndef UF3_PAIR_BSPLINE_H
#define UF3_PAIR_BSPLINE_H

namespace LAMMPS_NS {

class uf3_pair_bspline {
 private:
  int knot_vect_size, coeff_vect_size;
  std::vector<double> knot_vect, dnknot_vect;
  std::vector<double> coeff_vect, dncoeff_vect;
  std::vector<uf3_bspline_basis3> bspline_bases;
  std::vector<uf3_bspline_basis2> dnbspline_bases;
  LAMMPS *lmp;

 public:
  // dummy constructor
  uf3_pair_bspline();
  uf3_pair_bspline(LAMMPS *ulmp, const std::vector<double> &uknot_vect,
                   const std::vector<double> &ucoeff_vect);
  ~uf3_pair_bspline();
  double ret_val[2];
  double *eval(double value_rij);

  double memory_usage();
};
}    // namespace LAMMPS_NS
#endif
