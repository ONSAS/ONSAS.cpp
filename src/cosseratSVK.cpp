// Copyright (C) 2020, J. M. Perez Zerpa
//
// This file is part of ONSAS++.
//
// ONSAS++ is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// ONSAS is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ONSAS++.  If not, see <https://www.gnu.org/licenses/>.

#include <armadillo>

using namespace arma;

mat cosseratSVK ( vec hyperElasParamsVec, mat Egreen){

  double young = hyperElasParamsVec(0)       ;
  double nu    = hyperElasParamsVec(1)       ;

  double lambda  = young * nu / ( (1.0 + nu) * (1.0 - 2.0*nu) ) ;
  double shear   = young      / ( 2.0 * (1.0 + nu) )          ;
  
  return lambda * trace( Egreen ) * eye(3,3)  +  2.0 * shear * Egreen ;
}
// ==============================================================================

