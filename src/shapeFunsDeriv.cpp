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

#include <iostream>      /* printf */
#include <armadillo>

using namespace arma ;


// ==============================================================================
mat shapeFunsDeriv ( double x, double y, double z ){

  mat fun = zeros<mat>( 3, 4 ) ;
  fun( 0, 0 ) =  1 ;
  fun( 0, 1 ) = -1 ;
  fun( 1, 1 ) = -1 ;
  fun( 2, 1 ) = -1 ;
  fun( 2, 2 ) =  1 ;
  fun( 1, 3 ) =  1 ;

  return fun;
}

