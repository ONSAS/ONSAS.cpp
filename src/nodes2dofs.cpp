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

using namespace arma ;

// =====================================================================
//  nodes2dofs
// =====================================================================
ivec nodes2dofs( ivec nodes, int degreesPerNode ){

  int  n    = nodes.n_elem ;
  ivec dofs = zeros<ivec>( degreesPerNode * n ) ;
  
  for ( int i=0; i<n ; i++){
    for ( int j=0; j< degreesPerNode; j++){
      dofs( i * degreesPerNode + j ) = degreesPerNode*( nodes(i) - 1 ) + ( j+1 ) ;
    }
  }  
  return dofs;
}
// =====================================================================