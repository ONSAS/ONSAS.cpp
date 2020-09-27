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

// ------------------------------------------------
// To compile run "make" or from command line:
// g++ timeStepIteration.cpp -larmadillo -o timeStepIteration.lnx
// ------------------------------------------------

#include <iostream>      /* printf */
#include <armadillo>
//~ #include <cstdlib>   // funciones como atoi y atof
//~ #include <ctime>
//~ #include <stdio.h>      /* printf */
//~ #include <math.h>      /* printf */
//~ #include <time.h>       /* time_t, struct tm, difftime, time, mktime */
 //~ #include <fstream>   // file streaming

using namespace std  ;
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




// ==============================================================================
mat shapeFunsDeriv ( double x, double y, double z ){

  mat fun = zeros<mat>( 3, 4 ) ;
  fun(0,0) =  1 ;
  fun(0,1) = -1 ;
  fun(1,1) = -1 ;
  fun(2,1) = -1 ;
  fun(2,2) =  1 ;
  fun(1,3) =  1 ;
  return fun;
}
// ==============================================================================




// ======================================================================
//
// ======================================================================
mat cosseratSVK ( vec hyperElasParamsVec, mat Egreen){

  double young = hyperElasParamsVec(0)       ;
  double nu    = hyperElasParamsVec(1)       ;

  double lambda  = young * nu / ( (1.0 + nu) * (1.0 - 2.0*nu) ) ;
  double shear   = young      / ( 2.0 * (1.0 + nu) )          ;
  
  return lambda * trace( Egreen ) * eye(3,3)  +  2.0 * shear * Egreen ;
}
// ==============================================================================



// ======================================================================
//
// ======================================================================
mat BgrandeMats ( mat deriv , mat F ){

  mat matBgrande = zeros<mat>(6, 12);
  
  for (int k=1; k<=4; k++){

    for (int i=1; i<=3 ; i++){

      for (int j=1; j<=3; j++){
        matBgrande ( i-1 , (k-1)*3 + j -1  ) = deriv(i-1,k-1) * F(j-1,i-1) ;
      }
    }          

    for (int j=1; j<=3; j++){
      matBgrande ( 4-1 , (k-1)*3 + j-1 ) = deriv(2-1,k-1) * F(j-1,3-1) + deriv(3-1,k-1) * F(j-1,2-1) ;
      matBgrande ( 5-1 , (k-1)*3 + j-1 ) = deriv(1-1,k-1) * F(j-1,3-1) + deriv(3-1,k-1) * F(j-1,1-1) ;
      matBgrande ( 6-1 , (k-1)*3 + j-1 ) = deriv(1-1,k-1) * F(j-1,2-1) + deriv(2-1,k-1) * F(j-1,1-1) ;
    }
  }
  
  return matBgrande;
}
// ======================================================================


// ======================================================================
vec tranvoigSin2( mat Tensor){
    
  vec v = zeros<vec>(6);
    
  v(1-1) = Tensor(1-1,1-1) ;
  v(2-1) = Tensor(2-1,2-1) ;
  v(3-1) = Tensor(3-1,3-1) ;
  v(4-1) = Tensor(2-1,3-1) ;
  v(5-1) = Tensor(1-1,3-1) ;
  v(6-1) = Tensor(1-1,2-1) ;
  
  return v;
}
// ======================================================================



// ======================================================================
//
// ======================================================================
mat constTensor ( vec hyperElasParamsVec, mat Egreen ){

  mat ConsMat = zeros<mat> (6,6);
  
  double young = hyperElasParamsVec(0)       ;
  double nu = hyperElasParamsVec(1)          ;
  double shear   = young / ( 2.0 * (1+ nu) ) ;

  ConsMat (1-1,1-1) = ( shear / (1 - 2 * nu) ) * 2 * ( 1-nu  ) ; 
  ConsMat (1-1,2-1) = ( shear / (1 - 2 * nu) ) * 2 * (   nu  ) ; 
  ConsMat (1-1,3-1) = ( shear / (1 - 2 * nu) ) * 2 * (   nu  ) ; 

  ConsMat (2-1,1-1) = ( shear / (1 - 2 * nu) ) * 2 * (    nu ) ;
  ConsMat (2-1,2-1) = ( shear / (1 - 2 * nu) ) * 2 * (  1-nu ) ;
  ConsMat (2-1,3-1) = ( shear / (1 - 2 * nu) ) * 2 * (    nu ) ;
  
  ConsMat (3-1,1-1) = ( shear / (1 - 2 * nu) ) * 2 * (   nu ) ;
  ConsMat (3-1,2-1) = ( shear / (1 - 2 * nu) ) * 2 * (   nu ) ;
  ConsMat (3-1,3-1) = ( shear / (1 - 2 * nu) ) * 2 * ( 1-nu ) ;
  
  ConsMat (4-1,4-1 ) = shear ;
  ConsMat (5-1,5-1 ) = shear ;
  ConsMat (6-1,6-1 ) = shear ;

  return ConsMat;
}
// ======================================================================








// =====================================================================
//  elementTetraSVKSolidInternLoadsTangMat
// =====================================================================
void elementTetraSVKSolidInternLoadsTangMat( mat tetCoordMat, mat elemDispMat, vec  hyperElasParamsVec, int paramOut, vec& Finte, mat& KTe){

  Finte = zeros<vec>( 12 ) ; 
  
  //~ eledispmat = reshape( Ue, 3,4) ;
  //~ elecoordspa = tetCoordMat + eledispmat ;

  double xi = 0.25, wi = 1.0/6.0  ;
      
  // matriz de derivadas de fun forma respecto a coordenadas isoparametricas
  mat deriv = shapeFunsDeriv( xi, xi , xi )  ;


  // jacobiano que relaciona coordenadas materiales con isoparametricas
  mat jacobianMat ;
  
  jacobianMat = tetCoordMat * deriv.t() ;

//~ cout << "-------------------" << endl;
//~ cout << "deriv:" << deriv << endl;
//~ cout << "tetcoordmat:" << tetCoordMat << endl;
//~ cout << "jacobianmat:" << jacobianMat << endl;
  
  double vol = det( jacobianMat ) * wi ;
  
  if (vol<0){
    cout << "Element with negative volume, check connectivity." << endl;
    exit(0);
  }

  mat funder = inv(jacobianMat).t() * deriv ;
  mat H = elemDispMat * funder.t() ;
  
  mat F = H + eye(3,3);


  mat Egreen = 0.5 * ( H + H.t() + H.t() * H ) ;

//~ cout << "Engree: " << Egreen << endl;
  
  mat S = cosseratSVK( hyperElasParamsVec, Egreen) ;

  mat matBgrande = BgrandeMats ( funder , F ) ;
     
  vec Svoigt = tranvoigSin2( S ) ;
  
  mat ConsMat = constTensor ( hyperElasParamsVec, Egreen ) ;
  
  
  //~ %~ Scons = ConsMat * tranvoigCon2(Egreen) ;

  Finte  = matBgrande.t() * Svoigt * vol ;
    
  if (paramOut == 2){

    mat Kml        = matBgrande.t() * ConsMat * matBgrande * vol ;
    mat matauxgeom = funder.t() * S * funder  * vol ;
    mat Kgl        = zeros<mat>(12,12) ;
    for (int i=1; i<=4 ; i++){
      for (int j=1; j<=4; j++){
        Kgl( (i-1)*3+1-1 , (j-1)*3+1-1 ) = matauxgeom(i-1, j-1);
        Kgl( (i-1)*3+2-1 , (j-1)*3+2-1 ) = matauxgeom(i-1, j-1);
        Kgl( (i-1)*3+3-1 , (j-1)*3+3-1 ) = matauxgeom(i-1, j-1);
      }
    }
    KTe = Kml + Kgl ;
  } // if param out
}
// =====================================================================






// =====================================================================
// assembler
// =====================================================================
void assembler(){




  
  //~ // -------------------------------------------------------------------
  //~ // calculos
  //~ // -------------------------------------------------------------------
  
  //~ vec FintGt = zeros<vec>(6*nnodes);
  
  //~ int numIndexsKT = nelems*12*12 ;
  
  //~ vec indsIKT = zeros<vec>( numIndexsKT ) ;
  //~ vec indsJKT = zeros<vec>( numIndexsKT ) ;
  //~ vec valsIKT = zeros<vec>( numIndexsKT ) ;
  //~ int indTotal = 0;

  //~ vec Finte(12) ; 
  //~ mat KTe(12,12);
    
  //~ ivec nodesElem, dofsElem;

  //~ mat tetCoordMat = zeros<mat>(3,4);
  //~ mat elemDispMat = zeros<mat>(3,4);
  //~ vec hyperElasParamsVec(2) ;
  //~ ivec dofsTet(12) ;
  
  //~ for( int elem = 1; elem <= nelems; elem++){
//    cout << " elem: " << elem << endl;
    //~ nodesElem = ( conec( span(elem-1,elem-1), span(0,3) )).t(); 
    //~ dofsElem  = nodes2dofs( nodesElem , 6 ) ;

    //~ for (int indi=1; indi<= 12; indi++){
      //~ dofsTet( indi-1 ) = dofsElem( (indi-1)*2 ) ;
    //~ }

    //~ for (int indi=1; indi<= 3; indi++){
      //~ for (int indj=1; indj<= 4; indj++){
        //~ tetCoordMat( indi-1, indj-1 ) = coordsElemsMat(elem-1, (indi-1)*2 + (indj-1)*6 ) ;
        //~ elemDispMat( indi-1, indj-1 ) = Ut( dofsElem ( (indi-1)*2 + (indj-1)*6 ) -1 ) ;
      //~ }
    //~ }

    //~ hyperElasParamsVec(0) = hyperElasParamsMat( conec(elem-1,5-1)-1 , 1 ) ;
    //~ hyperElasParamsVec(1) = hyperElasParamsMat( conec(elem-1,5-1)-1 , 2 ) ;
    
    //~ // --- computes internal loads or tanget matrix ---
    //~ elementTetraSVKSolidInternLoadsTangMat( tetCoordMat, elemDispMat, hyperElasParamsVec, paramOut, Finte, KTe) ;
    
    //~ // assembly Fint
    //~ for (int indi=1; indi<= 12; indi++){
      //~ FintGt( dofsTet( indi-1 ) -1) = FintGt( dofsTet( indi-1 ) -1) + Finte( indi-1 ) ;
    //~ }
    
    //~ if (paramOut == 2){  
//          indVec = (indRow+1)/2 ;
      //~ for (int indi=1; indi<=12; indi++){
        //~ for (int indj=1; indj<=12; indj++){
          //~ indTotal++;     
        
          //~ indsIKT ( indTotal-1 ) = dofsTet( indi-1 )     ;
          //~ indsJKT ( indTotal-1 ) = dofsTet( indj-1 )     ;  
          //~ valsIKT ( indTotal-1 ) = KTe( indi-1, indj-1 ) ;
        //~ }     
      //~ }
    //~ } // if paramOut 2
    

  //~ } // ---   for elements -------------------------------
  
  //~ FintGt.save("FintGt.dat", raw_ascii);
  
  //~ if (paramOut == 2){  
    //~ indsIKT.save("indsIKT.dat", raw_ascii);
    //~ indsJKT.save("indsJKT.dat", raw_ascii);
    //~ valsIKT.save("valsIKT.dat", raw_ascii);
  //~ }
  
}



void extractMethodParams( mat numericalMethodParams, int & solutionMethod, \
                          double & stopTolDeltau, double & stopTolForces,  \
                          int & stopTolIts, double & targetLoadFactr, \
                          int & nLoadSteps, double & incremArcLen, \
                          double & deltaT, double & deltaNW, double & AlphaNW, \
                          double & alphaHHT, double & finalTime ){
  
  solutionMethod   = numericalMethodParams(1-1) ;
  
  if (solutionMethod == 1){

    // ----- resolution method params -----
    stopTolDeltau    = numericalMethodParams(2-1) ;
    stopTolForces    = numericalMethodParams(3-1) ;
    stopTolIts       = numericalMethodParams(4-1) ;
    targetLoadFactr  = numericalMethodParams(5-1) ;
    nLoadSteps       = numericalMethodParams(6-1) ;
  
    incremArcLen     = 0 ;
   
    deltaT = targetLoadFactr / double( nLoadSteps) ;
    
    finalTime = targetLoadFactr ;
    
    deltaNW =  0; AlphaNW = 0 ; alphaHHT = 0 ;
  }
}

// =============================================================================
//  main
// =============================================================================
int main(){
  
  cout << "\n=============================" << endl;
  cout << "=== C++ timeStepIteration ===" << endl;

  // ---------------------------------------------------------------------------
  // --------                       reading                          -----------
  // ---------------------------------------------------------------------------
  cout << "  reading inputs..." ;
  
  // declarations of variables read
  imat conec;
  mat numericalMethodParams;
  sp_mat systemDeltauMatrix;
  vec U, Fint;
  
  // reading
  conec.load("Conec.dat");
  numericalMethodParams.load("numericalMethodParams.dat");
  systemDeltauMatrix.load("systemDeltauMatrix.dat", coord_ascii);
  systemDeltauMatrix = systemDeltauMatrix.tail_rows(systemDeltauMatrix.n_rows-1);
  systemDeltauMatrix = systemDeltauMatrix.tail_cols(systemDeltauMatrix.n_cols-1);
  U.load("U.dat");
  Fint.load("Fint.dat");
  
  cout << " done" << endl ;
  // ---------------------------------------------------------------------------
  

  // ---------------------------------------------------------------------------
  // --------                       pre                              -----------
  // ---------------------------------------------------------------------------
  
  // declarations
  int nelems    = conec.n_rows ;  int ndofpnode = 6;
  int solutionMethod, nLoadSteps, stopTolIts ;
  double stopTolDeltau, stopTolForces, targetLoadFactr, \
    incremArcLen, deltaT, deltaNW, AlphaNW, alphaHHT, finalTime ;  

  extractMethodParams( numericalMethodParams, solutionMethod, stopTolDeltau, \
    stopTolForces, stopTolIts, targetLoadFactr, nLoadSteps, incremArcLen, \
    deltaT, deltaNW, AlphaNW, alphaHHT, finalTime );
  // ---------------------------------------------------------------------------


  // ---------------------------------------------------------------------------
  // ----       iteration in displacements or load-displacements         -------
  // ---------------------------------------------------------------------------
  
  sp_mat KTtred = systemDeltauMatrix ;

  // assign disps and forces at time t
  vec Ut    = U    ;
  vec Fintt = Fint ;
  
  // start iteration with previous displacements ---
  vec Utp1k = Ut       ;   // initial guess
  
  cout << ndofpnode << nelems << endl;


  return 0;
}
// =============================================================================



  //~ varsInps.load("varsInps.dat");
  //~ coordsElemsMat.load("coordsElemsMat.dat");
  //~ hyperElasParamsMat.load("materialsParamsMat.dat");

  // --- processing ---  
  
  //~ int nnodes = ( varsInps.n_rows -1) / 6 ;
  //~ int paramOut = varsInps( varsInps.n_rows - 1 ) ;
  //~ vec Ut = varsInps( span(0,varsInps.n_rows - 2), span(0,0) ) ;

  //~ cout << "  paramOut: " << paramOut << endl;
  //~ cout << "  nnodes: " << nnodes << endl;
  // -------------------------------------------------------------------
  
  //~ cout << "Ut: " << Ut << endl;