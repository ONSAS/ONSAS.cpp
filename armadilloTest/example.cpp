// from http://arma.sourceforge.net/docs.html#example_prog
// If you save the above program as example.cpp, under Linux and Mac OS X it can be compiled using:
// g++ example.cpp -o example.lnx -O2 -larmadillo

#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main()
  {
  mat A = randu<mat>(4,5);
  mat B = randu<mat>(4,5);
  
  cout << A*B.t() << endl;
  
  return 0;
  }
