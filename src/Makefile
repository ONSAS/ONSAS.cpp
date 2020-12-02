# based on https://www.cs.bu.edu/teaching/cpp/writing-makefiles/

# directorios
HDIR    = ../include
CXXDIR  = src
ODIR    = obj

# compiler
CXX = g++

# flag of compiler
CXXFLAGS = -Wall -I$(HDIR) -g -O2 -larmadillo

EXE = timeStepIteration.lnx

# target: dependencies
# TAB command to generate the target
main: timeStepIteration.o shapeFunsDeriv.o cosseratSVK.o
	$(CXX) -o $(EXE) timeStepIteration.o shapeFunsDeriv.o cosseratSVK.o $(CXXFLAGS)

timeStepIteration.o:
	$(CXX) $(CXXFLAGS) -c timeStepIteration.cpp

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
	
clean:
	rm $(EXE) *.o

# option direct from console without make:
#  g++ timeStepIteration.cpp -o timeStepIteration.lnx -O2 -larmadillo
