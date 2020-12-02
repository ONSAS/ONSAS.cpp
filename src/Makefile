# based on https://www.cs.bu.edu/teaching/cpp/writing-makefiles/

# compiler
CXX = g++

# flag of compiler
CXXFLAGS = -Wall -g -O2 -larmadillo

EXE = timeStepIteration.lnx

# target: dependencies
# TAB command to generate the target
main: timeStepIteration.o
	$(CXX) -o $(EXE) timeStepIteration.o $(CXXFLAGS)

timeStepIteration.o:
	$(CXX) $(CXXFLAGS) -c timeStepIteration.cpp
	
clean:
	rm $(EXE) *.o

# option direct from console without make:
#  g++ timeStepIteration.cpp -o timeStepIteration.lnx -O2 -larmadillo
