# based on https://www.cs.bu.edu/teaching/cpp/writing-makefiles/

# compiler
CXX = g++

# flag of compiler
CXXFLAGS = -Wall -g -O2 -larmadillo

EXE = timeStepIteration.lnx

SRC= src

# target: dependencies
# TAB command to generate the target
main: timeStepIteration.o nodes2dofs.o
	$(CXX) -o $(EXE) timeStepIteration.o nodes2dofs.o $(CXXFLAGS)

timeStepIteration.o:
	$(CXX) $(CXXFLAGS) -c timeStepIteration.cpp


%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<
	
clean:
	rm $(EXE) *.o

# option direct from console without make:
#  g++ timeStepIteration.cpp -o timeStepIteration.lnx -O2 -larmadillo
