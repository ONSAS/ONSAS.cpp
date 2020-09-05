# based on https://www.cs.bu.edu/teaching/cpp/writing-makefiles/

# compiler
CXX = g++

# flag of compiler
#~ CXXFLAGS = -Wall -g -O2 -larmadillo
CXXFLAGS = -Wall -g -O2 -larmadillo

#~ EXE = sts
EXE = onsasAssembler.lnx

SRC= src

# target: dependencies
# TAB command to generate the target
#~ sts: timeStep.o
main: onsasAssembler.o
	$(CXX) -o $(EXE) onsasAssembler.o $(CXXFLAGS)

onsasAssembler.o:
	$(CXX) $(CXXFLAGS) -c onsasAssembler.cpp
#~ 	$(CXX) $(CXXFLAGS) -o $(EXE) timeStep.o

clean:
	rm $(EXE) *.o

# option direct from console without make:
#  g++ onsasAssembler.cpp -o onsasAssembler -O2 -larmadillo
