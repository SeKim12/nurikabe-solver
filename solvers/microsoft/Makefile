# Define the compiler and compiler flags
CXX = g++
CXXFLAGS = -std=gnu++20

# Define the target executable and source files
TARGET = microsoft_solver
SRCS = microsoft_solver.cpp

# Rule to build the target executable
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

# Rule to clean up generated files
clean:
	rm -f $(TARGET)