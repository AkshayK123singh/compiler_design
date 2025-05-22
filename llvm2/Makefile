CXX=clang++
CXXFLAGS=`llvm-config --cxxflags` -g -std=c++17
LDFLAGS=`llvm-config --ldflags --libs core orcjit native` -lpthread -ldl

.PHONY: all clean

all: parser

parser.hpp parser.cpp: parser.y
	bison -d -o parser.cpp parser.y

lexer.cpp: lexer.l parser.hpp
	flex -o $@ $<

lexer.o: lexer.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

parser.o: parser.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

parser: lexer.o parser.o main.o
	$(CXX) -o $@ $^ $(LDFLAGS)

clean:
	rm -f parser *.o lexer.cpp parser.cpp parser.hpp