CFLAGS = `pkg-config --cflags opencv4`
LIBS = `pkg-config --libs opencv4`ù
%.o: %.cpp %.h
	g++ -c -std=gnu++17 $< ${CFLAGS} ${LIBS} -o $@
