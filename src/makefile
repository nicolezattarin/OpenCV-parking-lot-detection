CFLAGS = `pkg-config --cflags opencv4`
LIBS = `pkg-config --libs opencv4`
main: main.o utils.o parking.o camera_picture.o
	g++ -std=gnu++17 main.o utils.o parking.o camera_picture.o ${CFLAGS} ${LIBS} -o main

main.o: main.cpp utils.h parking.h camera_picture.h
	g++ -c -std=gnu++17 main.cpp -o main.o ${CFLAGS} ${LIBS}

%.o: %.cpp %.h
	g++ -c -std=gnu++17 $< ${CFLAGS} ${LIBS} -o $@

clean:
	rm *.o
	rm main