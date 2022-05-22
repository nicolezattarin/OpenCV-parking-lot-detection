CV_CFLAGS = `pkg-config --cflags opencv4`
CV_LIBS = `pkg-config --libs opencv4`
OBJ = main.o ../utils.o  ../data_structures/parking.o  ../data_structures/camera_picture.o ../preprocessing/preprocessing.o
INC = ../utils.h  ../data_structures/parking.h  ../data_structures/camera_picture.h ../preprocessing/preprocessing.h 

main: ${OBJ}
	g++ -std=gnu++17 ${OBJ} ${CV_CFLAGS} ${CV_LIBS} -o main

main.o: main.cpp ${INC}
	g++ -c -std=gnu++17 main.cpp -o main.o ${CV_CFLAGS} ${CV_LIBS}

utils.o: ../utils.cpp  ../utils.h
	g++ -c -std=gnu++17 ../utils.cpp ${CV_CFLAGS} ${CV_LIBS} -o ../utils.o

parking.o: ../data_structures/parking.cpp ../data_structures/parking.h
	g++ -c -std=gnu++17 ../data_structures/parking.cpp ${CV_CFLAGS} ${CV_LIBS} -o ../data_structures/parking.o

camera_picture.o: ../data_structures/camera_picture.cpp ../data_structures/camera_picture.h
	g++ -c -std=gnu++17 ../data_structures/camera_picture.cpp ${CV_CFLAGS} ${CV_LIBS} -o ../data_structures/camera_picture.o

preprocessing.o: ../preprocessing/preprocessing.cpp ../preprocessing/preprocessinge.h
	g++ -c -std=gnu++17 ../preprocessing/preprocessing.cpp ${CV_CFLAGS} ${CV_LIBS} -o ../preprocessing/preprocessing.o

%.o: %.cpp %.h
	g++ -c -std=gnu++17 $< ${CV_CFLAGS} ${CV_LIBS} -o $@

clean:
	rm *.o
	rm main
