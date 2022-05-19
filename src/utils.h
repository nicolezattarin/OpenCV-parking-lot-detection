#ifndef UTILS_H
#define UTILS_H
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <string>
#include "parking.h"
#include <glob.h>
#include <sstream>
#include "camera_picture.h"

vector<Parking> ReadCameraCSV(string);
vector<camera_picture> ReadImages(int, string);
char to_upper_char (char);
string to_upper (string);
void test(string);
vector<string> glob_path(const string& );
void ClassifyParkings(string, vector<camera_picture>& );

#endif // UTILS_H