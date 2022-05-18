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

vector<Parking> ReadCameraCSV(string filename);
vector<camera_picture> ReadImages(int camera_number, string weather);
char to_upper_char (char c);
string to_upper (string s);
void test(string path);
vector<string> glob_path(const string& pattern);

#endif // UTILS_H