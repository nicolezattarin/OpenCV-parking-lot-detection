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
#include "../data_structures/parking.h"
#include <glob.h>
#include <sstream>
#include "../data_structures/camera_picture.h"
#include "preprocessing.h"
// void preprocess_image(cv::Mat, cv::Mat&);

vector<camera_picture> GetCameraPictures(vector<string>, vector<Parking>, int , string );

vector<Parking> ReadCameraCSV(string);
vector<camera_picture> ReadImages(int, string);
char to_upper_char (char);
string to_upper (string);
void test(string);
vector<string> glob_path(const string&);
void save_patches(vector<camera_picture>, string, int);

#endif // UTILS_H