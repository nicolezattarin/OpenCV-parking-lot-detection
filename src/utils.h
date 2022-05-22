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
#include "data_structures/parking.h"
#include <glob.h>
#include <sstream>
#include "data_structures/camera_picture.h"
#include "preprocessing/preprocessing.h"
#include <filesystem>

// void preprocess_image(cv::Mat, cv::Mat&);

vector<camera_picture> GetCameraPictures(vector<string>, vector<Parking>, int , string, bool, bool, bool );
vector<Parking> ReadCameraCSV(string);
vector<camera_picture> ReadImages(int, string, bool, bool, bool);
char to_upper_char (char);
char to_lower_char (char);
string to_upper (string);
string to_lower (string);
void test(string);
vector<string> glob_path(const string&);
void save_patches(vector<camera_picture>, string, int, bool, bool, bool);
void ReadClassifiedSamples(vector<camera_picture>&, int, string, bool, bool, bool);
void ReadClassifiedSamples(camera_picture&, int, string, bool, bool, bool);
bool check_if_free(string, string);
void draw_free_lots(vector<camera_picture>& );
void draw_free_lots(camera_picture& );
void save_images_with_lots(vector<camera_picture>& , int , string, bool, bool, bool );

#endif // UTILS_H