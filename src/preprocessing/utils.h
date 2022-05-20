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

// void preprocess_image(cv::Mat, cv::Mat&);

float get_average_rotation(cv::Mat, int, int, int, string, float, string , int );
vector<camera_picture> GetCameraPictures(vector<string>, vector<Parking>, int , string );

vector<Parking> ReadCameraCSV(string);
vector<camera_picture> ReadImages(int, string);
char to_upper_char (char);
string to_upper (string);
void test(string);
vector<string> glob_path(const string&);
void DrawLines( cv::Mat& img, vector<cv::Vec2f>lines);
vector<cv::Point> PolarToCartesian(float, float);

cv::Mat preprocess_patch(cv::Mat, float);
void preprocess (vector<camera_picture>);
void equalize(cv::Mat& img, cv::Mat& equalized_img, 
            std::vector<cv::Mat>& hists, std::vector<cv::Mat>& equalized_hists, 
            string color_space);
#endif // UTILS_H