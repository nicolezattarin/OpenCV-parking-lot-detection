#ifndef PREPROCESSING_H
#define PREPROCESSING_H

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


float get_average_rotation(cv::Mat, int, int, int, string, float, string , int );
void DrawLines( cv::Mat&, vector<cv::Vec2f>);
vector<cv::Point> PolarToCartesian(float, float);
void preprocess(cv::Mat& src, cv::Mat& dst, bool, bool);
void preprocess_patches (vector<camera_picture> , bool);
cv::Mat preprocess_patch(cv::Mat, float, bool);
void equalize(cv::Mat&, cv::Mat&, std::vector<cv::Mat>&, std::vector<cv::Mat>& , string );

#endif // PREPROCESSING_H