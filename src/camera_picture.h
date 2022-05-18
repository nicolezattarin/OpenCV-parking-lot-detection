#ifndef CAMERA_PICTURE_H
#define CAMERA_PICTURE_H

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include "parking.h"

using namespace std;

/**
 * @brief Construct a new camera picture object, 
 * defined as an object with a picture and a series of parking slots
 */
class camera_picture
{
    public:
        camera_picture();
        camera_picture(vector<Parking>, cv::Mat);
        camera_picture(vector<Parking>, cv::Mat, cv::Mat);
        camera_picture(vector<Parking>, cv::Mat, cv::Mat, string);
        ~camera_picture(){};

        // setters/getters
        void setParking(vector<Parking>);
        vector<Parking> getParking();
        void setImg(cv::Mat);
        cv::Mat getImg();
        void setImgParkingLots(cv::Mat);
        cv::Mat getImgParkingLots();

    private:
        vector<Parking> m_parkings; //all the parkings in the image
        cv::Mat m_image; //image of the specific camera
        cv::Mat m_image_parking_lots; //image with rectangles around the parking lots
        string m_path; //path to the image
};
#endif