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
        camera_picture(vector<Parking>, cv::Mat, cv::Mat, string, string, string);
        ~camera_picture(){};

        // setters/getters
        void setParking(vector<Parking>);
        vector<Parking> getParking();
        void setImg(cv::Mat);
        cv::Mat getImg();
        void setImgParkingLots(cv::Mat);
        cv::Mat getImgParkingLots();
        void set_avg_rotation(float);
        float get_avg_rotation();

    private:
        vector<Parking> m_parkings; //all the parkings in the image
        cv::Mat m_image; //image of the specific camera
        cv::Mat m_image_parking_lots; //image with rectangles around the parking lots
        string m_path; //path to the image, format <CAPTURE_DATE>_<CAPTURE_TIME>.jpg,
        string m_capture_date;
        string m_capture_time;
        cv::Mat m_blob_img;
        float m_avg_rotation_angle; //for each picture of the same camera we perform edge 
                                    //detection to find the proper rotation angle for patches

};
#endif