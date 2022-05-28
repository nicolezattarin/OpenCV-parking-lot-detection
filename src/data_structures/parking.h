#ifndef PARKING_H
#define PARKING_H

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>

using namespace std;

/**
 * @brief parking class for CNR dataset
 */

class Parking
{
    public:
        Parking();
        Parking(int, int, int, int, int);
        Parking(int, int, int, int, int, cv::Mat);
        Parking(int, int, int, float, cv::Point, cv::Point, cv::Point, cv::Point, cv::Point);
        Parking(int, int, int, cv::Mat, float, cv::Point, cv::Point, cv::Point, cv::Point, cv::Point);

        ~Parking(){};

        // setters/getters
        void setStatus(bool);
        bool getStatus();
        void setId(int);
        int getId();
        void setX(int);
        int getX();
        void setY(int);
        int getY();
        void setWidth(int);
        int getWidth();
        void setHeight(int);
        int getHeight();
        void setImg(cv::Mat);
        cv::Mat getImg();
        void setAngle(float);
        float getAngle();
        cv::Point getLowerLeft();
        cv::Point getUpperRight();
        cv::Point getLowerRight();
        cv::Point getUpperLeft();
        cv::Point getCenter();
        void GetInfo();

    private:
        bool m_status;
        int m_id;
        int m_x;
        int m_y;
        int m_width;
        int m_height;
        float m_angle;
        cv::Point m_center;
        cv::Point m_lower_left;
        cv::Point m_upper_right;
        cv::Point m_lower_right;
        cv::Point m_upper_left;
        cv::Mat m_image; //image of the specific slot
};

#endif // PARKING_H