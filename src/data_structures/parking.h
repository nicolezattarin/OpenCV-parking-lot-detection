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
        ~Parking(){};

        // setters/getters
        void setEmpty(bool);
        bool isEmpty();
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

    private:
        bool m_isFree;
        int m_id;
        int m_x;
        int m_y;
        int m_width;
        int m_height;
        cv::Mat m_image; //image of the specific slot
};

#endif // PARKING_H