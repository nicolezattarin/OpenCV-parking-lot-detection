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
        Parking(int id, int x, int y, int width, int height);
        ~Parking(){};

        // setters/getters
        void setEmpty(bool isEmpty);
        bool isEmpty();
        void setId(int id);
        int getId();
        void setX(int x);
        int getX();
        void setY(int y);
        int getY();
        void setWidth(int width);
        int getWidth();
        void setHeight(int height);
        int getHeight();

    private:
        bool m_isEmpty;
        int m_id;
        int m_x;
        int m_y;
        int m_width;
        int m_height;
};

#endif // PARKING_H