#include "parking.h"

Parking :: Parking(){
    m_isEmpty = true;
    m_id = -1;
    m_x = -1;
    m_y = -1;
    m_width = -1;
    m_height = -1; 
    m_image = cv::Mat();
}

Parking :: Parking(int id, int x, int y, int width, int height){
    m_isEmpty = true;
    m_image = cv::Mat();
    m_id = id;
    m_x = x;
    m_y = y;
    m_width = width;
    m_height = height;
}

Parking :: Parking(int id, int x, int y, int width, int height, cv::Mat image){
    m_isEmpty = true;
    m_image = image;
    m_id = id;
    m_x = x;
    m_y = y;
    m_width = width;
    m_height = height;
}

// setters/getters
void Parking :: setEmpty(bool isEmpty){
    m_isEmpty = isEmpty;
}

bool Parking :: isEmpty(){
    return m_isEmpty;
}

void Parking :: setId(int id){
    m_id = id;
}

int Parking :: getId(){
    return m_id;
}

void Parking :: setX(int x){
    m_x = x;
}

int Parking :: getX(){
    return m_x;
}

void Parking :: setY(int y){
    m_y = y;
}

int Parking :: getY(){
    return m_y;
}

void Parking :: setWidth(int width){
    m_width = width;
}

int Parking :: getWidth(){
    return m_width;
}

void Parking :: setHeight(int height){
    m_height = height;
}

int Parking :: getHeight(){
    return m_height;
}

void Parking :: setImg(cv::Mat img){
    m_image = img;
}

cv::Mat Parking :: getImg(){
    return m_image;
}