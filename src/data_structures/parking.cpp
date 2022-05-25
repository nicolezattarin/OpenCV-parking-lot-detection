#include "parking.h"

Parking :: Parking(){
    m_isFree = true;
    m_id = -1;
    m_x = -1;
    m_y = -1;
    m_width = -1;
    m_height = -1; 
    m_image = cv::Mat();
    m_lower_left = cv::Point();
    m_upper_right = cv::Point();
    m_lower_right = cv::Point();
    m_upper_left = cv::Point();
}

Parking :: Parking(int id, int x, int y, int width, int height){
    m_isFree = true;
    m_image = cv::Mat();
    m_id = id;
    m_x = x;
    m_y = y;
    m_width = width;
    m_height = height;
    //assuming that if someone uses this constructor, the patch's axes are parallel to the image axes
    m_lower_left = cv::Point(x, y);
    m_upper_right = cv::Point(x + width, y + height);
    m_lower_right = cv::Point(x + width, y);
    m_upper_left = cv::Point(x, y + height);
}

Parking :: Parking(int id, int x, int y, int width, int height, cv::Mat image){
    m_isFree = true;
    m_image = image;
    m_id = id;
    m_x = x;
    m_y = y;
    m_width = width;
    m_height = height;
    //assuming that if someone uses this constructor, the patch's axes are parallel to the image axes
    m_lower_left = cv::Point(x, y);
    m_upper_right = cv::Point(x + width, y + height);
    m_lower_right = cv::Point(x + width, y);
    m_upper_left = cv::Point(x, y + height);
}

Parking :: Parking(int id, int width, int height, float angle, cv::Point center,
                cv::Point lower_left, cv::Point upper_right, cv::Point lower_right, cv::Point upper_left){
    m_isFree = true;
    m_id = id;
    m_image = cv::Mat();
    m_x = -1; //not initialized
    m_y = -1; //not initialized
    m_width = width;
    m_height = height;
    m_angle = angle;
    m_center = center;

    m_lower_left = lower_left;
    m_upper_right = upper_right;
    m_lower_right = lower_right;
    m_upper_left = upper_left;
}
Parking :: Parking(int id, int width,  int height, cv::Mat img, float angle, cv::Point center,
                cv::Point lower_left, cv::Point upper_right, cv::Point lower_right, cv::Point upper_left){
    m_isFree = true;
    m_id = id;
    m_image = img;
    m_x = -1; //not initialized
    m_y = -1; //not initialized
    m_width = width;
    m_height = height;
    m_angle = angle;
    m_center = center;

    m_lower_left = lower_left;
    m_upper_right = upper_right;
    m_lower_right = lower_right;
    m_upper_left = upper_left;
}
// setters/getters
void Parking :: setStatus(bool isEmpty){
    m_isFree = isEmpty;
}

bool Parking :: getStatus(){
    return m_isFree;
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

float Parking :: getAngle(){
    return m_angle;
}

void Parking :: setAngle(float angle){
    m_angle = angle;
}

cv::Point Parking :: getLowerLeft(){
    return m_lower_left;
}

cv:: Point Parking :: getUpperRight(){
    return m_upper_right;
}

cv::Point Parking :: getLowerRight(){
    return m_lower_right;
}

cv::Point Parking :: getUpperLeft(){
    return m_upper_left;
}

cv::Point Parking :: getCenter(){
    return m_center;
}

