#include "camera_picture.h"

camera_picture :: camera_picture(){
    m_parkings = vector<Parking>();
    m_image = cv::Mat();
    m_image_parking_lots = cv::Mat();
}

camera_picture :: camera_picture(vector<Parking> parkings, cv::Mat image){
    m_parkings = parkings;
    m_image = image;
    m_image_parking_lots = cv::Mat();
}

camera_picture :: camera_picture(vector<Parking> parkings, cv::Mat image, cv::Mat image_parking_lots){
    m_parkings = parkings;
    m_image = image;
    m_image_parking_lots = image_parking_lots;
}

void camera_picture :: setParking(vector<Parking> parkings){
    m_parkings = parkings;
}

vector <Parking> camera_picture :: getParking(){
    return m_parkings;
}

void camera_picture :: setImg(cv::Mat image){
    m_image = image;
}

cv::Mat camera_picture :: getImg(){
    return m_image;
}

void camera_picture :: setImgParkingLots(cv::Mat image){
    m_image_parking_lots = image;
}

cv::Mat camera_picture :: getImgParkingLots(){
    return m_image_parking_lots;
}

