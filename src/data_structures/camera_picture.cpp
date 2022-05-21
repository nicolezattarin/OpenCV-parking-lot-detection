#include "camera_picture.h"

camera_picture :: camera_picture(){
    m_parkings = vector<Parking>();
    m_image = cv::Mat();
    m_image_parking_lots = cv::Mat();
    m_capture_date = ""; // default, non initialized
    m_capture_time = -1;// default, non initialized
    m_blob_img = cv::Mat();
    m_avg_rotation_angle = 0;
}

camera_picture :: camera_picture(vector<Parking> parkings, cv::Mat image){
    m_parkings = parkings;
    m_image = image;
    m_image_parking_lots = cv::Mat();
    m_capture_date = ""; // default, non initialized
    m_capture_time = -1;// default, non initialized
    m_blob_img = cv::dnn::blobFromImage(image, 1, cv::Size(150, 150), cv::Scalar(104, 117, 123));
    m_avg_rotation_angle = 0;
}

camera_picture :: camera_picture(vector<Parking> parkings, cv::Mat image, cv::Mat image_parking_lots){
    m_parkings = parkings;
    m_image = image;
    m_image_parking_lots = image_parking_lots;
    m_capture_date = ""; // default, non initialized
    m_capture_time = -1;// default, non initialized
    m_blob_img = cv::dnn::blobFromImage(image, 1, cv::Size(150, 150), cv::Scalar(104, 117, 123));
    m_avg_rotation_angle = 0;
}

camera_picture :: camera_picture(vector<Parking> parkings, cv::Mat image, cv::Mat image_parking_lots, 
                                string path, string date, string time){
    m_parkings = parkings;
    m_image = image;
    m_image_parking_lots = image_parking_lots;
    m_path = path;
    m_blob_img = cv::dnn::blobFromImage(image, 1, cv::Size(150, 150), cv::Scalar(104, 117, 123));
    m_capture_date = date;
    m_capture_time = time;
    m_avg_rotation_angle = 0;
}

camera_picture :: camera_picture(vector<Parking> parkings, cv::Mat image, cv::Mat image_parking_lots, 
                                string path, string date, string time, float avg_rotation){
    m_parkings = parkings;
    m_image = image;
    m_image_parking_lots = image_parking_lots;
    m_path = path;
    m_blob_img = cv::dnn::blobFromImage(image, 1, cv::Size(150, 150), cv::Scalar(104, 117, 123));
    m_capture_date = date;
    m_capture_time = time;
    m_avg_rotation_angle = avg_rotation;
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

void camera_picture :: set_avg_rotation(float avg_rotation){
    m_avg_rotation_angle = avg_rotation;
}

float camera_picture :: get_avg_rotation(){
    return m_avg_rotation_angle;
}

string camera_picture :: get_capture_date(){
    return m_capture_date;
}

string camera_picture :: get_capture_time(){
    return m_capture_time;
}
