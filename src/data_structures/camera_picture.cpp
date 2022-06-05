#include "camera_picture.h"

/**
 * @brief default constructor
 */
camera_picture :: camera_picture(){
    m_parkings = vector<Parking>();
    m_image = cv::Mat();
    m_image_parking_lots = cv::Mat();
    m_capture_date = ""; // default, non initialized
    m_capture_time = -1;// default, non initialized
    m_blob_img = cv::Mat();
    m_avg_rotation_angle = 0;
}

/**
 * @brief constructor with parameters
 * 
 * @param parkings 
 * @param image 
 * @return camera_picture 
 */
camera_picture :: camera_picture(vector<Parking> parkings, cv::Mat image){
    m_parkings = parkings;
    m_image = image;
    m_image_parking_lots = cv::Mat();
    m_capture_date = ""; // default, non initialized
    m_capture_time = -1;// default, non initialized
    m_blob_img = cv::dnn::blobFromImage(image, 1, cv::Size(150, 150), cv::Scalar(104, 117, 123));
    m_avg_rotation_angle = 0;
}

/**
 * @brief constructor with parameters, set images and parking lots
 * 
 * @param parkings 
 * @param image 
 * @param image_parking_lots 
 * @return camera_picture 
 */
camera_picture :: camera_picture(vector<Parking> parkings, cv::Mat image, cv::Mat image_parking_lots){
    m_parkings = parkings;
    m_image = image;
    m_image_parking_lots = image_parking_lots;
    m_capture_date = ""; // default, non initialized
    m_capture_time = -1;// default, non initialized
    m_blob_img = cv::dnn::blobFromImage(image, 1, cv::Size(150, 150), cv::Scalar(104, 117, 123));
    m_avg_rotation_angle = 0;
}

/**
 * @brief constructor with parameters, sets information about the capture date and time
 * 
 * @param parkings 
 * @param image 
 * @param image_parking_lots 
 * @param path 
 * @param capture_date 
 * @param capture_time 
 * @return camera_picture 
 */
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

/**
 * @brief constructor with parameters, sets information about the capture date and time and the average rotation angle
 * 
 * @param parkings 
 * @param image 
 * @param image_parking_lots 
 * @param path 
 * @param date 
 * @param time 
 * @param avg_rotation 
 * @return camera_picture 
 */
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

/**
 * @brief Set the Parking object
 * 
 * @param parkings 
 */
void camera_picture :: setParking(vector<Parking> parkings){
    m_parkings = parkings;
    // for (int i; i<parkings.size();i++){
    //     cout << parkings[i].getId() << endl;

    //     m_parkings.push_back(parkings[i]);
    // }
}

/**
 * @brief Get the Parking object
 * 
 * @return vector <Parking> 
 */
vector <Parking> camera_picture :: getParking(){
    return m_parkings;
}

/**
 * @brief Set the Img object
 * 
 * @param image 
 */
void camera_picture :: setImg(cv::Mat image){
    m_image = image;
}

/**
 * @brief Get the Img object
 * 
 * @return cv::Mat 
 */
cv::Mat camera_picture :: getImg(){
    return m_image;
}

/**
 * @brief Set the Img Parking Lots object
 * 
 * @param image 
 */
void camera_picture :: setImgParkingLots(cv::Mat image){
    m_image_parking_lots = image;
}

/**
 * @brief Get the Img Parking Lots object
 * 
 * @return cv::Mat 
 */
cv::Mat camera_picture :: getImgParkingLots(){
    return m_image_parking_lots;
}

/**
 * @brief Set the avg rotation object
 * 
 * @param avg_rotation 
 */
void camera_picture :: set_avg_rotation(float avg_rotation){
    m_avg_rotation_angle = avg_rotation;
}

/**
 * @brief Get the avg rotation object
 * 
 * @return float 
 */
float camera_picture :: get_avg_rotation(){
    return m_avg_rotation_angle;
}

/**
 * @brief Get the capture date object
 * 
 * @return string 
 */
string camera_picture :: get_capture_date(){
    return m_capture_date;
}

/**
 * @brief Get the capture time object
 * 
 * @return string 
 */
string camera_picture :: get_capture_time(){
    return m_capture_time;
}
