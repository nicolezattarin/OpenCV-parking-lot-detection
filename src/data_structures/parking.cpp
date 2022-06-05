#include "parking.h"

/**
 * @brief default constructor
 * 
 * @return Parking 
 */
Parking :: Parking(){
    m_status = true;
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

/**
 * @brief constructor with parameters
 * 
 * @param id 
 * @param x: lower left corner x coordinate
 * @param y: lower left corner y coordinate
 * @param width 
 * @param height 
 * @return Parking 
 */
Parking :: Parking(int id, int x, int y, int width, int height){
    m_status = true;
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

/**
 * @brief 
 * 
 * @param id 
 * @param x: lower left corner x
 * @param y: lower left corner y
 * @param width 
 * @param height 
 * @param image 
 * @return Parking 
 */
Parking :: Parking(int id, int x, int y, int width, int height, cv::Mat image){
    m_status = true;
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

/**
 * @brief for pklot dataset, the parking lot is a rectangle, so we can use the rectangle's coordinates to set the parking lot's coordinates
 * 
 * @param id 
 * @param width 
 * @param height 
 * @param angle 
 * @param center 
 * @param lower_left 
 * @param upper_right 
 * @param lower_right 
 * @param upper_left 
 * @return Parking 
 */
Parking :: Parking(int id, int width, int height, float angle, cv::Point center,
                cv::Point lower_left, cv::Point upper_right, cv::Point lower_right, cv::Point upper_left){
    m_status = true;
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

/**
 * @brief constructor with parameters for pklot dataset
 * 
 * @param id 
 * @param width 
 * @param height 
 * @param img 
 * @param angle 
 * @param center 
 * @param lower_left 
 * @param upper_right 
 * @param lower_right 
 * @param upper_left 
 * @return Parking 
 */
Parking :: Parking(int id, int width,  int height, cv::Mat img, float angle, cv::Point center,
                cv::Point lower_left, cv::Point upper_right, cv::Point lower_right, cv::Point upper_left){
    m_status = true;
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

/**
 * @brief Set the Status object
 * 
 * @param status 
 */
void Parking :: setStatus(bool status){
    m_status = status;
}

/**
 * @brief Get the Status object
 * 
 * @return true 
 * @return false 
 */
bool Parking :: getStatus(){
    return m_status;
}

/**
 * @brief Set the Id object
 * 
 * @param id 
 */
void Parking :: setId(int id){
    m_id = id;
}

/**
 * @brief Get the Id object
 * 
 * @return int 
 */
int Parking :: getId(){
    return m_id;
}

/**
 * @brief Set the X object
 * 
 * @param x 
 */
void Parking :: setX(int x){
    m_x = x;
}

/**
 * @brief get the X object
 * 
 * @return int 
 */
int Parking :: getX(){
    return m_x;
}

/**
 * @brief set the Y object
 * 
 * @param y 
 */
void Parking :: setY(int y){
    m_y = y;
}

/**
 * @brief get the Y object
 * 
 * @return int 
 */
int Parking :: getY(){
    return m_y;
}

/**
 * @brief Set the Width object
 * 
 * @param width 
 */
void Parking :: setWidth(int width){
    m_width = width;
}

/**
 * @brief Get the Width object
 * 
 * @return int 
 */
int Parking :: getWidth(){
    return m_width;
}

/**
 * @brief Set the Height object
 * 
 * @param height 
 */
void Parking :: setHeight(int height){
    m_height = height;
}

/**
 * @brief Get the Height object
 * 
 * @return int 
 */
int Parking :: getHeight(){
    return m_height;
}

/**
 * @brief Set the Img object
 * 
 * @param img 
 */
void Parking :: setImg(cv::Mat img){
    m_image = img;
}

/**
 * @brief Get the Img object
 * 
 * @return cv::Mat 
 */
cv::Mat Parking :: getImg(){
    return m_image;
}

/**
 * @brief Get the Angle object
 * 
 * @return float 
 */
float Parking :: getAngle(){
    return m_angle;
}

/**
 * @brief Set the Angle object
 * 
 * @param angle 
 */
void Parking :: setAngle(float angle){
    m_angle = angle;
}

/**
 * @brief Get the Lower Left object
 * 
 * @return cv::Point 
 */
cv::Point Parking :: getLowerLeft(){
    return m_lower_left;
}

/**
 * @brief Get the Upper Right object
 * 
 * @return cv:: 
 */
cv:: Point Parking :: getUpperRight(){
    return m_upper_right;
}

/**
 * @brief Get the Lower Right object
 * 
 * @return cv::Point 
 */
cv::Point Parking :: getLowerRight(){
    return m_lower_right;
}

/**
 * @brief Get the Upper Left object
 * 
 * @return cv::Point 
 */
cv::Point Parking :: getUpperLeft(){
    return m_upper_left;
}

/**
 * @brief Get the Center object
 * 
 * @return cv::Point 
 */
cv::Point Parking :: getCenter(){
    return m_center;
}

/**
 * @brief Get the Info of Parking object
 * 
 * @return void:: 
 */
void:: Parking :: GetInfo(){
    cout << "ID: " << m_id << endl;
    cout << "isFree: " << m_status << endl;
    cout << "X: " << m_x << endl;
    cout << "Y: " << m_y << endl;
    cout << "Width: " << m_width << endl;
    cout << "Height: " << m_height << endl;
}
