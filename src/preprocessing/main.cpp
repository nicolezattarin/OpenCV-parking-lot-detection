#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <string>
#include "../data_structures/parking.h"
#include "utils.h"
#include <fstream>
#include <opencv2/dnn/dnn.hpp>

using namespace std;
using namespace cv;
/**
 * @brief 
 * 
 * @param argc
 * @param argv camera number, from 1 to 9, and weater, weather can be 'rainy', 'sunny', 'overcast' or 'all' and flags to discriminate which steps of preprocessing to perform
 * @return int 
 **/

int main(int argc, char** argv){
    /************************************************************************/
    /*                      READ DATA AND INITIALIZE                        */
    /************************************************************************/

    if (argc != 6){
        cout << "usage: ./main <camera number> <weather> <rotation> <equalization> <blur>" << endl;
        return -1;}

    // read camera number
    int camera_number = atoi(argv[1]);
    if (camera_number < 1 || camera_number > 9){
        cout << "camera number should be from 1 to 9" << endl;
        return -1;}

    // read weather
    string weather = argv[2];
    if (weather != "rainy" && weather != "sunny" && weather != "overcast" && weather != "all"){
        cout << "weather should be 'rainy', 'sunny', 'overcast' or 'all'" << endl;
        return -1;}

    // read parameters for preprocessing
    string rotation = argv[3];
    if (rotation != "0" && rotation != "1"){
        cout << "rotation should be either '0' or '1'" << endl;
        return -1;}
    bool rotation_flag = stoi(rotation);

    string equalization = argv[4];
    if (equalization != "0" && equalization != "1"){
        cout << "equalization should be either '0' or '1'" << endl;
        return -1;}
    bool equalization_flag =  stoi(equalization);

    string blur = argv[5];
    if (blur != "0" && blur != "1"){
        cout << "blur should be either '0' or '1'" << endl;
        return -1;}
    bool blur_flag =  stoi(blur);

    cout << "camera number: " << camera_number << endl;
    cout << "weather: " << weather << endl;
    cout << "rotation: " << rotation_flag << endl;
    cout << "equalization: " << equalization_flag << endl;
    cout << "blur: " << blur_flag << endl;
    
    // read images: each image is an object camera_picture, 
    // with all the informations regarding each parking lot (vector of parking), the global camera image,
    // the global camera with parkings individualized and the path, in a format <W_ID>_<CAPTURE_DATE>_<CAPTURE_TIME>_C0<CAM_ID>_<SLOT_ID>.jpg

    // now we have a vector with all the characteristics of the images, ready to process the patches
    cout << "\nreading images..." << endl;
    vector<camera_picture> images = ReadImages(camera_number, weather, rotation_flag, equalization_flag, blur_flag);

    /************************************************************************/
    /*                            PROCESSING                                */
    /************************************************************************/

    cout << "\npreprocessing images..." << endl;
    preprocess_patches(images, rotation_flag);

    /************************************************************************/
    /*                            SAVING                                    */
    /************************************************************************/

    cout << "\nsaving images..." << endl;
    save_patches(images, weather, camera_number, rotation_flag, equalization_flag, blur_flag);

    // DEBUG: print and save first image information
    // camera_picture first_image = images[0];
    // imwrite("test_img/first_image.jpg", first_image.getImg());
    // imwrite("test_img/first_image_lots.jpg", first_image.getImgParkingLots());
    // vector<Parking> parkings = first_image.getParking();
    // for (int i=0; i<parkings.size(); i++){
    //     cv::Mat patch = parkings[i].getImg();
    //     cv::resize(patch, patch, cv::Size(150, 150));
    //     imwrite("test_img/first_image_p"+to_string(parkings[i].getId())+".jpg", patch);
    // }

    return 0;
    }