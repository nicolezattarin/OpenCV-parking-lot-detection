#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <string>
#include "parking.h"
#include "utils.h"
#include <fstream>
#include <opencv2/dnn/dnn.hpp>

using namespace std;
using namespace cv;
/**
 * @brief 
 * 
 * @param argc
 * @param argv camera number, from 1 to 9, and weater, weather can be 'rainy', 'sunny', 'overcast' or 'all'
 * @return int 
 **/

int main(int argc, char** argv){
    /************************************************************************/
    /*                      READ DATA AND INITIALIZE                        */
    /************************************************************************/

    if (argc != 3){
        cout << "usage: ./main <camera number> <weather>" << endl;
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
    
    // read images: each image is an object camera_picture, 
    // with all the informations regarding each parking lot (vector of parking), the global camera image,
    // the global camera with parkings individualized and the path, in a format <W_ID>_<CAPTURE_DATE>_<CAPTURE_TIME>_C0<CAM_ID>_<SLOT_ID>.jpg
    cout << "\nreading images..." << endl;
    vector<camera_picture> images = ReadImages(camera_number, weather);
    
    // DEBUG: print and save first image information
    // camera_picture first_image = images[0];
    // imwrite("test_img/first_image.jpg", first_image.getImg());
    // imwrite("test_img/first_image_lots.jpg", first_image.getImgParkingLots());
    // vector<Parking> parkings = first_image.getParking();
    // for (int i=0; i<parkings.size(); i++){
    //     imwrite("test_img/first_image_p"+to_string(parkings[i].getId())+".jpg", parkings[i].getImg());
    // }

    // load model trained in tensorflow
    ClassifyParkings("CNN_model/15_epochs_32_batch_classifier/saved_model_graph.pb", images);
    //TO DO: LINE DETECTION->GET ANGLE OF MAIN ->ROTATE IMG

    return 0;
    }