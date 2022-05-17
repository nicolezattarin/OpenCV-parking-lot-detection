#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <string>
#include "parking.h"
#include "utils.h"
#include <fstream>

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
    
    // read camera csv with position of parking lots
    cout << "\nreading camera csv..." << endl;
    string parking_lots = "../CNR-EXT_FULL_IMAGE_1000x750/camera" + to_string(camera_number) + ".csv";
    vector<Parking> parkings = ReadCameraCSV(parking_lots);

    if (parkings.size() == 0){
        cout << "can't read parking lots from " << parking_lots << endl;
        return -1;}

    // DEBUG: print all parking lots
    // for (int i = 0; i < parkings.size(); i++){
    //     cout << "id: " << parkings[i].getId() 
    //         << ", x: " << parkings[i].getX() << ", y: " << parkings[i].getY() 
    //         << ", width: " << parkings[i].getWidth() << ", height: " << parkings[i].getHeight() 
    //         << endl;}
    
    // read images
    cout << "\nreading images..." << endl;
    vector<cv::Mat> images = ReadImages(camera_number, weather);
    if (images.size() == 0){
        cout << "can't read images from CNR-EXT_FULL_IMAGE_1000x750/" << weather << "/*/camera" << camera_number << "/*.jpg" << endl;
        return -1;}
    
    // DEBUG: print some images
    imshow("image", images[0]);
    waitKey(0);


    return 0;
    }