#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <string>
#include "../data_structures/parking.h"
#include "../utils.h"
#include <fstream>
#include <opencv2/dnn/dnn.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
    /************************************************************************/
    /*                      INITIALIZE                        */
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

    // read camera pictures
    cout << "\nreading images..." << endl;
    vector<camera_picture> images = ReadImages(camera_number, weather, rotation_flag, equalization_flag, blur_flag);

    /************************************************************************/
    /*                 READ CLASSIFICATION RESULTS                          */
    /************************************************************************/

    /** 
     * 
     * We saved the results of the classification in a file at path 
     * "../../results/CNR/camera{}/{}/classified_sample_preproc_{}.csv".format(camera_number, weather, preprocessing)
     * The file contains the following columns:
     *      - filename
     *      - class id 
     *      - class name
     *      - confidence (probability)
     * 
     * note thate preprocessing is a string which takes values: rot, blur or eq
    **/

    // read classified samples
    cout << "\nreading classified samples and setting the status of each lot..." << endl;
    ReadClassifiedSamples(images, camera_number, weather, rotation_flag, equalization_flag,blur_flag);

    // draw free lots
    cout << "\ndrawing free lots..." << endl;
    draw_free_lots(images);

    // save images with lots
    cout << "\nsaving images with lots..." << endl;
    save_images_with_lots(images, camera_number, weather, rotation_flag, equalization_flag, blur_flag);
   
    return 0;
}