#include "utils.h"

int ORIGINAL_WIDTH = 2592;
int ORIGINAL_HEIGHT = 1944;
int FINAL_WIDTH = 1000;
int FINAL_HEIGHT = 750;


/**
 * @brief return a vector of parking lots from a csv file, assigns id, x, y, width, height,
 *  while image and isEmpty are initialized as default in parking.h
 * 
 * @param filename 
 * @return vector<Parking> 
 */

vector<Parking> ReadCameraCSV(string filename){

    // Pixel coordinates of the bouding boxes refer to the 2592x1944 version of 
    // the image and need to be rescaled to match the 1000x750 version.

    cout << "reading file " << filename << endl;
    ifstream file (filename);
    vector<Parking> parkings;

    if (!file.is_open()){
        cerr << "can't open file " << filename << endl;}

    string line;
    getline(file, line); // skip first line
    while (getline(file, line)){
        stringstream ss(line);
        string id;
        string x;
        string y;
        string width;
        string height;
        getline(ss, id, ',');
        getline(ss, x, ',');
        getline(ss, y, ',');
        getline(ss, width, ',');
        getline(ss, height);
        //rescaling
        int xx = stof(x)/ORIGINAL_WIDTH * FINAL_WIDTH;
        int yy = stof(y)/ORIGINAL_HEIGHT * FINAL_HEIGHT;
        int wwidth = stof(width)/ORIGINAL_WIDTH * FINAL_WIDTH;
        int hheight = stof(height)/ORIGINAL_HEIGHT * FINAL_HEIGHT;
        Parking p(stoi(id), xx, yy, wwidth, hheight);
        parkings.push_back(p);
    }
    return parkings;
}

/**
 * @brief Get the Camera Pictures object
 * 
 * @param filenames 
 * @param parkings_coordinates 
 * @return vector<camera_picture> 
 */

vector<camera_picture> GetCameraPictures(vector<string> filenames, 
                                        vector<Parking> parkings_coordinates, 
                                        int camera_number, string weather){
    vector<camera_picture> camera_images;
    //iterate over the images (pov of a camera)
    for (int i = 0; i < filenames.size(); i++){
        cv::Mat image = cv::imread(filenames[i]);

        preprocess(image, image); //equalization and blur

        cv::Mat image_parking_lots = image.clone();//original image with squares in correspondence of parking lots
        vector<Parking> parkings;
        // iterate over the parking lots, since each camera_picture is composed by a vector of parking lots
        for (int j = 0; j < parkings_coordinates.size(); j++){
            Parking p = parkings_coordinates[j];
            // set image of the parking slot in parking, i.e. fine the 
            // subimage corresponding to the parking slot in the image
            cv::Range rows(p.getY(), p.getY() + p.getHeight());
            cv::Range cols(p.getX(), p.getX() + p.getWidth());
            cv::Mat slot_image = image.clone();
            slot_image = slot_image(rows, cols);

            cv::rectangle(image_parking_lots, cv::Point(p.getX(), p.getY()), 
                        cv::Point(p.getX() + p.getWidth(), p.getY() + p.getHeight()), 
                        cv::Scalar(0, 255, 0), 2);
            // create object parking and push
            parkings.push_back(Parking(p.getId(), p.getX(), p.getY(), p.getWidth(), p.getHeight(), slot_image));
        }
        // create object camera_images and push
        string date = filenames[i].substr(filenames[i].find_last_of("/")+1, 10);
        string time = filenames[i].substr(filenames[i].find_last_of("_")+1, 4);
        camera_images.push_back(camera_picture(parkings, image, image_parking_lots, filenames[i], date, time));
    }

    // detect average angle of the main lines in the image, really not the most efficient way to do it, 
    // but we leave optimization of this part for a second moment
    float avg_angle = 0;
    for (int i = 0; i < camera_images.size(); i++){
        avg_angle+= get_average_rotation(camera_images[i].getImg(), 100, 200, 3, filenames[i], 300, weather, camera_number);
    }
    // cout << "average angle: " << avg_angle << endl;

    avg_angle /= camera_images.size();
    for (int i = 0; i < camera_images.size(); i++){
        camera_images[i].set_avg_rotation(-avg_angle); // rotation shouuld be the opposite of the average angle
    }
    cout << "average angle: " << avg_angle << endl;
    return camera_images;
}

/**
 * @brief read all images from CNR folder and return a vector of images
 * 
 * @param camera_number 
 * @param weather 
 * @return vector<Mat> 
 */

vector<camera_picture> ReadImages(int camera_number, string weather){
    if (camera_number < 1 || camera_number > 9){
        cerr << "camera number should be from 1 to 9" << endl;}
    if (weather != "rainy" && weather != "sunny" && weather != "overcast" && weather != "all"){
        cerr << "weather should be 'rainy', 'sunny', 'overcast' or 'all'" << endl;}

    /* READ PARKING LOTS INFORMATIONS */

    // read camera csv with position of parking lots: this is generic,once we choose the camera it hold in eahc condition 
    cout << "\nreading camera csv..." << endl;
    string parking_lots = "../../CNR-EXT_FULL_IMAGE_1000x750/camera" + to_string(camera_number) + ".csv";
    vector<Parking> parkings_coordinates = ReadCameraCSV(parking_lots);

    if (parkings_coordinates.size() == 0){
        cerr << "can't read parking lots from " << parking_lots << endl;}

    /* READ IMAGES */
    // read camera csv with position of parking lots for a specific weather
    if (weather != "all"){
        const string base_dir = 
        "/Users/nicolez/Documents/GitHub/OpenCV-parking-lot-detection/CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/" + to_upper(weather);
        const string pattern = 
        "/*/camera" + to_string(camera_number)+ "/*.jpg";

        string path = base_dir + pattern;
        vector<string> filenames = glob_path(base_dir + pattern);

        //define a vector of camera pictures:
        vector<camera_picture> camera_images = GetCameraPictures(filenames, parkings_coordinates, camera_number, weather);
        return camera_images;
    }
    else {
        const string base_dir = 
        "/Users/nicolez/Documents/GitHub/OpenCV-parking-lot-detection/CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/*";
        const string pattern = 
        "/*/camera" + to_string(camera_number)+ "/*.jpg";

        string path = base_dir + pattern;
        vector<string> filenames = glob_path(base_dir + pattern);
        //define a vector of camera pictures:
        vector<camera_picture> camera_images = GetCameraPictures(filenames, parkings_coordinates, camera_number, weather);
        return camera_images;
    }
  }

/**
 * @brief returns a char converted to upper case
 * 
 * @param c 
 * @return char 
 */

char to_upper_char (char c){return toupper(c);}

/**
 * @brief  return a the string converted to upper case
 * 
 * @param s 
 * @return string 
 */
string to_upper (string s){
    transform(s.begin(), s.end(), s.begin(), to_upper_char);
    return s;
}

/**
 * @brief Glob management to find matching paths, taken from
 *  https://stackoverflow.com/questions/8401777/simple-glob-in-c-on-unix-system
 * 
 * @param pattern 
 * @return std::vector<std::string> 
 */
vector<string> glob_path(const string& pattern) {

    // glob struct resides on the stack
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    // do the glob operation
    int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if(return_value != 0) {
        globfree(&glob_result);
        stringstream ss;
        ss << "glob() failed with return_value " << return_value << endl;
        throw std::runtime_error(ss.str());
    }

    // collect all the filenames into a std::list<std::string>
    vector<string> filenames;
    for(size_t i = 0; i < glob_result.gl_pathc; ++i) {
        filenames.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return filenames;
}

void save_patches(vector<camera_picture> camera_images, string weather, int camera_number){
    // run over all the images of the camera, considering thatpatches from the datsset are loaded as:

    // CNR-EXT-Patches-150x150/PATCHES/<WEATHER>/<CAPTURE_DATE>/camera<CAM_ID>/<W_ID>_<CAPTURE_DATE>_<CAPTURE_TIME>_C0<CAM_ID>_<SLOT_ID>.jpg
    // we save the images in the same format, but in a different folder
    // in particular in PATCHES_PROCESSED
    char weather_id = to_upper(weather)[0];

    for (int i = 0; i < camera_images.size(); i++){
        camera_picture c = camera_images[i];
        vector<Parking> parkings = c.getParking();
        string date = c.get_capture_date();
        string time = c.get_capture_time();

        for (int j = 0; j<parkings.size(); j++){
            Parking p = parkings[j];
            cv::Mat patch = p.getImg();
            int parking_id = p.getId();

            cv::resize(patch, patch, cv::Size(150, 150)); //the CNN requires patches of 150x150
            imwrite ("../../PATCHES_PROCESSED/"+to_upper (weather)+"/camera"+to_string(camera_number)+"/"+weather_id+"_"+date+"_"+time+"_C0"+to_string(camera_number)+"_"+to_string(parking_id)+".jpg", patch);
        }
    }
}
