#include "utils.h"
#include <filesystem>
namespace fs = std::filesystem;

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

vector<Parking> CNRReadCameraCSV(string filename){

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
 * @brief return a vector of parking lots from a csv file, assigns id, x, y, width, height,
 *  while image and isEmpty are initialized as default in parking.h
 * 
 * @param filename 
 * @return vector<Parking> 
 */

vector<Parking> PKlotReadCameraCSV(string filename){

    cout << "reading file " << filename << endl;
    ifstream file (filename);
    vector<Parking> parkings;

    if (!file.is_open()){
        cerr << "can't open file " << filename << endl;}

    string line;
    getline(file, line); // skip first line
    // format is 
    // id, occupied, center_x, center_y, size_w, size_h, angle, center
    // point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y, filename

    while (getline(file, line)){
        stringstream ss(line);
        string id, center_x, center_y, size_w, size_h, angle;
        string point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y, filename;
       
        getline(ss, id, ',');
        getline(ss, center_x, ',');
        getline(ss, center_y, ',');
        getline(ss, size_w, ',');
        getline(ss, size_h, ',');
        getline(ss, angle, ',');
        getline(ss, point1_x, ',');
        getline(ss, point1_y, ',');
        getline(ss, point2_x, ',');
        getline(ss, point2_y, ',');
        getline(ss, point3_x, ',');
        getline(ss, point3_y, ',');        
        getline(ss, point4_x, ',');
        getline(ss, point4_y, ','); 

        //DEBUG
        // cout << "id: " << id << endl;
        // cout << "point1_x: " << point1_x << " point1_y: " << point1_y << endl;
        // cout << "point2_x: " << point2_x << " point2_y: " << point2_y << endl;
        // cout << "point3_x: " << point3_x << " point3_y: " << point3_y << endl;
        // cout << "point4_x: " << point4_x << " point4_y: " << point4_y << endl;

        // format with lossless compression (quality 100%) 
        // in a resolution of 1280 ?? 720 pixels
        // using constructor:
        // Parking(int id, int width, int height, 
        //         cv::Point lower_left, cv::Point upper_right, cv::Point lower_right, cv::Point upper_left)
        Parking p(stoi(id), stoi(size_w), stoi(size_h), stof(angle),
                cv::Point(stoi(center_x), stoi(center_y)),
                cv::Point(stoi(point1_x), stoi(point1_y)),
                cv::Point(stoi(point2_x), stoi(point2_y)),
                cv::Point(stoi(point3_x), stoi(point3_y)),
                cv::Point(stoi(point4_x), stoi(point4_y)));
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

vector<camera_picture> GetCameraPictures(vector<string> filenames, vector<Parking> parkings_coordinates, 
                                        int camera_number, string weather,  bool rotation, bool equalization, bool blur,
                                        string dataset){
    vector<camera_picture> camera_images;
    //iterate over the images (pov of a camera)
    for (int i = 0; i < filenames.size(); i++){
        cv::Mat image = cv::imread(filenames[i]);
       // cout << "ciao " << endl;
        preprocess(image, image, equalization, blur); //equalization and blur

        cv::Mat image_parking_lots = image.clone();//original image with squares in correspondence of parking lots
        vector<Parking> parkings;
        // iterate over the parking lots, since each camera_picture is composed by a vector of parking lots
        for (int j = 0; j < parkings_coordinates.size(); j++){
            Parking p = parkings_coordinates[j];
            // set image of the parking slot in parking, i.e. fine the 
            // subimage corresponding to the parking slot in the image

            if (to_lower(dataset) == "cnr"){
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
            else if (to_lower(dataset) == "pklot"){
                cv::Point ll = p.getLowerLeft();
                cv::Point lr = p.getLowerRight();
                cv::Point ur = p.getUpperRight();
                cv::Point ul = p.getUpperLeft();
                float angle = p.getAngle();

                // DEBUG:
                // cout << "ll " << ll.x << ", " << ll.y << endl;
                // cout << "lr " << lr.x << ", " << lr.y << endl;
                // cout << "ul " << ul.x << ", " << ul.y << endl;
                // cout << "ur " << ur.x << ", " << ur.y << endl;

                cv::Range rows(min(ll.y, lr.y), max(ul.y, ur.y));
                cv::Range cols(min(ll.x, ul.x), max(lr.x, ur.x));
                cv::Mat subimg = image(rows, cols);

                // rotate
                if (rotation){
                    cv::Point2f center(subimg.cols/2., subimg.rows/2.);
                    cv::Mat rot_mat = cv::getRotationMatrix2D(center, -angle, 1);
                    cv::warpAffine(subimg, subimg, rot_mat, cv::Size(p.getHeight(), p.getWidth()));
                }
                
                cv::Mat slot_image = subimg.clone();

                vector<cv::Point> points = {ll, lr, ur, ul};
                //draw rectangle
                for (int i = 0; i < 3; i++)
                    line(image_parking_lots, points[i], points[i+1], cv::Scalar(0,255,0), 2);
                
                // create object parking and push
                parkings.push_back(Parking(p.getId(), p.getWidth(), p.getHeight(), slot_image, 
                                        p.getAngle(), p.getCenter(), ll, lr, ur, ul));
            }

        }
        string date, time;
        if (to_lower(dataset) == "cnr"){
            // create object camera_images and push
            date = filenames[i].substr(filenames[i].find_last_of("/")+1, 10);
            time = filenames[i].substr(filenames[i].find_last_of("_")+1, 4);
        }
        else if(to_lower(dataset) == "pklot"){
            // create object camera_images and push
            date = filenames[i].substr(filenames[i].find_last_of("/")+1, 10);
            time = filenames[i].substr(filenames[i].find_last_of("_")-5, 8);
        }
        camera_images.push_back(camera_picture(parkings, image, image_parking_lots, filenames[i], date, time));
    }

    if (rotation){
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
    }
    return camera_images;
}


/**
 * @brief read all images from CNR folder and return a vector of images
 * 
 * @param camera_number 
 * @param weather 
 * @return vector<Mat> 
 */

vector<camera_picture> ReadImages(int camera_number, string weather, bool rotation, bool equalization, bool blur, string dataset, string nimgs){
    
    if (dataset == "cnr"){
        if (camera_number < 1 || camera_number > 9){
            cerr << "camera number should be from 1 to 9" << endl;}
        if (weather != "rainy" && weather != "sunny" && weather != "overcast" && weather != "all"){
            cerr << "weather should be 'rainy', 'sunny', 'overcast' or 'all'" << endl;}
    }
    else if (dataset == "pklot"){
        if (camera_number < 1 || camera_number > 3){
            cerr << "camera number should be from 1 to 3" << endl;}
        if (weather != "rainy" && weather != "sunny" && weather != "cloudy" && weather != "all"){
            cerr << "weather should be 'rainy', 'sunny', 'cloudy' or 'all'" << endl;}
    }
    /* READ PARKING LOTS INFORMATIONS */

    // read camera csv with position of parking lots: this is generic,once we choose the camera it hold in eahc condition 
    cout << "\nreading camera csv..." << endl;
    vector<Parking> parkings_coordinates;
    string parking_lots;
    if (dataset == "cnr"){
        string parking_lots = "../../CNR-EXT_FULL_IMAGE_1000x750/camera" + to_string(camera_number) + ".csv";
        parkings_coordinates = CNRReadCameraCSV(parking_lots);
    }
    else if (dataset == "pklot"){
        string parking_lots = "../../PKLot_reduced/camera"+to_string(camera_number)+".csv";
        parkings_coordinates = PKlotReadCameraCSV(parking_lots);
    }

    
    if (parkings_coordinates.size() == 0){
        cerr << "can't read parking lots from " << parking_lots << endl;
    }

    /* READ IMAGES */
    // read camera csv with position of parking lots for a specific weather
    string base_dir;
    string pattern;
    if (weather != "all"){
        if (dataset == "cnr"){
            base_dir = 
            "/Users/nicolez/Documents/GitHub/OpenCV-parking-lot-detection/CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/" + to_upper(weather);
            pattern = "/*/camera" + to_string(camera_number)+ "/*.jpg";
        }
        else if (dataset == "pklot"){
            base_dir = "/Users/nicolez/Documents/GitHub/OpenCV-parking-lot-detection/PKLot_reduced";
            pattern = "/camera" + to_string(camera_number)+ "/" +weather+ "/*.jpg";
        }
        string path = base_dir + pattern;
        vector<string> filenames = glob_path(base_dir + pattern);
        if (nimgs != "all"){
            filenames.resize(stoi(nimgs));
        }
        

        //define a vector of camera pictures:
        vector<camera_picture> camera_images = GetCameraPictures(filenames, parkings_coordinates, camera_number, weather, rotation, equalization, blur, dataset);
        return camera_images;
    }
    else {
        if (dataset == "cnr"){
            base_dir = 
            "/Users/nicolez/Documents/GitHub/OpenCV-parking-lot-detection/CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/*";
            pattern = "/*/camera" + to_string(camera_number)+ "/*.jpg";
        }
        else if (dataset == "pklot"){
            base_dir = "/Users/nicolez/Documents/GitHub/OpenCV-parking-lot-detection/PKLot_reduced";
            pattern = "/camera" + to_string(camera_number)+ "/*/*.jpg";
        }
        string path = base_dir + pattern;
        vector<string> filenames = glob_path(base_dir + pattern);
        //define a vector of camera pictures:
        vector<camera_picture> camera_images = GetCameraPictures(filenames, parkings_coordinates, camera_number, weather, rotation, equalization, blur, dataset);
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
 * @brief returns a char converted to lowe case
 * 
 * @param c 
 * @return char 
 */
char to_lower_char (char c){return tolower(c);}

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
 * @brief  return a the string converted to lower case
 * 
 * @param s 
 * @return string 
 */
string to_lower (string s){
    transform(s.begin(), s.end(), s.begin(), to_lower_char);
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

/**
 * @brief save patches in a folder
 * 
 * @param camera_images 
 * @param weather 
 * @param camera_number 
 * @param rotation 
 * @param equalization 
 * @param blur 
 * @param dataset 
 */
void save_patches(vector<camera_picture> camera_images, string weather, int camera_number,
                bool rotation, bool equalization, bool blur, string dataset){
    // run over all the images of the camera, considering thatpatches from the datsset are loaded as:

    // CNR-EXT-Patches-150x150/PATCHES/<WEATHER>/<CAPTURE_DATE>/camera<CAM_ID>/<W_ID>_<CAPTURE_DATE>_<CAPTURE_TIME>_C0<CAM_ID>_<SLOT_ID>.jpg
    // we save the images in the same format, but in a different folder
    // in particular in PATCHES_PROCESSED
    string weather_id (1,to_upper(weather)[0]);

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

            string dir = "../../"+to_upper(dataset)+"_PATCHES_PROCESSED/"+to_upper (weather)+"/camera"+
                        to_string(camera_number)+"_"+to_string(rotation)+to_string(equalization)+to_string(blur);
            
            if (!fs::exists(dir)){
                fs::create_directories(dir);
            }
            if (dataset == "cnr"){
                string filename = weather_id+"_"+date+"_"+time[0]+time[1]+"."+time[2]+time[3]+"_C0"+to_string(camera_number)+"_"+to_string(parking_id)+".jpg";
                cv::imwrite (dir+"/"+filename, patch);
            }
            else if (dataset == "pklot"){
                //date_time#id.jpg
                //format for id string
                std::stringstream ss;
                ss << std::setw(3) << std::setfill('0') << parking_id;
                std::string parking_id_str = ss.str();
                string filename = date+"_"+time+"#"+parking_id_str+".jpg";
                cv::imwrite (dir+"/"+filename, patch);
            } 
        }
    }
}

/**
 * @brief goes through a vector of camera_picture and apply the overloaded function on each element
 * 
 * @param images 
 * @param camera_number 
 * @param weather 
 * @param rotation_flag 
 * @param equalization_flag 
 * @param blur_flag 
 */
void ReadClassifiedSamples(vector<camera_picture>& images, int camera_number, 
                                        string weather, bool rotation_flag, 
                                        bool equalization_flag, bool blur_flag, string dataset)
{
    for (int i=0; i<images.size(); i++){
        ReadClassifiedSamples(images[i], camera_number, weather, rotation_flag, equalization_flag, blur_flag, dataset);
    }
}

/**
 * @brief given a camera picture, goes through all the pathes and updates the status of the 
 *        parking according to the classification already performed
 * 
 * @param images 
 * @param camera_number 
 * @param weather 
 * @param rotation_flag 
 * @param equalization_flag 
 * @param blur_flag 
 */

void ReadClassifiedSamples(camera_picture& images, int camera_number, 
                                        string weather, bool rotation_flag, 
                                        bool equalization_flag, bool blur_flag, string dataset)
{
    // read file with classfied samples
    // we now support only one odf the three preprocessing, since these turned out to be not very effective
    // a reinfed version of the code could generalize this
    if ((rotation_flag && equalization_flag) || (rotation_flag && blur_flag) || (equalization_flag && blur_flag)){
        cerr << "Error: only one of the three preprocessing can be applied" << endl;
    }
    string preprocessing;
    if (rotation_flag){
        preprocessing = "rot";
    }
    else if (equalization_flag){
        preprocessing = "eq";
    }
    else if (blur_flag){
        preprocessing = "blur";
    }
    else{
        preprocessing = "none";
    }
    
    string path = "../../results/"+to_upper(dataset)+"/camera"+to_string(camera_number)+"/"+to_lower(weather)+
                    "/classified_sample_preproc_"+preprocessing+".csv";

    /*
     *  we have to go through all the parking slots of the given camera_picture, recall that the 
     * file name reported in the file, i.e. the file name of each path
    */
    string date = images.get_capture_date();
    string time = images.get_capture_time();
    string weather_id (1,to_upper(weather)[0]);

    vector<Parking> parkings = images.getParking();

    for (int i = 0; i<parkings.size(); i++){
        Parking p = parkings[i];
        int parking_id = p.getId();

        bool is_busy = 0;
        if (dataset == "cnr"){
            string filename = weather_id +"_"+date+"_"+time[0]+time[1]+"."+time[2]+time[3]+
                        "_C0"+to_string(camera_number)+"_"+to_string(parking_id)+".jpg";
            is_busy = check_if_free(path, filename);
        }
        else if (dataset == "pklot"){
            //format 2013-03-05_14_55_10#016.jpg	
            std::stringstream ss;
            ss << std::setw(3) << std::setfill('0') << parking_id;
            std::string parking_id_str = ss.str();
            string filename = date+"_"+time+"#"+parking_id_str+".jpg";
            is_busy = check_if_free(path, filename);
        }
        parkings[i].setStatus(is_busy);
        // parkings[i].GetInfo();
    }
    images.setParking(parkings);
}

/**
 * @brief goes through the file containing the results of classfication and checks if the given path is occupied or not
 * 
 * @param path path to file containing results eg "classified_sample_preproc_none.csv"
 * @param filename filename contained in the file of results eg "S_2016-01-12_10.47_C01_264.jpg"
 * @return true 
 * @return false 
 */
bool check_if_free(string path, string filename){
    ifstream file(path);
    // check if file is open
    if (!file.is_open()){
        cerr << "Error: file " << path << " not found" << endl;
        return false;
    }

    string line;
    // skip the header
    getline(file, line);
    while (getline(file, line)){
        // find the line corresponding to that patch
        stringstream ss(line);
        // first column contains the filename
        string file;
        getline(ss, file, ',');
        if (file == filename){
            // second column contains the label
            string label;
            getline(ss, label, ',');
            if (label == "0"){ //label 0 means that the patch is free
                //DEBUG
                // cout << "patch " << filename << " is free" << endl;
                return true;
            }
            else{
                //DEBUG
                // cout << "patch " << filename << " is occupied" << endl;
                return false; //label 1 means that the patch is occupied
            }
        }
    }
    cerr << "Error: the patch " << filename << " was not found in the file " << path << endl;
}

/**
 * @brief Draw free and busy slots with different colors on a vector of camera images
 * 
 * @param image 
 */
void draw_free_lots(vector<camera_picture>& image, string dataset){
    // draw the free lots in each camera_picture
    for (int i=0; i<image.size(); i++){
        draw_free_lots(image[i], dataset);
    }
}

/**
 * @brief Draw free and busy slots with different colors on a camera image
 * 
 * @param image 
 */
void draw_free_lots(camera_picture& image, string dataset){
    cv::Mat img = image.getImg();
    vector<Parking> parkings = image.getParking();
    
    cv::Mat img_with_slots = img.clone();

    for (int i=0; i<parkings.size(); i++){
        Parking p = parkings[i];
        if (dataset == "cnr"){
            if (p.getStatus()){
                // set image of the parking slot in parking, i.e. fine the 
                // subimage corresponding to the parking slot in the image
                cv::rectangle(img_with_slots, cv::Point(p.getX(), p.getY()), 
                        cv::Point(p.getX() + p.getWidth(), p.getY() + p.getHeight()), 
                        cv::Scalar(0, 255, 0), 3);
            }
            else{
                cv::rectangle(img_with_slots, cv::Point(p.getX(), p.getY()), 
                        cv::Point(p.getX() + p.getWidth(), p.getY() + p.getHeight()), 
                        cv::Scalar(0, 0, 255), 3);
            }
        }
        else if (dataset == "pklot"){
            cv::Point p1 = p.getLowerLeft();
            cv::Point p2 = p.getLowerRight();
            cv::Point p3 = p.getUpperRight();
            cv::Point p4 = p.getUpperLeft();
            vector<cv::Point> points = {p1, p2, p3, p4};
            if (p.getStatus()){
                //draw rectangle
                for (int i = 0; i < 3; i++){
                    line(img_with_slots, points[i], points[i+1], cv::Scalar(0,255,0), 2);
                }
                line(img_with_slots, points[3], points[0], cv::Scalar(0,255,0), 2);
            }   
            else{
                for (int i = 0; i < 3; i++){
                    line(img_with_slots, points[i], points[i+1], cv::Scalar(0,0,255), 2);
                }
                line(img_with_slots, points[3], points[0], cv::Scalar(0,0,255), 2);
            }

        }
        
    }
    //set the image of the parking slot in the image
    image.setImgParkingLots(img_with_slots);
}

/**
 * @brief save the image of the parking lots in the camera_picture
 * 
 * @param images 
 * @param camera_number 
 * @param weather 
 */
void save_images_with_lots(vector<camera_picture>& images, int camera_number, string weather,
                            bool rotation_flag, bool equalization_flag, bool blur_flag, string dataset){
    // read file with classfied samples
    // we now support only one odf the three preprocessing, since these turned out to be not very effective
    // a reinfed version of the code could generalize this
    if ((rotation_flag && equalization_flag) || (rotation_flag && blur_flag) || (equalization_flag && blur_flag)){
        cerr << "Error: only one of the three preprocessing can be applied" << endl;
    }
    string preprocessing;
    if (rotation_flag){
        preprocessing = "rot";
    }
    else if (equalization_flag){
        preprocessing = "eq";
    }
    else if (blur_flag){
        preprocessing = "blur";
    }
    else{
        preprocessing = "none";
    }
    cout << "Reading classified samples from " << preprocessing << " preprocessing" << endl;

    for (int i=0; i<images.size(); i++){
        string path = "../../results/"+to_upper(dataset)+"/camera"+to_string(camera_number)+"/detected_"+weather+"_"+preprocessing+"/";
        if (!fs::exists(path)){
            fs::create_directories(path);
        }

        string filename = images[i].get_capture_date()+"_"+images[i].get_capture_time()+".jpg";
        cv::imwrite(path+filename, images[i].getImgParkingLots());
    }
}