#include "utils.h"

int ORIGINAL_WIDTH = 2592;
int ORIGINAL_HEIGHT = 1944;
int FINAL_WIDTH = 1000;
int FINAL_HEIGHT = 750;

double RHO_CELL_SIZE = 1;
double THETA_CELL_SIZE = CV_PI/180;


/**
 * @brief Get the average rotation of an image by identifying the main lines in 
 * the image and the corresponding angle. The idea is that theoretically each parking lot should be
 *  rotated of a spefici value, but we want to automatize the process of rotation, so we decide to
 *  lack of precision, considering the average angle of the most important lines.
 * @param img: image to be processed
 * @param lower_threshold: lower threshold for the canny edge detector
 * @param upper_threshold: upper threshold for the canny edge detector
 * @param sigma_canny: sigma for the canny edge detector
 * @param path: path of the image
 * @param threshold_HoughLines: threshold for the HoughLines
 * @param weather: weather of the image
 * @param camera_number: camera number of the image
 * @return float
 */

float get_average_rotation(cv::Mat img, int lower_threshold, int upper_threshold, int sigma_canny, 
                            string path, float threshold_HoughLines, string weather, int camera_number){
    cv::Mat img_gray = img.clone();
    cv::cvtColor(img_gray, img_gray, cv::COLOR_BGR2GRAY);
    cv::Mat img_canny;
    cv::Canny(img_gray, img_canny, lower_threshold, upper_threshold, sigma_canny);

    //save images
    // string path_canny = path.substr(path.find_last_of("/"), path.size()-4);
    // imwrite("../../results/canny_imgs_CNR/"+path_canny+"_cam"+to_string(camera_number)+"_"+weather+"_.png", img_canny);

    vector<cv::Vec2f> lines;
    HoughLines(img_canny, lines, RHO_CELL_SIZE, THETA_CELL_SIZE, threshold_HoughLines, 0, 0);
    
    // save images
    // cv::Mat img_lines = img.clone();
    // DrawLines( img_lines,lines);
    // imwrite("../../results/HoughLines_imgs_CNR/"+path_canny+"_cam"+to_string(camera_number)+"_"+weather+"_.png", img_lines);

    float avg_angle = 0;
    int n_lines = lines.size();
    if (n_lines == 0) return 0;
    for(int i = 0; i < n_lines; i++){
        float rho = lines[i][0];
        float theta = lines[i][1];
        avg_angle += theta;
        // cout << "avg_angle: " << avg_angle << " theta: " << theta << endl;
    }
    // cout << "nlines: " << n_lines << endl;
    avg_angle /= n_lines;
    return avg_angle;
}

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
        //equalize the histogram of the image
        vector<cv::Mat> hists(3);
        vector<cv::Mat> equalized_hists(3);
        equalize(image, image, hists, equalized_hists, "RGB");


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

/**
 * @brief Draw lines on the given image
 * 
 * @param img: image to draw lines on
 * @param lines: vector of lines to draw
 */
void DrawLines( cv::Mat& img, vector<cv::Vec2f>lines) {
    for( size_t i = 0; i < lines.size(); i++ ){
        float rho = lines[i][0], theta = lines[i][1];
        vector<cv::Point>CartesianLine = PolarToCartesian (rho, theta);
        cv::Point pt1 = CartesianLine[0], pt2 = CartesianLine[1];
        line(img, pt1, pt2, cv::Scalar(0,255, 0), 2);
        }
}

/**
 * @brief convert polar coordinates to cartesian coordinates
 * 
 * @param rho: distance from the origin
 * @param theta: angle from the x axis
 * @return vector<cv::Point> 
 */
vector<cv::Point> PolarToCartesian(float rho, float theta){
    cv::Point pt1, pt2;
    double ct = cos(theta), st = sin(theta); 
    double x0 = ct*rho, y0 = st*rho;
    pt1.x = cvRound(x0 + 1000*(-st));
    pt1.y = cvRound(y0 + 1000*(ct));
    pt2.x = cvRound(x0 - 1000*(-st));
    pt2.y = cvRound(y0 - 1000*(ct));
    vector<cv::Point> cartesianLine = {pt1, pt2};
    return cartesianLine;
}

/**
 * @brief Perform preprocess of each patch corresponding to a parking lot
 * 
 * @param input 
 * @param output 
 */
cv::Mat preprocess_patch(cv::Mat img, float rotation){
    //we work directly on each image and overwrite the input image

    cv::Point2f center(img.cols/2.0, img.rows/2.0);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, rotation*180./CV_PI, 1);
    cv::warpAffine(img, img, rot_mat, img.size());

    // blur image
    cv::GaussianBlur(img, img, cv::Size(5,5), 0, 0);

    // resize the image to a fixed size: in order to give it to the CNN, 
    // which is trained on a fixed size (150x150)
    cv::resize(img, img, cv::Size(150, 150));
    
    return img;
}

void preprocess (vector<camera_picture> camera_pictures){
    // run over all the camera pictures
    for (int i = 0; i < camera_pictures.size(); i++){
        vector<Parking> parkings = camera_pictures[i].getParking();
        float rotation = camera_pictures[i].get_avg_rotation();
        // run over all the patches
        for (int j = 0; j < parkings.size(); j++){
            // preprocess the patch
            cv::Mat patch = parkings[j].getImg();
            cv::Mat processed_patch = preprocess_patch(patch, rotation);
            parkings[j].setImg(processed_patch);
        }
    }
    
}

void equalize(cv::Mat& img, cv::Mat& equalized_img, 
            std::vector<cv::Mat>& hists, std::vector<cv::Mat>& equalized_hists, 
            string color_space){

        if (color_space != "RGB" && color_space != "BGR2HSV" && color_space != "BGR2Lab"){
            cout << "color_space must be RGB or HSV" << endl;
            return;}

        //histo params
        int nbins = 256;
        int nimages = 1;
        int dims = 1;
        const int * nchannels = 0;
        float r[] = {0,255};
        const int histSize[] = {nbins};
        const float* ranges[] = {r};
        std::vector<cv::Mat> channels(3);
        cv::Mat hsv;

        if (color_space == "RGB"){
            split(img, channels);// split the image into 3 channels
            // equalize each channel
            for (int i = 0; i < 3; i++) cv::equalizeHist(channels[i], channels[i]);
            merge(channels, equalized_img);// merge the equalized channels

            for (int i = 0; i < 3; i++) cv::calcHist(&channels[i], nimages, nchannels, cv::Mat(), equalized_hists[i], 
                                        dims, histSize, ranges, true, false);
        }
        else if (color_space == "BGR2Lab"){
            cv::Mat hsv;
            cv::cvtColor(img, hsv, cv::COLOR_BGR2Lab);
            cv::split(hsv, channels);
            cv::equalizeHist(channels[0], channels[0]); //equalize the luminance channel
            cv::merge(channels, equalized_img);
            cv::cvtColor(equalized_img, equalized_img, cv::COLOR_HSV2BGR); //change back to BGR
            for (int i = 0; i < 3; i++) cv::calcHist(&channels[i], nimages, nchannels, cv::Mat(), equalized_hists[i], 
                                                dims, histSize, ranges, true, false);
        }
        else{
            cv::Mat hsv;
            cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
            cv::split(hsv, channels);
            cv::equalizeHist(channels[2], channels[2]); 
            cv::merge(channels, equalized_img);
            cv::cvtColor(equalized_img, equalized_img, cv::COLOR_HSV2BGR); //change back to BGR
            for (int i = 0; i < 3; i++) cv::calcHist(&channels[i], nimages, nchannels, cv::Mat(), equalized_hists[i], 
                                                dims, histSize, ranges, true, false);
        }

        for (int i = 0; i < 3; i++) cv::calcHist(&channels[i], nimages, nchannels, cv::Mat(), hists[i], 
                    dims, histSize, ranges, true, false);
        }

