#include "utils.h"

/**
 * @brief return a vector of parking lots from a csv file, assigns id, x, y, width, height,
 *  while image and isEmpty are initialized as default in parking.h
 * 
 * @param filename 
 * @return vector<Parking> 
 */

vector<Parking> ReadCameraCSV(string filename){
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

        Parking p(stoi(id), stoi(x), stoi(y), stoi(width), stoi(height));
        parkings.push_back(p);
    }
    return parkings;
}

/**
 * @brief read all images from CNR folder and return a vector of images
 * 
 * @param camera_number 
 * @param weather 
 * @return vector<Mat> 
 */

vector<cv::Mat> ReadImages(int camera_number, string weather){
    if (camera_number < 1 || camera_number > 9){
        cerr << "camera number should be from 1 to 9" << endl;}
    if (weather != "rainy" && weather != "sunny" && weather != "overcast" && weather != "all"){
        cerr << "weather should be 'rainy', 'sunny', 'overcast' or 'all'" << endl;}

    // read camera csv with position of parking lots for a specific weather
    if (weather != "all"){
        const string base_dir = "/Users/nicolez/Documents/GitHub/OpenCV-parking-lot-detection/CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/" + to_upper(weather);
        const string pattern = "/*/camera" + to_string(camera_number)+ "/*.jpg";
        string path = base_dir + pattern;
        vector<string> filenames = glob_path(base_dir + pattern);

        // DEBUG: 
        // cout << "filenames: " << filenames.size() << endl;

        vector<cv::Mat> images;
        for (int i = 0; i < filenames.size(); i++){
            cv::Mat image = cv::imread(filenames[i]);
            images.push_back(image);
        }
        return images;
    }
    else {
        const string base_dir = "/Users/nicolez/Documents/GitHub/OpenCV-parking-lot-detection/CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/*";
        const string pattern = "/*/camera" + to_string(camera_number)+ "/*.jpg";
        string path = base_dir + pattern;
        vector<string> filenames = glob_path(base_dir + pattern);

        // DEBUG: 
        cout << "filenames: " << filenames.size() << endl;

        vector<cv::Mat> images;
        for (int i = 0; i < filenames.size(); i++){
            cv::Mat image = cv::imread(filenames[i]);
            images.push_back(image);
        }
        return images;
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
 * @brief Glob management to find matching paths, taken from https://stackoverflow.com/questions/8401777/simple-glob-in-c-on-unix-system
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

    // cleanup
    globfree(&glob_result);

    // done
    return filenames;
}