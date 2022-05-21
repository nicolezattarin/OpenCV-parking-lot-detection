#include "preprocessing.h"

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
 * @brief Preprocess the image taken by the entire camera to remove noise and equalize the histogram of each channel
 * 
 * @param src 
 * @param dst 
 */
void preprocess(cv::Mat& src, cv::Mat& dst, bool equalization, bool blur){
    if (equalization){
        //equalize the histogram of the image
        vector<cv::Mat> hists(3);
        vector<cv::Mat> equalized_hists(3);
        equalize(src, dst, hists, equalized_hists, "RGB");
    }
    else{
        dst = src.clone();
    }
    if (blur){
        // blur image
        cv::medianBlur(dst, dst, 5);
    }
    else{
        dst = src.clone();
    }
}


/**
 * @brief Perform the rotation of the image
 * 
 * @param input 
 * @param output 
 */
cv::Mat preprocess_patch(cv::Mat img, float rotation, bool rotation_flag){
    if (rotation_flag){
        cv::Point2f center(img.cols/2.0, img.rows/2.0);
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, rotation*180./CV_PI, 1);
        cv::warpAffine(img, img, rot_mat, img.size());
    }
    return img;
}

/**
 * @brief preprocess all the patches of a vector of camera pictures
 * 
 * @param camera_pictures 
 */
void preprocess_patches (vector<camera_picture> camera_pictures, bool rotation_flag){
    // run over all the camera pictures
    for (int i = 0; i < camera_pictures.size(); i++){
        vector<Parking> parkings = camera_pictures[i].getParking();
        float rotation = camera_pictures[i].get_avg_rotation();
        // run over all the patches
        for (int j = 0; j < parkings.size(); j++){
            // preprocess the patch
            cv::Mat patch = parkings[j].getImg();
            cv::Mat processed_patch = preprocess_patch(patch, rotation, rotation_flag);
            parkings[j].setImg(processed_patch);

        }
    } 
}

/**
 * @brief equalize the histogram of each channel of the image
 * 
 * @param img 
 * @param equalized_img 
 * @param hists 
 * @param equalized_hists 
 * @param color_space 
 */
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