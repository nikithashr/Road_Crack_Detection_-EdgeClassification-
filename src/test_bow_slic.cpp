/*
 * test_texture_slic.cpp.
 *
 * Written by: Srivatsan Varadharajan.
 *
 * Modified by: Vinay Palakkode
 * Testing file for texture_slic function, written in texture_slic.cpp and texture_slic.hpp
 * 
 */

#include "../include/bow_slic.hpp"
#include "../include/debugging_functions.hpp"

#include <fstream>
#include <string.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <sys/types.h>
#include <unistd.h>


#define NUM_SUPER_PIXEL 2000
#define COMPACTNESS 40
#define HISTOGRAM_WEIGHT 60
#define MIN_AREA 15

using namespace std;
using namespace cv;

string type2str(int type) {
    string r;
    
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    
    switch ( depth ) {
        case CV_8U:
            r = "8U";
            break;
        case CV_8S:
            r = "8S";
            break;
        case CV_16U:
            r = "16U";
            break;
        case CV_16S:
            r = "16S";
            break;
        case CV_32S:
            r = "32S";
            break;
        case CV_32F:
            r = "32F";
            break;
        case CV_64F:
            r = "64F";
            break;
        default:
            r = "User";
            break;
    }
    r += "C";
    r += (chans+'0');
    return r;
}

int main(int argc, char *argv[]) {

    
    if (argc < 4) {
        cout << "Usage: " << argv[0] << " <0/1 - training/testing> <path to image> <path to mask image> <label for svm training>"<< endl << " Here's an example:  ./test_bow_slic train_cracks/cracked_road_cloudy 0 20130719_163110_833_004.jpg 20130719_163110_833_004_mask.png 1" << endl;
        return 0;
    }

    
    // all the image names are stored in a file by doing ls on the directory
    string path = argv[1];
//    path += argv[1];
    string image_names_file = "image_filenames.txt";
    int num_superpixels = NUM_SUPER_PIXEL;
    
    int m = COMPACTNESS;
    int histogram_distance_weight = HISTOGRAM_WEIGHT;
    int area_threshold = MIN_AREA;
//    string model = argv[2];
    int testing = atoi(argv[2]);
    string model_filename;
    
    //read image names and mask names
//    ifstream image_names(image_names_file, ios::in);
//    while (!image_names.eof()) {
        string image_name = argv[3];
    
        //image_names >> image_name;
        string image_path = path + "/" + image_name;
    string mask_name = argv[4];
        //image_names >> mask_name;
        string mask_path = path + "/" + mask_name;
    
    cout << "image #:  " << image_path << endl;
    Mat input_image = imread(image_path);
    Mat mask_image = imread(mask_path);

    resize(input_image,input_image,Size(0,0),0.5,0.5);
    resize(mask_image,mask_image,Size(0,0),0.5,0.5);

    
    //Texton map generation for each image
    Textons textons(KCENTERS, input_image);
    Mat texton_map;
    texton_map.create(input_image.rows, input_image.cols, CV_8UC1);
    textons.TextonMapGeneration(texton_map);
    
    Mat texton_map_converted;
    texton_map.convertTo(texton_map_converted, CV_16U);
    
    /* Perform the SLIC superpixel algorithm. */
    Mat superpixel_image(input_image.rows,input_image.cols,CV_8UC1);
    Mat centroids;
    
    int label_svm = atoi(argv[5]);
    BagOfWordsSlic superpixel_generator(histogram_distance_weight, area_threshold, m);
    superpixel_generator.GenerateSuperpixels(label_svm, input_image,superpixel_image,num_superpixels,texton_map_converted,mask_image,centroids);
    cout << "outside" << endl;
    
//        imshow("input_image", input_image);
//        waitKey();
    int num_superpixels_total;


    num_superpixels_total = superpixel_generator._number_of_superpixels_total;
    DisplaySuperpixelsColorized(superpixel_image,"final");
    displaySuperpixelBoundaries(input_image,superpixel_image);
    if (testing == 1) {
        system("./libsvm-3.20/svm-predict features.txt featureModel.txt output_svm.txt");
        ifstream svmOutput("output_svm.txt", ios::in);
        while (svmOutput.good() == 0) {
            
        }
        DisplaySuperpixelSVMColorized(input_image, superpixel_image, "output_svm.txt", num_superpixels_total);
               
    } 
//    destroyAllWindows();
    cout << "here??" << endl;
return 0;
}
