//
// Version 0: Srivatsan Varadhrajan
// Version 1: Vinay Palakkode
//
//
//
//


#include "../include/oversegmentation.hpp"

/*------------------------------------------------------------*/
/*---------------------   ImageSegment   ---------------------*/
/*------------------------------------------------------------*/

/*-----------  Constructor - 1  -----------*/
ImageSegment::ImageSegment(int _label, Point _centroid){
    label_ = _label;
    centroid_ = _centroid;
    area_ = 1;
}

/*-----------  Constructor - 2  -----------*/
ImageSegment::ImageSegment(int _label){
    area_ = 0;
    label_ = _label;
    centroid_ = Point(-1,-1);
}

void ImageSegment::AddPixel(Point pt){
    pixel_list_.push_back(pt);
}
void ImageSegment::CalculateHistogram(Mat HSV_image) {
    // hue range: 0-179, saturation: 0-255, value: 0-255
    int RangeH = 180;
    int RangeS = 256;

    
    for (int i = 0; i < pixel_list_.size(); i++) {
        Vec3b intensity = HSV_image.at<Vec3b>(pixel_list_[i].y, pixel_list_[i].x);
        int indexH = RangeH/binsH;
        indexH = (int)intensity.val[0]/indexH;
        int indexS = RangeS/binsS;
        indexS = (int)intensity.val[1]/indexS;
        HHistogram[indexH]++;
        SHistogram[indexS]++;
    }
    for (int j = 0; j < binsH; j++) {
        HHistogram[j] = HHistogram[j]/pixel_list_.size();
    }
    for (int j = 0; j < binsS; j++) {
        SHistogram[j] = SHistogram[j]/pixel_list_.size();
    }
//    cout << "SIZE: " << pixel_list_.size() << endl;
//    cout << HHistogram[0] << ", " << HHistogram[1] << ", " << HHistogram[2] << ", " << HHistogram[3] << ", " << HHistogram[4] << endl;
//    cout << SHistogram[0] << ", " << SHistogram[1] << ", " << SHistogram[2] << endl;

    return;
}
void ImageSegment::CalculateTextonHistogram(Mat Texton_image) {
    
    int RangeTextons = 50;
    int binsTextons = 50;
    for (int i = 0; i < NUMTEXTONS; i++) {
        THistogram[i] = 0;
    }
    for (int i = 0; i < pixel_list_.size(); i++) {
        int index = (int)Texton_image.at<uchar>(pixel_list_[i].y, pixel_list_[i].x);
        THistogram[index]++;
    }
    
    for (int j = 0; j < NUMTEXTONS; j++) {
        THistogram[j] = THistogram[j]/pixel_list_.size();
    }
    //    cout << "SIZE: " << pixel_list_.size() << endl;
    //    cout << HHistogram[0] << ", " << HHistogram[1] << ", " << HHistogram[2] << ", " << HHistogram[3] << ", " << HHistogram[4] << endl;
    //    cout << SHistogram[0] << ", " << SHistogram[1] << ", " << SHistogram[2] << endl;
    
    return;
    
}
void ImageSegment::ComputeLabel() {
    float sumBlack = 0;
//    float sumWhite = 0;
    
    for( int i = 0; i < pixel_list_.size(); i++) {
        int value = (int)_edges.at<uchar>(pixel_list_[i].y, pixel_list_[i].x);
        if (value == 0)
            sumBlack++;
//        else
//            sumWhite++;
    }
//    sumWhite /= pixel_list_.size();
    sumBlack /= pixel_list_.size();
    
    if (sumBlack < 0.92)
        crack_label = 1;
    else
        crack_label = -1;

}
void ImageSegment::ComputeFeatures() {
    
    /*------- mean RGB ---------*/
    float sumR = 0;
    float sumB = 0;
    float sumG = 0;
    for (int i = 0; i < pixel_list_.size(); i++) {
        Vec3b intensity = _original_image.at<Vec3b>(pixel_list_[i].y, pixel_list_[i].x);
        sumB += (int)intensity.val[0];
        sumG += (int)intensity.val[1];
        sumR += (int)intensity.val[2];
    }
    meanBGR[0] = sumB/pixel_list_.size();
    meanBGR[1] = sumG/pixel_list_.size();
    meanBGR[2] = sumR/pixel_list_.size();
    
    /*-------- mean HSV ----------*/
    Mat HSV_image;
    cvtColor(_original_image, HSV_image, CV_BGR2HSV);
    
//    imshow("hsv", HSV_image);
//    waitKey();
    
    Mat MatHSVMeanBGR;
    MatHSVMeanBGR.create(1,1,CV_8UC3);
    MatHSVMeanBGR.at<Vec3b>(0,0)[0] = meanBGR[0];
    MatHSVMeanBGR.at<Vec3b>(0,0)[1] = meanBGR[1];
    MatHSVMeanBGR.at<Vec3b>(0,0)[2] = meanBGR[2];
    
    cvtColor(MatHSVMeanBGR, MatHSVMeanBGR, CV_BGR2HSV);
    
    
    HSVmeanBGR[0] = MatHSVMeanBGR.at<Vec3b>(0,0)[0];
    HSVmeanBGR[1] = MatHSVMeanBGR.at<Vec3b>(0,0)[1];
    HSVmeanBGR[2] = MatHSVMeanBGR.at<Vec3b>(0,0)[2];
    
    
    /*--------- Normalizing mean and hsv mean ----------*/
    meanBGR[0] = meanBGR[0]/255;
    meanBGR[1] = meanBGR[1]/255;
    meanBGR[2] = meanBGR[2]/255;
    
    HSVmeanBGR[0] = HSVmeanBGR[0]/255;
    HSVmeanBGR[1] = HSVmeanBGR[1]/255;
    HSVmeanBGR[2] = HSVmeanBGR[2]/255;
    
    /*--------- Hue and Saturation Histogram ----------*/
    // hue range: 0-179, saturation: 0-255, value: 0-255
    
    //calculating my own histogram
    CalculateHistogram(HSV_image);
    CalculateTextonHistogram(Texton_image);

    
}

void ImageSegment::WriteFeaturesToFile() {
    //cout << "where1" << endl;
    ofstream features;
    features.open("features.txt", ios::app);
    
    int label_svm = crack_label;

    features << label_svm << " ";
    int feature_count = 1;
//    //features << "MeanBGR: " << endl;
    for (int i = 0; i < 3; i++) {
        features << feature_count << ":" << meanBGR[i] << " ";
        feature_count++;
    }
    //features << endl;
    //features << "MeanHSVBGR: " << endl;
    for (int i = 0; i < 3; i++) {
        features << feature_count << ":" << HSVmeanBGR[i] << " ";
        feature_count++;
    }
    //features << endl;
    //features << "HHistogram: " << endl;
    for (int i = 0; i < binsH; i++) {
        features << feature_count << ":" << HHistogram[i] << " ";
        feature_count++;
    }
    //features << endl;
    //features << "SHistogram: " << endl;
    for (int i = 0; i < binsS; i++) {
        features << feature_count << ":" << SHistogram[i] << " ";
        feature_count++;
    }
    //features << endl;
    //features << "TEXTONS: " << endl;
    for (int i = 0; i < NUMTEXTONS; i++) {
        features << feature_count << ":" << THistogram[i] << " ";
        feature_count++;
    }
    //features << endl << "--------------------" << endl;
    features << endl;
    features.close();
}
/*------------------------------------------------------------*/
/*-------------------   Oversegmentation   -------------------*/
/*------------------------------------------------------------*/

/*-----------  Constructor  -----------*/
Oversegmentation::Oversegmentation(){
}

Oversegmentation::Oversegmentation(InputArray _original_image){
	Mat original_image_ = _original_image.getMat();
	im_height_ = original_image_.rows;
	im_width_ = original_image_.cols;
    pixel_labels_ = Mat(Size(im_width_,im_height_),CV_32S, -1);
}

Oversegmentation::~Oversegmentation(){
    // cout << segments_.size();
    for (int i = 0; i < segments_.size();++i){
        delete segments_[i];
    }
}

bool Oversegmentation::IsNotSet(Point pt){
    return pixel_labels_.at<int>(pt) == -1;
}

void Oversegmentation::ResetSegmentation(){
    for (int i = 0; i < segments_.size();++i)
        delete segments_[i];
    segments_.clear();
    pixel_labels_.setTo(-1);
}

size_t Oversegmentation::NumberOfSegments(){
    return segments_.size();
}

// /*-----------  Get Pixel Labels - returns Mat  -----------*/
// Mat Oversegmentation::getPixelLabelMat(){
//     return pixel_labels_;
// }

/*-----------  Add new superpixel at point  -----------*/
int Oversegmentation::AddNewSegmentAt(Point pt){
    int segment_label = segments_.size();
    segments_.push_back(new ImageSegment(segment_label, Point2f(pt.x,pt.y)));
    pixel_labels_.at<int>(pt) = segment_label;
    return segment_label;
}

void Oversegmentation::AddPixelToSegment(Point pt, int segment_label){
    if (segment_label >= 0){
        pixel_labels_.at<int>(pt) = segment_label;
        segments_[segment_label]->AddPixel(pt);
    }
}

/*-----------  Recompute superpixel centroids  -----------*/
vector<Point> Oversegmentation::ComputeSegmentCentroids(){
    vector<Point> mean_centroids(segments_.size(), Point(0,0));
    int* ptr_pix_labels;

    for (int i = 0; i < im_height_; ++i){
        ptr_pix_labels = pixel_labels_.ptr<int>(i);
        for (int j = 0; j < im_width_; ++j){
            if(ptr_pix_labels[j] >= 0){
                mean_centroids[ptr_pix_labels[j]].x += j;
                mean_centroids[ptr_pix_labels[j]].y += i;
            }
        }
    }

    for (int i = 0; i < segments_.size(); ++i) {
        mean_centroids[i] *= 1.0/segments_[i]->area_;
        segments_[i]->centroid_ = mean_centroids[i];
    }

    return mean_centroids;
}

/*-----------  Recompute superpixel areas  -----------*/
vector<int> Oversegmentation::ComputeSegmentAreas(){
    vector<int> segment_areas(segments_.size(),0);
    int* ptr_pix_labels;

    for (int i = 0; i < im_height_; ++i){
        ptr_pix_labels = pixel_labels_.ptr<int>(i);

        for (int j = 0; j < im_width_; ++j)
            if(ptr_pix_labels[j] >= 0)
                segment_areas[ptr_pix_labels[j]]++;
    }

    for (int i = 0; i < segments_.size(); ++i) {
        segments_[i]->area_ = segment_areas[i];
    }

    return segment_areas;
}

/*----------- Compute Segment Features ------------*/
void Oversegmentation::ComputeSegmentFeatures(){
    for (int i = 0; i < segments_.size(); i++) {
        _original_image.copyTo(segments_[i]->_original_image);
        Texton_image.copyTo(segments_[i]->Texton_image);
        _edges.copyTo(segments_[i]->_edges);
        segments_[i]->ComputeFeatures();
        segments_[i]->ComputeLabel();
        segments_[i]->WriteFeaturesToFile();

    }
    return;
}

/*----------- Return superpixel centroids -----------*/
vector<Point> Oversegmentation::GetCentroids(){
    vector<Point> vector_of_centroids;

    for (int i = 0; i < segments_.size(); ++i)
        vector_of_centroids.push_back(segments_[i]->centroid_); 

    return vector_of_centroids;
}

/*----------- Get centroid for superpixel[i] (int) -----------*/
Point Oversegmentation::SegmentCentroid(int i){
    Point pt = segments_[i]->centroid_;
    return pt;
}

/*-----------  Get centroid for superpixel[i] (float)  -----------*/
Point2f Oversegmentation::SegmentCentroid_f(int i){
    return segments_[i]->centroid_;
}

/*-----------  Get Pixel Areas - returns vector of areas -----------*/
vector<int> Oversegmentation::GetAreasOfAllSegments(){
    vector<int> pixel_areas;
    for(int i=0; i < segments_.size(); ++i) pixel_areas.push_back(segments_[i]->area_);
    return pixel_areas;
}

/*----------- Get area of a single segment -----------*/
int Oversegmentation::SegmentArea(int segment_label){
    return segments_[segment_label]->area_;
}

/*-----------  Delete and renumber segments -----------*/
// Pass a boolean vector containing True for segments to be deleted

vector<int> Oversegmentation::DeleteSegments(vector<bool> discard_list){
    vector<int> new_indices(segments_.size(),0);
    vector<ImageSegment*> new_segments;
    
    for (int i = 0; i < discard_list.size(); ++i){
        if(discard_list[i]){
            new_indices[i] = -1;
            delete segments_[i];
        }
        else{
            new_segments.push_back(segments_[i]);
            new_indices[i] = new_segments.size()-1;
        }
    }
    segments_.clear();
    segments_ = new_segments;

    MatIterator_<int> itr_pixel_labels = pixel_labels_.begin();
    while(itr_pixel_labels != pixel_labels_.end()){
        if (*itr_pixel_labels >= 0)
            *itr_pixel_labels = new_indices[*itr_pixel_labels];
        itr_pixel_labels++;
    }
    return new_indices;
}

/*-------- Construct list of pixels for each segment --------*/
// Creates explicit list of points in each segment

void Oversegmentation::ListPixelsForEachSegment(){
    for(int i = 0; i < segments_.size(); ++i){
        segments_[i]->area_ = 0;
        segments_[i]->pixel_list_.clear();
        segments_[i]->label_ = i;
    }

    int* ptr_pix_labels;
    for(int i = 0; i < im_height_; ++i){
        ptr_pix_labels = pixel_labels_.ptr<int>(i);
        for(int j = 0; j < im_width_; ++j){ 
            if (ptr_pix_labels[j] >= 0){
                AddPixelToSegment(Point(j,i), ptr_pix_labels[j]);
            }
        }
    }
    ComputeSegmentAreas();
    ComputeSegmentCentroids();
    ComputeSegmentFeatures();

}
/*---------- Visualizing the crack label computed using edge detection ---------*/
void Oversegmentation::ShowClassifiedLabelImage(Mat mask){
//    imshow("mask", mask);
//    waitKey();
//    cout << mask;
//    cout << mask.size() <<", " <<  _original_image.size() << endl;
    int count = 0;
    for (int i = 0; i < segments_.size(); i++) {
        if (segments_[i]->crack_label == 1) {
            for (int j = 0; j < segments_[i]->pixel_list_.size(); j++){
                if (mask.at<uchar>(segments_[i]->pixel_list_[j].y, segments_[i]->pixel_list_[j].x) == 1) {
                Vec3b temp = _original_image.at<Vec3b>(segments_[i]->pixel_list_[j].y, segments_[i]->pixel_list_[j].x);
                _original_image.at<Vec3b>(segments_[i]->pixel_list_[j].y, segments_[i]->pixel_list_[j].x) = Vec3b(temp.val[0], temp.val[1], 2*temp.val[2]);
                }
            }
        }
    }
    
    imwrite("image_classify.png", _original_image);
    return;
}

/*---------- Change label of segment ----------*/
// Works only if pixel_list_ has been populated for each segment

// pixel_list_ can be populated by using AddPixelToSegment to assign
// pixels to segments instead of directly modifying the Mat pixel_labels_

// If segments have been created using Mat pixel_labels_, 
// call ListPixelsForEachSegment to populate the pixel_list_ for each segment

void Oversegmentation::RelabelSegment(int orig_label, int new_label){
    vector<Point> pixels_to_relabel = segments_[orig_label]->pixel_list_;

    for (int i = 0; i < pixels_to_relabel.size(); ++i)
        AddPixelToSegment(pixels_to_relabel[i], new_label);

    segments_[orig_label]->pixel_list_.clear();
}

/*-----------  Visualize superpixel borders  -----------*/
void Oversegmentation::PrintSegmentBorders(InputArray _input_image, OutputArray _segment_borders){
    Mat image = _input_image.getMat();
    Mat display_img = _segment_borders.getMat();
    display_img = image.clone();
    Vec3b* p_disp_img;
    int* p_label_above;
    int* p_label_below;
    int* p_label_this;
    for(int i = 1; i < pixel_labels_.rows - 1; ++i){
        p_label_above = pixel_labels_.ptr<int>(i-1);
        p_label_below = pixel_labels_.ptr<int>(i+1);
        p_label_this = pixel_labels_.ptr<int>(i);
        p_disp_img = display_img.ptr<Vec3b>(i);
        for(int j = 1; j < pixel_labels_.cols - 1; ++j){
            p_disp_img[j] = (p_label_above[j] == p_label_below[j] && p_label_this[j-1] == p_label_this[j+1]) ? p_disp_img[j] : Vec3b(0,0,255);
        }
    }
}

void Oversegmentation::PrintClusterCentroids(InputArray _input_image, OutputArray _segment_centroids){

}
void Oversegmentation::RecolorWithCentroids(InputArray _input_image, OutputArray _recolored_image){

}
