// Version  : Srivatsan Varadhrajan
// Version 1: Vinay Palakkode
//
//
//
#include "../include/bow_slic.hpp"

Point four_neighbors[] = {Point(-1,0),
                        Point(1,0),
                        Point(0,-1),
                        Point(0,1)};

Point eight_neighbors[] = {Point(-1,0),
                        Point(1,0),
                        Point(0,-1),
                        Point(0,1),
                        Point(-1,-1),
                        Point(1,-1),
                        Point(-1,1),
                        Point(1,1)};

BagOfWordsSlic::BagOfWordsSlic(int histogram_distance_weight, int area_threshold, int tightness)
:kMaxIter(10),
kMinSuperpixelAreaThreshold(area_threshold),
kHistogramDistanceWeight(histogram_distance_weight),
kCentroidErrorThreshold(4){
    m_ = tightness;
}

BagOfWordsSlic::~BagOfWordsSlic(){
    distance_matrix_.release();
    visual_word_histogram_matrix_.release();
    cluster_centroids_.clear();
    delete image_oversegmentation_;
}

void BagOfWordsSlic::RenumberEachConnectedComponent(){
    int new_label, old_label;
    Mat old_labels_mat = image_oversegmentation_->pixel_labels_.clone();
    image_oversegmentation_->ResetSegmentation();

    int* old_labels_ptr;
//    int* new_labels_ptr;

    for(int i = 0; i < im_height_; ++i){
        old_labels_ptr = old_labels_mat.ptr<int>(i);

        for(int j = 0; j < im_width_; ++j){

            if (image_oversegmentation_->IsNotSet(Point(j,i))){
                old_label = old_labels_ptr[j];
                Point this_point(j,i), neighboring_point;
                stack<Point> pixels_to_check;
                pixels_to_check.push(this_point);
                new_label = image_oversegmentation_->NumberOfSegments();
                image_oversegmentation_->AddNewSegmentAt(this_point);

                while(!pixels_to_check.empty()){
                    this_point = pixels_to_check.top();
                    pixels_to_check.pop();
                    image_oversegmentation_->AddPixelToSegment(this_point, new_label);

                    for (int k = 0; k < 4; ++k){
                        neighboring_point = this_point + four_neighbors[k];

                        if(!OutOfBounds(neighboring_point, im_width_, im_height_)
                            && (old_labels_mat.at<int>(neighboring_point) == old_label)
                            && (image_oversegmentation_->IsNotSet(neighboring_point))){
                                pixels_to_check.push(neighboring_point);
                        }
                    }
                }
            }
        }
    }
   
    image_oversegmentation_->ComputeSegmentAreas();
    image_oversegmentation_->ComputeSegmentCentroids();
}

void BagOfWordsSlic::RelabelSmallSegmentsToNearestNeighbor(int area_threshold){
    Mat labels = image_oversegmentation_->pixel_labels_;
    int num_superpixels = image_oversegmentation_->NumberOfSegments();
    vector<bool> discard_list(num_superpixels, false);
    for(int i = 0; i < num_superpixels; ++i){
        if (image_oversegmentation_->SegmentArea(i) < area_threshold){
            discard_list[i] = true;
            queue<Point> pixel_bfs_queue;
            Point this_point = image_oversegmentation_->SegmentCentroid(i);
            Point next_point;
            pixel_bfs_queue.push(this_point);
            int new_label = i;
            while(!pixel_bfs_queue.empty() && new_label == i){
                this_point = pixel_bfs_queue.front();
                pixel_bfs_queue.pop();
                for (int k = 0; k < 4; ++k){
                    next_point = this_point + four_neighbors[k];
                    if(OutOfBounds(next_point, im_width_, im_height_))
                        continue;

                    new_label = labels.at<int>(next_point);
                    if (new_label == i)
                        pixel_bfs_queue.push(next_point);
                    else
                        break;
                }
            }
            image_oversegmentation_->RelabelSegment(i, new_label);
        }
    }
    image_oversegmentation_->DeleteSegments(discard_list);
    image_oversegmentation_->ComputeSegmentAreas();
    image_oversegmentation_->ComputeSegmentCentroids();
}

void BagOfWordsSlic::ComputeVisualWordHistograms(int half_window_height, int half_window_width, const Mat& visual_word_map){
    Mat accumulated_arrays(visual_word_map.size(),CV_32SC(KCENTERS));
    const unsigned short* visual_word_map_row;
    Vec50s* this_row;
    Vec50s* prev_row;
    int upper_val, left_val, upper_left_val;
    for(int i = 0; i < visual_word_map.rows; ++i){
        visual_word_map_row = visual_word_map.ptr<unsigned short>(i);
        this_row = accumulated_arrays.ptr<Vec50s>(i);
        if(i > 0) prev_row = accumulated_arrays.ptr<Vec50s>(i - 1);
        for(int j = 0; j < visual_word_map.cols; ++j)
           for(int k = 0; k < KCENTERS; ++k){
                upper_val = i > 0 ? prev_row[j][k] : 0;
                left_val = j > 0 ? this_row[j-1][k] : 0;
                upper_left_val = (i > 0 && j > 0) ? prev_row[j-1][k] : 0;
                this_row[j][k] = upper_val + left_val - upper_left_val + (k == visual_word_map_row[j]);
            } 
    }
    visual_word_histogram_matrix_.create(visual_word_map.size(),CV_64FC(KCENTERS));
    Vec50d* visual_word_histogram_row;
    Vec50s* window_bottom;
    Vec50s* window_top;
    int top_ind, bottom_ind, left_ind, right_ind;
    double window_area;
    for(int i = 0; i < visual_word_histogram_matrix_.rows; ++i){
        top_ind = max(i - half_window_height - 1, 0);
        window_top = accumulated_arrays.ptr<Vec50s>(top_ind);
        bottom_ind = min(i + half_window_height, im_height_ - 1);
        window_bottom = accumulated_arrays.ptr<Vec50s>(bottom_ind);
        visual_word_histogram_row = visual_word_histogram_matrix_.ptr<Vec50d>(i);
        for(int j = 0; j < visual_word_histogram_matrix_.cols; ++j){
            left_ind = max(0, j - half_window_width - 1);
            right_ind = min(im_width_-1, j + half_window_width);
            visual_word_histogram_row[j] = window_bottom[right_ind] + window_top[left_ind] - (window_top[right_ind] + window_bottom[left_ind]);
            window_area = (right_ind - left_ind) * (bottom_ind - top_ind);
            visual_word_histogram_row[j] /= window_area;
        }
    }
}

ClusterPoint::ClusterPoint():
pt_(Point2f(0,0)),
lab_color_(Vec3b()),visual_word_histogram_(Vec50d()){
}

ClusterPoint::ClusterPoint(Point2f _pt, Vec3b _lab_color, Vec50d _vword_histogram):
pt_(_pt),
lab_color_(_lab_color),
visual_word_histogram_(_vword_histogram){
}

ClusterPoint::~ClusterPoint(){
}

double ClusterPoint::distance_to(const ClusterPoint& a, int m, int S, int histogram_distance_weight){
    double squared_xy_dist = pow(this->pt_.x - a.pt_.x,2) + pow(this->pt_.y - a.pt_.y,2);
    squared_xy_dist *= (double)m*m/(S*S);

    double squared_lab_dist = 0;
    squared_lab_dist += pow((this->lab_color_[0] - a.lab_color_[0])* 100/255,2) ;
    squared_lab_dist += pow(this->lab_color_[1] - a.lab_color_[1],2);
    squared_lab_dist += pow(this->lab_color_[2] - a.lab_color_[2],2);

    double vword_histogram_similarity = 0, vword_histogram_distance = 0;
    for (int i = 0; i < KCENTERS; ++i)
        vword_histogram_similarity += min(this->visual_word_histogram_[i], a.visual_word_histogram_[i]);
    vword_histogram_distance = histogram_distance_weight * (1 - vword_histogram_similarity); 

    return sqrt(squared_xy_dist + squared_lab_dist) + vword_histogram_distance;
}

void BagOfWordsSlic::MoveCentroidsToLocalGradientMinima(){

    Mat grayscale_input_image(input_image_.rows, input_image_.cols, input_image_.depth());
    cvtColor(input_image_, grayscale_input_image, COLOR_BGR2GRAY);

    Mat horizontal_gradient_image(grayscale_input_image.size(),CV_32F);
    Mat vertical_gradient_image(grayscale_input_image.size(),CV_32F);
    Mat gradient_magnitude_image(grayscale_input_image.size(),CV_32F);

    Sobel(grayscale_input_image,horizontal_gradient_image,gradient_magnitude_image.depth(),1,0);
    Sobel(grayscale_input_image,vertical_gradient_image,gradient_magnitude_image.depth(),0,1);
    gradient_magnitude_image = horizontal_gradient_image.mul(horizontal_gradient_image) + vertical_gradient_image.mul(vertical_gradient_image);

    int x_ind, y_ind; 
    float min_magnitude;
    Point adjusted_centroid, neighboring_point;

    for(int i = 0; i < (int)cluster_centroids_.size(); ++i){
        min_magnitude = FLT_MAX;
        x_ind = min(max((int)cluster_centroids_[i].pt_.x,1),im_width_-2);
        y_ind = min(max((int)cluster_centroids_[i].pt_.y,1),im_height_-2);
        adjusted_centroid = Point(x_ind, y_ind);
        for(int j = 0; j < 8; ++j){
            neighboring_point = adjusted_centroid + eight_neighbors[j];
            if(gradient_magnitude_image.at<float>(neighboring_point) < min_magnitude){
                cluster_centroids_[i].pt_ = neighboring_point;
                min_magnitude = gradient_magnitude_image.at<float>(neighboring_point);
            }
        }
    }
}

void BagOfWordsSlic::UpdateClusterCentroids(Mat lab_image){

    /*--Reset centroids to zeros--*/
    // num_superpixels_ = image_oversegmentation_->NumberOfSegments();
    cluster_centroids_.clear();
    for(int i = 0; i < num_superpixels_; ++i)
        cluster_centroids_.push_back(ClusterPoint());

    /*Recompute cluster centroids using latest superpixel assignment---*/        
    int superpixel_label;
    int* superpixel_label_matrix_row;
    vector<Vec3i> sum_of_lab_within_superpixel(num_superpixels_,Vec3i(0,0,0));
    Vec50d* visual_word_histogram_row;
    Vec3b* lab_image_row;
    for(int i = 0; i < im_height_; ++i){
        superpixel_label_matrix_row = image_oversegmentation_->pixel_labels_.ptr<int>(i);
        lab_image_row = lab_image.ptr<Vec3b>(i);
        visual_word_histogram_row = visual_word_histogram_matrix_.ptr<Vec50d>(i);
        for(int j = 0; j < im_width_; ++j){
            if(superpixel_label_matrix_row[j] != -1){
                /*---iterate through each pixel---*/
                superpixel_label = superpixel_label_matrix_row[j];
                sum_of_lab_within_superpixel[superpixel_label] += lab_image_row[j];
                cluster_centroids_[superpixel_label].visual_word_histogram_ += visual_word_histogram_row[j];
            }
        }
    }

    image_oversegmentation_->ComputeSegmentCentroids();
    /* Compute Means */
    for(int i = 0; i < num_superpixels_; ++i){
        cluster_centroids_[i].pt_ = image_oversegmentation_->SegmentCentroid_f(i);
        cluster_centroids_[i].lab_color_ = sum_of_lab_within_superpixel[i]/image_oversegmentation_->SegmentArea(i);
        cluster_centroids_[i].visual_word_histogram_ = cluster_centroids_[i].visual_word_histogram_/image_oversegmentation_->SegmentArea(i);
    }
}


void BagOfWordsSlic::GenerateSuperpixels(int label_svm, InputArray _input_image, OutputArray _superpixels,
    int _number_of_superpixels, InputArray _visual_word_map, 
    InputArray _mask, OutputArray _superpixel_centroids){

    cout << "generating super pixels" << endl;
    /*--Convert from input arguments to class data-structures--*/
    input_image_ = _input_image.getMat();
    Mat visual_word_map = _visual_word_map.getMat();
    Mat mask = _mask.getMat();

    cvtColor(mask, mask, CV_BGR2GRAY);
    im_height_ = input_image_.rows; im_width_ = input_image_.cols;
    num_superpixels_ = _number_of_superpixels;

    // Compute step size
    S_ = sqrt(im_height_*im_width_/num_superpixels_);

    /*--Get Lab Image--*/
    Mat lab_image(input_image_.size(), CV_8UC3);
    cvtColor(input_image_, lab_image, COLOR_BGR2Lab);

    /*--Initialize centroid locations to regular grid with computed step size--*/
    int grid_width = ceil((float)im_width_/S_);
    int grid_height = ceil((float)im_height_/S_);
    num_superpixels_  = grid_width*grid_height;
    image_oversegmentation_ = new Oversegmentation(input_image_);
    cluster_centroids_.resize(num_superpixels_);
    
    int index_cluster = 0;

    for(int i = 0; i < (int)cluster_centroids_.size(); ++i){
        int offset = 0;
        if (((i-1)/grid_width)%2 == 0)
            offset = S_/4;
        else
            offset = 3*S_/4;
        int x = offset + ((i-1)%grid_width)*S_;
        int y = offset + ((i-1)/grid_width)*S_;
        int s_ = ceil((float)S_/2);
        
        if ((int)(mask.at<uchar>(y,x)) == 1
            && mask.at<uchar>(min(y+s_, im_height_),min(x+s_, im_width_)) == 1
            && mask.at<uchar>(max(y-s_, 0), max(x-s_,0)) ==1
            && mask.at<uchar>(max(y-s_, 0), min(x+s_, im_width_))==1
            && mask.at<uchar>(min(y+s_, im_height_), max(x-s_, 0))==1) {
//            cluster_centroids_[i].pt_.x = offset + ((i-1)%grid_width)*S_;
//            cluster_centroids_[i].pt_.y = offset + ((i-1)/grid_width)*S_;
            cluster_centroids_[i].pt_.x = x;
            cluster_centroids_[i].pt_.y = y;
        }
            index_cluster++;
    }

    // Find local minimum of gradient magnitude and adjust centroid locations
    MoveCentroidsToLocalGradientMinima();

    for(int i = 0; i < (int)cluster_centroids_.size(); ++i)
        image_oversegmentation_->AddNewSegmentAt(cluster_centroids_[i].pt_);

    if (visual_word_map.empty()){
        visual_word_histogram_matrix_.create(input_image_.size(),CV_64FC(KCENTERS));
    }else{
        //Compute visual_word histograms
        ComputeVisualWordHistograms(5,5,visual_word_map);        
    }

    /*---Generate descriptors at centroid locations---*/

    // Get L,A,B vector for each centroid
    for(int i = 0; i < (int)cluster_centroids_.size();++i){
        cluster_centroids_[i].lab_color_  = lab_image.at<Vec3b>(image_oversegmentation_->SegmentCentroid(i));
        cluster_centroids_[i].visual_word_histogram_ = visual_word_histogram_matrix_.at<Vec50d>(image_oversegmentation_->SegmentCentroid(i));
    }

    /*--Initialize distance to nearest centroid for each pixel--*/
    distance_matrix_.create(input_image_.size(), CV_64F);
    double dist;
    int* superpixel_label_matrix_row;
    Vec50d* visual_word_histogram_row;
    double* distance_matrix_row;
    Vec3b* lab_image_row;

    /*---First loop starts: for iter = 1 to kMaxIter---*/
    for(int iter = 0; iter < kMaxIter; ++iter){

        /* Reset distances */
        distance_matrix_.setTo(DBL_MAX);

        cout << "EM iteration: " << iter << "\n";
        cout << "Number of segments now:   " << image_oversegmentation_->NumberOfSegments() << "\n";

        int x_lower_limit, x_upper_limit, y_lower_limit, y_upper_limit;
        /*---Second loop starts: for cInd = 1 to num_superpixels_---*/
        for(int i = 0; i < num_superpixels_; ++i){
            int centroid_x = cluster_centroids_[i].pt_.x, centroid_y = cluster_centroids_[i].pt_.y;
            /*---Third loop starts: Iterate through each pixel in 2S+1 x 2S+1 window size around centroid[i]---*/
            y_lower_limit = max(centroid_y - S_,0);
            y_upper_limit = min(centroid_y + S_,im_height_);
            x_lower_limit = max(centroid_x - S_,0);
            x_upper_limit = min(centroid_x + S_,im_width_);

            for(int pixel_y = y_lower_limit; pixel_y < y_upper_limit; pixel_y++){

                lab_image_row = lab_image.ptr<Vec3b>(pixel_y);
                distance_matrix_row = distance_matrix_.ptr<double>(pixel_y);
                visual_word_histogram_row = visual_word_histogram_matrix_.ptr<Vec50d>(pixel_y);
                superpixel_label_matrix_row = image_oversegmentation_->pixel_labels_.ptr<int>(pixel_y);

//                int temp_x = x_lower_limit;
                for(int pixel_x = x_lower_limit; pixel_x < x_upper_limit; ++pixel_x){

                    if (mask.at<uchar>(pixel_y, pixel_x) != 1)
                        continue;
                    //Compute the pixel's distance to centroid[i]
                    ClusterPoint pixel(Point2f(pixel_x,pixel_y), lab_image_row[pixel_x], visual_word_histogram_row[pixel_x]);
                    if (visual_word_map.empty()){
                        dist = cluster_centroids_[i].distance_to(pixel, m_, S_, 0);
                    }else{
                        dist = cluster_centroids_[i].distance_to(pixel, m_, S_, kHistogramDistanceWeight);
                    }
                    /*---Update the superpixel[pixel] and distance[pixel] if required---*/
                    if(dist < distance_matrix_row[pixel_x]){
                        distance_matrix_row[pixel_x] = dist;
                        superpixel_label_matrix_row[pixel_x] = i;
                    } 
                }
            }
            /*---Third loop ends---*/    
        }/*---Second loop ends---*/
        image_oversegmentation_->ComputeSegmentAreas();

        //Create vector of flags to indicate discardedsuperpixel_labels 
        vector<bool> discard_list(num_superpixels_,false);

        /*---Fourth loop: iterate through each centroid(superpixel) and count number of pixels within.
        If count is too small, mark superpixel for discarding---*/
        for(int i = 0; i < num_superpixels_; ++i) {
            if (discard_list[i] != 1) {
                discard_list[i] = image_oversegmentation_->SegmentArea(i) < kMinSuperpixelAreaThreshold;

            }
        }

        int num_discarded = 0;
        for(int i = 0; i < (int)discard_list.size(); ++i)
            if(discard_list[i])
                ++num_discarded;

        image_oversegmentation_->DeleteSegments(discard_list);
        num_superpixels_ = image_oversegmentation_->NumberOfSegments();

        vector<Point> old_centroids = image_oversegmentation_->GetCentroids();
        UpdateClusterCentroids(lab_image);
        vector<Point> new_centroids = image_oversegmentation_->GetCentroids();

        /*---Check for convergence - if converged, then break from loop---*/
        int max_centroid_displacement = -1; 
        for(int i = 0; i < num_superpixels_ ; ++i){
            int x_difference = abs(old_centroids[i].x-new_centroids[i].x);
            int y_difference = abs(old_centroids[i].y-new_centroids[i].y);
            max_centroid_displacement = std::max(max_centroid_displacement,x_difference);
            max_centroid_displacement = std::max(max_centroid_displacement,y_difference);
        }

//        cout << "max distance:  " << max_centroid_displacement << "\n";
        if (max_centroid_displacement <= kCentroidErrorThreshold){
            RenumberEachConnectedComponent();
            RelabelSmallSegmentsToNearestNeighbor(kMinSuperpixelAreaThreshold);
//            cout << "Number of segments now:   " << image_oversegmentation_->NumberOfSegments() << "\n";

            break;
        }
 
    /*---First loop ends---*/
    }
    image_oversegmentation_->pixel_labels_.copyTo(_superpixels);

    vector<Point> centroids = image_oversegmentation_->GetCentroids();
    _superpixel_centroids.create(centroids.size(), 2, CV_32S);
    Mat superpixel_centroids = _superpixel_centroids.getMat();
    cout << "where3" << endl;

    for (int i = 0; i < (int)centroids.size(); ++i){
        superpixel_centroids.at<int>(i,0) = centroids[i].x;
        superpixel_centroids.at<int>(i,1) = centroids[i].y;
    }
    _input_image.copyTo(image_oversegmentation_->_original_image);
    visual_word_map.copyTo(image_oversegmentation_->Texton_image);
    image_oversegmentation_->label_svm = label_svm;
    image_oversegmentation_->ListPixelsForEachSegment();
    cout << "where2" << endl;
//    image_oversegmentation_->ComputeSegmentFeatures(src, visual_word_map, label_svm);
    //cout << "where3" << endl;
//    cout << "total num superpixels: " << centroids.size() << endl;
//    _number_of_superpixels_total = centroids.size();
//    cout << "total num superpixels: " << centroids.size() << endl;
//
//    image_oversegmentation_->DeleteOversegmentation();
//    cout << "total num superpixels2: " << centroids.size() << endl;

    return;
    /*---Clean up image_oversegmentation_->pixel_labels_---*/

}

void BagOfWordsSlic::OverlaySuperpixelBoundaries(InputArray _image, InputArray _superpixels, OutputArray _boundaries){
    Mat image = _image.getMat();
    Mat labels = _superpixels.getMat();
    Mat overlay_image = image.clone();
    Vec3b* overlay_pixel_ptr;
    int* label_row_above;
    int* label_row;
    for(int i = 1; i < labels.rows; ++i){
        label_row_above = labels.ptr<int>(i-1);
        label_row = labels.ptr<int>(i);
        overlay_pixel_ptr = overlay_image.ptr<Vec3b>(i);
        for(int j = 1; j < labels.cols; ++j){
            overlay_pixel_ptr[j] = (label_row_above[j] == label_row[j] && label_row[j-1] == label_row[j+1]) ? overlay_pixel_ptr[j] : Vec3b(0,0,255);
        }
    }
    overlay_image.copyTo(_boundaries);
}
Textons::Textons(int DictionarySize, InputArray input_image):
k(DictionarySize){
    input_image.copyTo(test_image);
}
void gauss1d(double *g, int size, int scale, float *pts, int order) {
    
    double sqrPts[size];
    int variance  = scale*scale;
    double denom = 2*variance;
    for (int i = 0; i < size; i++) {
        sqrPts[i] = pts[i]*pts[i];
        g[i] = exp(-1*sqrPts[i]/denom);
        g[i] /= sqrt(PI*denom);
        
        switch (order) {
            case 1:
                g[i] = -1*g[i]*pts[i]/variance;
                break;
            case 2:
                g[i] = g[i]*(sqrPts[i]-variance)/(variance*variance);
                break;
        }
    }
    return;
}

void normalise(Mat *F, double *gx, double *gy, int sup) {
    
    double sum = 0, absSum = 0;
    //ofstream f;
    //f.open("f.txt", ios::app);
    Mat Ftemp(sup, sup, CV_64F);
    for (int i = 0; i < sup; i++) {
        for (int j = 0; j < sup; j++) {
            double temp = (double)(gx[i*sup+j]*gy[i*sup+j]);
            Ftemp.at<double>(j,i) = temp;
            sum = sum + temp;
        }
    }
    double mean = sum/(sup*sup);
    for (int i = 0; i < sup; i++) {
        for (int j = 0; j < sup; j++) {
            Ftemp.at<double>(i,j) -= mean;
            absSum += abs(Ftemp.at<double>(i,j));
        }
    }
    
    for (int i = 0; i < sup; i++) {
        for (int j = 0; j < sup; j++) {
            Ftemp.at<double>(i,j) /= absSum;
            F->at<float>(i,j) = (float)Ftemp.at<double>(i,j);
            //        f << F->at<float>(i,j) << ", ";
        }
        //    f << endl;
    }
    //f << "----------" << endl;
    //f.close();
    return;
}
void makeFilter(Mat *F, int scale, int phasex, int phasey, float rotPtsx[], float rotPtsy[], int sup) {
    double gx[sup*sup];
    double gy[sup*sup];
    gauss1d(gx, sup*sup, 3*scale, rotPtsx, phasex);
    gauss1d(gy, sup*sup, scale, rotPtsy, phasey);
    
    normalise(F, gx, gy, sup);
    return;
}
void Textons::makeRFSFilters() {
    
    int SCALEX[3] = {1, 2, 4};
    for (int i = 0; i < NF; i++) {
        F[i].create(SUP, SUP, CV_32FC1);
    }
    
    int hsup = (SUP - 1)/2;
    float x[SUP*SUP], y[SUP*SUP];
    
    for (int i = 0; i < SUP; i++) {
        for (int j = 0; j < SUP; j++) {
            x[i*SUP + j] = -1*hsup + i;
            y[j*SUP + i] = hsup - i;
        }
    }
    
    int count = 0;
    for (int scale = 0; scale < NSCALES; scale++) {
        for (int orient = 0; orient < NORIENT; orient++) {
            float angle = PI*orient/NORIENT;
            float c = cos(angle);
            float s = sin(angle);
            
            // Calculate rotated points
            float rotPtsx[SUP*SUP];
            float rotPtsy[SUP*SUP];
            for (int i = 0; i < SUP; i++) {
                for (int j = 0; j < SUP; j++) {
                    float x_prime = x[i*SUP+j];
                    float y_prime = y[i*SUP+j];
                    rotPtsx[i*SUP+j] = x_prime*c - y_prime*s;
                    rotPtsy[i*SUP+j] = x_prime*s + y_prime*c;
                }
            }
            makeFilter(&F[count], SCALEX[scale], 0, 1, rotPtsx, rotPtsy, SUP);
            makeFilter(&F[count + NEDGE], SCALEX[scale], 0, 2, rotPtsx, rotPtsy, SUP);
            count++;
        }
    }
    return;
}
cv::Mat fspecialLoG(int WinSize, double sigma){
    
    cv::Mat xx (WinSize,WinSize,CV_64F);
    for (int i=0;i<WinSize;i++){
        for (int j=0;j<WinSize;j++){
            xx.at<double>(j,i) = (i-(WinSize-1)/2)*(i-(WinSize-1)/2);
        }
    }
    cv::Mat yy;
    cv::transpose(xx,yy);
    cv::Mat arg = -(xx+yy)/(2*pow(sigma,2));
    cv::Mat h (WinSize,WinSize,CV_64F);
    for (int i=0;i<WinSize;i++){
        for (int j=0;j<WinSize;j++){
            h.at<double>(j,i) = pow(exp(1),(arg.at<double>(j,i)));
        }
    }
    double minimalVal, maximalVal;
    minMaxLoc(h, &minimalVal, &maximalVal);
    cv::Mat tempMask = (h>DBL_EPSILON*maximalVal)/255;
    tempMask.convertTo(tempMask,h.type());
    cv::multiply(tempMask,h,h);
    
    if (cv::sum(h)[0]!=0){h=h/cv::sum(h)[0];}
    
    cv::Mat h1 = (xx+yy-2*(pow(sigma,2)))/(pow(sigma,4));
    cv::multiply(h,h1,h1);
    h = h1 - cv::sum(h1)[0]/(WinSize*WinSize);
    return h;
}
void Textons::createFilterResponses() {
    Mat grey_image_float_, grey_image_;
    float scales[3] = {(float)sqrt(1), (float)sqrt(2), (float)sqrt(3)};
    
    cvtColor(test_image, grey_image_, CV_BGR2GRAY);
    grey_image_.convertTo(grey_image_float_, CV_64F);
    
    Mat ImageFilterResponses[NumFilters];
    for (int i = 0; i < NF; i++) {
        filter2D(grey_image_float_, ImageFilterResponses[i], -1, F[i], Point(-1,-1),
                 0, BORDER_REPLICATE);
        ImageFilterResponses[i].convertTo(ImageFilterResponses[i], CV_32F);
    }

    int sup = 51;
    
    for (int i = 0; i < 3; i++) {
        Mat h = fspecialLoG(sup, scales[i]);
        filter2D(grey_image_float_, ImageFilterResponses[i+NF], -1, h, Point(-1,-1),0, BORDER_DEFAULT);
        ImageFilterResponses[i+NF].convertTo(ImageFilterResponses[i+NF], CV_32F);
    }

    for (int r = 0; r < grey_image_.rows; r++) {
        for (int c = 0; c < grey_image_.cols; c++) {
            int i = 0;
            int index_bar = 0;
            float max_temp_bar = abs(ImageFilterResponses[0].at<float>(r,c));
            for (i = 1; i < NBAR; i++) {
                if (max_temp_bar < abs(ImageFilterResponses[i].at<float>(r,c))) {
                    max_temp_bar = abs(ImageFilterResponses[i].at<float>(r,c));
                    index_bar = i;
                }
            }
            int flag1 = 0;
            if (ImageFilterResponses[index_bar].at<float>(r,c) < 0)
                flag1 = 1;
            
            int max = index_bar;
            max = max%NORIENT;
            
            for (int i = 0; i < NSCALES*2; i++) {
                float values[NORIENT];
                for (int j = 0; j < NORIENT; j++) {
                    values[j] = ImageFilterResponses[NORIENT*i+j].at<float>(r,c);
                    if (flag1 == 1 && i < NSCALES)
                        values[j] = -1*values[j];
                }
                
                int iter = 0;
                for (int k = max; k < NORIENT; k++) {
                    ImageFilterResponses[NORIENT*i + iter].at<float>(r,c) = values[k];
                    iter++;
                }
                for (int k = 0; k < max; k++) {
                    if ((NORIENT*i + iter) < NBAR)
                        ImageFilterResponses[NORIENT*i + iter].at<float>(r,c) = -1*values[k];
                    else
                        ImageFilterResponses[NORIENT*i + iter].at<float>(r,c) = values[k];
                    iter++;
                }
            }
            
        }
    }

    Textons::pushToImageTextons(ImageFilterResponses);
    return;
}

void Textons::pushToImageTextons(Mat Responses[]){
    
    TestImageTextonMap.clear();
    for (int r = 0; r < Responses[0].rows; r++) {
        for (int c = 0; c < Responses[0].cols; c++) {
            FilterResponses temp;
            for (int i = 0; i < NumFilters; i++) {
                temp.Filters[i] = Responses[i].at<float>(r,c);
            }
            TestImageTextonMap.push_back(temp);
        }
    }
}

void Textons::KmeansCentersReadFromFile() {
    ifstream kmeansCenters("kmeans.txt", ios::in);
    
    cout << "reading kmeans from file" << endl;
    for (int i = 0; i < k; i++) {
        FilterResponses temp;
        for (int j = 0; j < NumFilters; j++) {
            kmeansCenters >> temp.Filters[j];
        }
        Dictionary.push_back(temp);
    }
    return;
}
double computeDistance(Mat a, Mat b) {
    double dist = 0;
    for (int i = 0; i < Textons::NumFilters; i++) {
        dist += pow((a.at<float>(i,0) - b.at<float>(i,0)), 2);
    }
    dist = sqrt(dist);
    return dist;
}
typedef struct str_thread_Args {
    int tid;
    vector <FilterResponses> *dictionary_thread;
    vector <FilterResponses> *pointsThread;
    Mat *textonMapLocal;
    int rows;
    int cols;
    int numFilters;
}thread_args_t;

void *computeTexton(void *ptr) {
    
    thread_args_t *args = (thread_args_t *)ptr;
    int t = args->tid;
    int r = args->rows;
    int c = args->cols;
    int n = args->numFilters;
    vector <FilterResponses> dict_local;
    vector <FilterResponses> points;
    points = *args->pointsThread;
    dict_local = *args->dictionary_thread;

    Mat *textonMapThreadLocal = args->textonMapLocal;
    
    for(int i = t; i < r; i+= NUM_THREADS) {
        for (int c_iter = 0; c_iter < c; c_iter++) {
            double dist1 = (double) numeric_limits<int>::max();
            Mat a,b;
            a.create(n,1, CV_32F);
            b.create(n,1, CV_32F);
            for (int j = 0; j < n; j++) {
                a.at<float>(j,0) = points[i*c + c_iter].Filters[j];
            }
            int TextonLabel = 0;
            for (int j = 0; j < (int)dict_local.size(); j++) {
                for (int l = 0; l < n; l++) {
                    b.at<float>(l,0) = dict_local[j].Filters[l];
                }
                double dist2 = computeDistance(a, b);
                if (dist2 < dist1){
                    TextonLabel = j;
                    dist1 = dist2;
                }
            }
            textonMapThreadLocal->at<uchar>(i,c_iter) = (int)TextonLabel+1;
        }
    }
    return NULL;
    
}


/* ----- Generate Texton Map for a given Image ----- */
void Textons::generateTextonMap(Mat TextonMapLocal) {//, Mat TextonMap) {
//    int width = test_image.cols;
//    int height = test_image.rows;
    TextonMap.create(1,1,CV_8UC3);
    
    
//    int numThreads = TextonMapLocal.rows*TextonMapLocal.cols;
    pthread_t threads[NUM_THREADS];
    thread_args_t thread_args[NUM_THREADS];
    
    for (int tIter = 0; tIter < NUM_THREADS; tIter++) {

        thread_args[tIter].tid = (int)tIter;
        thread_args[tIter].rows = TextonMapLocal.rows;
        thread_args[tIter].cols = TextonMapLocal.cols;
        thread_args[tIter].pointsThread = &TestImageTextonMap;
        thread_args[tIter].numFilters = NumFilters;
        thread_args[tIter].textonMapLocal = &TextonMapLocal;
        thread_args[tIter].dictionary_thread = &Dictionary;
        
        pthread_create(&threads[tIter], NULL,  computeTexton, (void *) &thread_args[tIter]);
    }
    
    // wait for each thread to complete
    for (int index = 0; index < NUM_THREADS ; index++) {
        // block until thread 'index' completes
        int result_code = pthread_join(threads[index], NULL);
        assert(0 == result_code);
    }
    

    
    int colors[64][3];
    int variant[4] = {0, 85, 170, 255};
    int i = 0;
    for (int p = 0; p < 4; p++) {
        for (int q = 0; q < 4; q++) {
            for (int r = 0; r < 4; r++) {
                colors[i][0] = variant[p];
                colors[i][1] = variant[q];
                colors[i][2] = variant[r];
                
//                int temp1 = colors[i][0];
//                int temp2 = colors[i][1];
//                int temp3 = colors[i][2];
                // cout << colors[i][0] << " " << colors[i][1] << " " << colors[i][2] << endl;
                i++;
                
                //cout << variant[p] << " " << variant[q] << " " << variant[r] << endl;
            }
        }
    }
    
    Mat TextonMapColors(TextonMapLocal.rows, TextonMapLocal.cols, CV_8UC3);
    for (int i = 0; i < TextonMapLocal.rows; i++) {
        for (int j = 0; j < TextonMapLocal.cols; j++) {
            uchar TextonLabel = TextonMapLocal.at<uchar>(i,j);
            
            //cout << colors[k-TextonLabel-1][0] << " " << colors[k-TextonLabel-1][1] << " " << colors[k-TextonLabel-1][2] << endl;
            TextonMapColors.at<Vec3b>(i,j)[0] = colors[k-TextonLabel-1][0];
            TextonMapColors.at<Vec3b>(i,j)[1] = colors[k-TextonLabel-1][1];
            TextonMapColors.at<Vec3b>(i,j)[2] = colors[k-TextonLabel-1][2];
        }
    }
//    imshow("textonMapColor", TextonMapColors);
    imwrite("output_texton.png", TextonMapColors);
    imwrite("texton_bw.png", TextonMapLocal);
//    imwrite("TextonMap.png", TextonMapLocal);
//    waitKey();
    /*    imshow("textonMap", TextonMapLocal);
     waitKey();
     */
    return;
}

void Textons::TextonMapGeneration(Mat TextonMap){
    makeRFSFilters();
    cout << "creating filter responses for test image" << endl;
    createFilterResponses();
    KmeansCentersReadFromFile();
    cout << "generating textons" << endl;
    time_t before = time(0);
    generateTextonMap(TextonMap);
    time_t after = time(0);
    cout << "Time taken to generate texton map: " << after-before << " seconds" << endl;
    return;

}
Textons::~Textons(){
}
