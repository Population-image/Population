#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

using namespace cv;
int main(){
   
  Mat img_1 = imread( "D:/Users/vtariel/Desktop/cni1.jpg", CV_LOAD_IMAGE_GRAYSCALE );
  Mat img_2 = imread( "D:/Users/vtariel/Desktop/cni3.png", CV_LOAD_IMAGE_GRAYSCALE );

  if( !img_1.data || !img_2.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;

  SiftFeatureDetector detector;

  std::vector<KeyPoint> keypoints_1, keypoints_2;

  detector.detect( img_1, keypoints_1 );
  detector.detect( img_2, keypoints_2 );

  //-- Step 2: Calculate descriptors (feature vectors)
  SiftDescriptorExtractor extractor;

  Mat descriptors_1, descriptors_2;

  extractor.compute( img_1, keypoints_1, descriptors_1 );
  extractor.compute( img_2, keypoints_2, descriptors_2 );

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_2, matches );

  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_1.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );

  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
  //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
  //-- small)
  //-- PS.- radiusMatch can also be used here.
  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors_1.rows; i++ )
  { if( matches[i].distance <= max(2*min_dist, 0.02) )
    { good_matches.push_back( matches[i]); }
  }

  //-- Draw only "good" matches
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  //-- Show detected matches
  imshow( "Good Matches", img_matches );

  for( int i = 0; i < (int)good_matches.size(); i++ )
  { printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }

  waitKey(0);

  return 0;
}

//#include"Population.h"//Single header
//using namespace pop;//Population namespace

//template<typename Descriptor>
//Vec<DescriptorMatch<Descriptor >   > descriptorMatch2(const Vec<Descriptor > & descriptor1, const Vec<Descriptor > & descriptor2){


//    Vec<DescriptorMatch<Descriptor > > v_match;
//    for(unsigned int j=0;j<descriptor1.size();j++){
//        double distance =1000;
//        int index=-1;
//        for(unsigned int i=0;i<descriptor2.size();i++){
//            const Descriptor& d1=descriptor1[j];
//            const Descriptor& d2=descriptor2[i];
//            double dist_temp = pop::distance(d1.data(),d2.data(),2);
//            if(dist_temp<distance){
//                index = i;
//                distance =dist_temp;
//            }
//        }
//        std::cout<<distance<<std::endl;
//        DescriptorMatch<Descriptor > match;
//        match._d1 = descriptor1[j];
//        match._d2 = descriptor2[index];
//        match._error= distance;
//        v_match.push_back(match);
//    }
//    std::sort(v_match.begin(),v_match.end());
//    return v_match;
//}
//int main(){
//    int number_match_point=30;
//    int min_overlap = 20;
//    Mat2UI8 img1,img2;
//    img1.load("D:/Users/vtariel/Desktop/cni1.jpg");
//    img2.load("D:/Users/vtariel/Desktop/cni4.png");

//    typedef KeyPointPyramid<2> KeyPointAlgo;
//    Pyramid<2,F64> pyramid1 = Feature::pyramidGaussian(img1);
//    Vec<KeyPointAlgo > keypoint1 = Feature::keyPointSIFT(pyramid1);
//    Feature::drawKeyPointsCircle(img1,keypoint1).display("keypoint",false);
//    Vec<Descriptor<KeyPointAlgo > >descriptor1 = Feature::descriptorPieChart(img1,keypoint1);
//    Feature::drawDescriptorArrow(img1,descriptor1).display("descriptor",false);

//    Pyramid<2,F64> pyramid2 = Feature::pyramidGaussian(img2);
//    Vec<KeyPointAlgo > keypoint2 = Feature::keyPointSIFT(pyramid2);

//    Feature::drawKeyPointsCircle(img2,keypoint2).display("keypoint2",false);
//    Vec<Descriptor<KeyPointAlgo > >descriptor2 = Feature::descriptorPieChart(img2,keypoint2);
//    Feature::drawDescriptorArrow(img2,descriptor2).display("descriptor",false);
//    Vec<DescriptorMatch<Descriptor<KeyPointAlgo > > > match = descriptorMatch2(descriptor1,descriptor2);
//    if(number_match_point<match.size())
//        match.erase(match.begin()+number_match_point,match.end());
//    std::cout<<match.size()<<std::endl;
//    match = Feature::descriptorFilterNoOverlap(match,min_overlap);
//    Feature::drawDescriptorMatch(img1,img2,match).display();
//    return 1;
//    Mat2UI8 img;
//    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer
//    img.display("Initial image",false);
//    img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
//    img.display();
//    double value;
//    Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
//    threshold.save("iexthreshold.png");
//    Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
//    color.display("Segmented image",true);
//    return 0;
//}
