#ifndef CONVERTOROPENCV_H
#define CONVERTOROPENCV_H

#include"PopulationConfig.h"
#if defined(HAVE_OPENCV)

#include"data/mat/MatN.h"
#include"opencv2/highgui/highgui.hpp"

namespace pop{


struct ConvertorOpenCVMat
{
    /*!
        \class pop::ConvertorOpenCVMat
        \brief  bridge between opencv and population
        \author Tariel Vincent

         From opencv, the input image
        \code
#include"Population.h"
#include"dependency/ConvertorOpenCV.h"
#include"opencv2/opencv.hpp"
using namespace pop;//Population namespace
int main(){

    cv::Mat opencv_image = cv::imread(POP_PROJECT_SOURCE_DIR+std::string("/image/Lena.bmp"), CV_LOAD_IMAGE_COLOR);
    cv::Mat dest;
    cv::cvtColor(opencv_image, dest,CV_BGR2RGB);
    Mat2RGBUI8 pop_img_color = pop::ConvertorOpenCVMat::fromCVMatRGBUI8ToPopMatRGBUI8(dest);
    pop_img_color.display();
    cv::cvtColor(opencv_image, dest,CV_BGR2GRAY);
    Mat2RGBUI8 pop_img_grey = pop::ConvertorOpenCVMat::fromCVMatUI8ToPopMatUI8(dest);
    pop_img_grey.display();
}
        \endcode
        From Population
        \code
#include"Population.h"
#include"dependency/ConvertorOpenCV.h"
#include"opencv2/opencv.hpp"
using namespace pop;//Population namespace
int main()
{
    Mat2RGBUI8 pop_img_color;
    pop_img_color.load(POP_PROJECT_SOURCE_DIR+std::string("/image/Lena.bmp"));
    cv::Mat cv_img_color = pop::ConvertorOpenCVMat::fromPopMatToCVMat(pop_img_color);
    Mat2UI8 pop_img_grey(pop_img_color);
    cv::Mat cv_img_grey =pop::ConvertorOpenCVMat::fromPopMatToCVMat(pop_img_grey);
    cv::namedWindow( "Color", CV_WINDOW_AUTOSIZE );
    cv::imshow( "Color", cv_img_color );
    cv::namedWindow( "Grey", CV_WINDOW_AUTOSIZE );
    cv::imshow( "Grey", cv_img_grey );
    cv::waitKey(0);
}
         \endcode


    */
    static pop::MatN<2,RGBUI8> fromCVMatRGBUI8ToPopMatRGBUI8(const cv::Mat& mat)throw(pexception);
    static MatN<2,UI8> fromCVMatUI8ToPopMatUI8(const cv::Mat& mat)throw(pexception);

    static cv::Mat fromPopMatToCVMat(const pop::MatN<2,RGBUI8>& mat);

    static cv::Mat fromPopMatToCVMat(const pop::MatN<2,UI8>& mat);
};



}
#endif
#endif // CONVERTOROPENCV_H
