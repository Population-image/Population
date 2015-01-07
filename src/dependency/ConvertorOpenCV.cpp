#include "dependency/ConvertorOpenCV.h"
#include"PopulationConfig.h"
#if defined(HAVE_OPENCV)
#include "opencv2/imgproc/imgproc.hpp"
namespace pop{
MatN<2,RGBUI8> ConvertorOpenCVMat::fromCVMatRGBUI8ToPopMatRGBUI8(const cv::Mat& mat)
{
    if(mat.type()!=CV_8UC3){
        std::cerr<<"ConvertorOpenCV::fromCVMatRGBUI8ToPopMatRGBUI8, your opencv image is not a RGB image with 1 byte per channel use cv::cvtColor";
    }
    MatN<2,RGBUI8> m(mat.rows,mat.cols);
    if(3*mat.cols==(int)mat.step)
        std::copy(mat.data,mat.data+3*mat.rows*mat.cols,reinterpret_cast<uchar*>(&(*m.begin())));
    else{
        for(int j = 0;j < mat.cols;j++){
            std::copy(mat.data+j*mat.step,mat.data+j*mat.step+j*mat.cols,reinterpret_cast<uchar*>(&(*m.begin())));
        }
    }
    return m;
}
MatN<2,UI8> ConvertorOpenCVMat::fromCVMatUI8ToPopMatUI8(const cv::Mat& mat)
{

    if(mat.type()!=CV_8UC1){
        std::cerr<<"ConvertorOpenCV::fromCVMatUI8ToPopMatUI8, your opencv image is not a grey image in 1 byte use cv::cvtColor";
    }

    MatN<2,UI8> m(mat.rows,mat.cols);

    if(mat.cols==(int)mat.step)
        std::copy(mat.data,mat.data+mat.rows*mat.cols,reinterpret_cast<uchar*>(m.data()));
    else{
        for(int j = 0;j < mat.cols;j++){
            std::copy(mat.data+j*mat.step,mat.data+j*mat.step+j*mat.cols,reinterpret_cast<uchar*>(m.data()));
        }
    }
    return m;
}
cv::Mat  ConvertorOpenCVMat::fromPopMatToCVMat(const MatN<2,RGBUI8> &mat){
    cv::Mat mat_cv(mat.sizeI(),mat.sizeJ(),CV_8UC3);
    const uchar* ptrbegin = reinterpret_cast<const uchar*>(mat.data());
    const uchar* ptrend = reinterpret_cast<const uchar*>(mat.data())+3*mat.sizeI()*mat.sizeJ();
    std::copy(ptrbegin,ptrend,mat_cv.data);
    cv::cvtColor(mat_cv, mat_cv,CV_RGB2BGR);
    return mat_cv;

}
cv::Mat  ConvertorOpenCVMat::fromPopMatToCVMat(const MatN<2,UI8> &mat){
    cv::Mat mat_cv(mat.sizeI(),mat.sizeJ(),CV_8UC1);
    const uchar* ptrbegin = reinterpret_cast<const uchar*>(mat.data());
    const uchar* ptrend = reinterpret_cast<const uchar*>(mat.data())+mat.sizeI()*mat.sizeJ();
    std::copy(ptrbegin,ptrend,mat_cv.data);
    return mat_cv;

}

}
#endif
