/******************************************************************************\
|*                   Population library for C++ X.X.X                         *|
|*----------------------------------------------------------------------------*|
The Population License is similar to the MIT license in adding this clause:
for any writing public or private that has resulted from the use of the
software population, the reference of this book "Population library, 2012,
Vincent Tariel" shall be included in it.

So, the terms of the Population License are:

Copyright © 2012-2015, Tariel Vincent

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software and for any writing
public or private that has resulted from the use of the software population,
the reference of this book "Population library, 2012, Vincent Tariel" shall
be included in it.

The Software is provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising
from, out of or in connection with the software or the use or other dealings
in the Software.
\***************************************************************************/
#ifndef CONVERTORCIMG_H
#define CONVERTORCIMG_H
#include"PopulationConfig.h"
#if defined(HAVE_CIMG)
#define UNICODE 1


#include"data/mat/MatN.h"
#include"algorithm/FunctionProcedureFunctorF.h"
#include"dependency/CImg.h"

namespace pop
{

//#include "opencv2/imgproc/imgproc.hpp" //for image processing
//#include "opencv2/highgui/highgui.hpp" //for GUI
//template<typename T>
//void convert(const cv::Mat & M, MatN<2,T>& img ){
//    img.resize(M.cols,M.rows);


//    for(int i = 0; i < M.rows; i++)
//    {
//        for(int j = 0; j < M.cols; j++){


//            if(typeid(T)!=typeid(RGB)){
//                if(CV_MAT_TYPE(M.type()) == CV_8UC1){
//                    img.operator ()(j,i)=M.at<uchar>(i,j);
//                }
//                else if(CV_MAT_TYPE(M.type()) == CV_8UC3){
//                    cv::Vec3b intensity = M.at<cv::Vec3b>(i, j);
//                    img.operator ()(j,i)=0.299*intensity[0]+ 0.587*intensity[1] + 0.114*intensity[2]+0.000001;
//                }
//            }
//        }
//    }
//}

namespace Private{
template<int DIM, typename Result>
struct convertCImg
{
    static cimg_library::CImg<typename TypeTraitsTypeScalar<Result>::Result> toCimg(const MatN<DIM,Result> ){
        return cimg_library::CImg<typename TypeTraitsTypeScalar<Result>::Result>();
    }

     static MatN<DIM,Result> fromCimg(const  cimg_library::CImg<Result> &){
         return MatN<DIM,Result>();
     }

};
template<typename Result>
struct convertCImg<2,Result>
{

    static cimg_library::CImg<typename TypeTraitsTypeScalar<Result>::Result> toCimg(const MatN<2,Result> img){
        cimg_library::CImg<typename TypeTraitsTypeScalar<Result>::Result> temp(img.getDomain()[0],img.getDomain()[1]);

        typename MatN<2,Result>::IteratorEDomain it (img.getIteratorEDomain());

        FunctorF::FunctorAccumulatorMin<Result > funcmini;
        it.init();
        typename TypeTraitsTypeScalar<Result>::Result min = normValue(FunctionProcedureFunctorAccumulatorF(img,funcmini,it));
        FunctorF::FunctorAccumulatorMax<Result > funcmaxi;
        it.init();
        typename TypeTraitsTypeScalar<Result>::Result max = normValue(FunctionProcedureFunctorAccumulatorF(img,funcmaxi,it));
        for(int i =0;i<img.getDomain()[0];i++)
            for(int j =0;j<img.getDomain()[1];j++){
                if(typeid(Result)==typeid(UI8))
                    temp.operator ()(i,j)= normValue(img.operator ()(i,j));
                else
                    temp.operator ()(i,j)= normValue( (img.operator ()(i,j)-min)*255./(max-min));
            }
        return temp;
    }

    static MatN<2,Result> fromCimg(const  cimg_library::CImg<Result> &img){
        MatN<2,Result>  temp(static_cast<unsigned int>(img.width()),static_cast<unsigned int>(img.height()));
        for(int i =0;i<temp.getDomain()[0];i++)
            for(int j =0;j<temp.getDomain()[1];j++){
                if(img.spectrum ()==3){
                    temp.operator ()(i,j) =  0.299*img.operator ()(i,j,0,0) + 0.587*img.operator ()(i,j,0,1) + 0.114*img.operator ()(i,j,0,2)+0.000001;
                }else if(img.spectrum ()==1){
                    temp.operator ()(i,j) = img.operator ()(i,j);
                }
            }
        return temp;
    }

};
template<>
struct convertCImg<2,RGBUI8>
{

    static cimg_library::CImg<UI8> toCimg(const MatN<2,RGBUI8> img){
        cimg_library::CImg<UI8> temp(img.getDomain()[0],img.getDomain()[1],1,3);
        for(int i =0;i<img.getDomain()[0];i++)
            for(int j =0;j<img.getDomain()[1];j++){
                temp.operator ()(i,j,0,0)=img.operator ()(i,j).r();
                temp.operator ()(i,j,0,1)=img.operator ()(i,j).g();
                temp.operator ()(i,j,0,2)=img.operator ()(i,j).b();
            }
        return temp;
    }
    static MatN<2,RGBUI8> fromCimg(const  cimg_library::CImg<UI8> &img){
        MatN<2,RGBUI8>  temp(img.width(),img.height());
        for(int i =0;i<temp.getDomain()[0];i++)
            for(int j =0;j<temp.getDomain()[1];j++){
                if(img.spectrum ()==3){
                    temp.operator ()(i,j).r() = img.operator ()(i,j,0,0);
                    temp.operator ()(i,j).g() = img.operator ()(i,j,0,1);
                    temp.operator ()(i,j).b() = img.operator ()(i,j,0,2);
                }else if(img.spectrum ()==1){
                    temp.operator ()(i,j).r() = img.operator ()(i,j);
                    temp.operator ()(i,j).g() = img.operator ()(i,j);
                    temp.operator ()(i,j).b() = img.operator ()(i,j);
                }
            }
        return temp;
    }

};
template<typename Result>
struct convertCImg<3,Result>
{

    static cimg_library::CImg<typename TypeTraitsTypeScalar<Result>::Result> toCimg(const MatN<3,Result> img){
        cimg_library::CImg<typename TypeTraitsTypeScalar<Result>::Result> temp(img.getDomain()[0],img.getDomain()[1],img.getDomain()[2]);
        for(int i =0;i<img.getDomain()[0];i++)
            for(int j =0;j<img.getDomain()[1];j++)
                for(int k =0;k<img.getDomain()[2];k++)
                    temp.operator ()(i,j,k)=normValue(img.operator ()(i,j,k));
        return temp;
    }
    static MatN<3,Result> fromCimg(const  cimg_library::CImg<Result> &img){
        MatN<3,Result>  temp(img.width(),img.height(),img.depth());
        for(int i =0;i<temp.getDomain()[0];i++)
            for(int j =0;j<temp.getDomain()[1];j++)
                for(int k =0;k<temp.getDomain()[2];k++)
                    temp.operator ()(i,j,k)=img.operator ()(i,j,k);
        return temp;
    }
};
template<>
struct convertCImg<3,RGBUI8>
{

    static cimg_library::CImg< TypeTraitsTypeScalar<RGBUI8>::Result> toCimg(const MatN<3,RGBUI8> img){
        cimg_library::CImg< TypeTraitsTypeScalar<RGBUI8>::Result> temp(img.getDomain()[0],img.getDomain()[1],img.getDomain()[2],3);
        for(int i =0;i<img.getDomain()[0];i++)
            for(int j =0;j<img.getDomain()[1];j++)
                for(int k =0;k<img.getDomain()[2];k++)
                {
                    temp.operator ()(i,j,k,0)=img.operator ()(i,j,k).r();
                    temp.operator ()(i,j,k,1)=img.operator ()(i,j,k).g();
                    temp.operator ()(i,j,k,2)=img.operator ()(i,j,k).b();
                }
        return temp;
    }
    static MatN<3,RGBUI8> fromCimg(const  cimg_library::CImg<UI8> &img){
        MatN<3,RGBUI8>  temp(img.width(),img.height(),img.depth());
        for(int i =0;i<temp.getDomain()[0];i++)
            for(int j =0;j<temp.getDomain()[1];j++)
            for(int k =0;k<temp.getDomain()[2];k++)
            {
                temp.operator ()(i,j,k).r() = img.operator ()(i,j,k,0);
                temp.operator ()(i,j,k).g() = img.operator ()(i,j,k,1);
                temp.operator ()(i,j,k).b() = img.operator ()(i,j,k,2);
            }
        return temp;
    }
};

}

struct ConvertorCImg
{
    /*!
    \fn  cimg_library::CImg<typename TypeTraitsTypeScalar<F>::Result> toCImg()const
    \return CImg data structure
    *
    *  The CIMG library is extraordinary  http://cimg.sourceforge.net/. Just for CImgDisplay, you should use it !!!
    */
    template<I32 DIM,typename Result>
    static cimg_library::CImg<typename TypeTraitsTypeScalar<Result>::Result> toCImg(const MatN<DIM,Result> & img){
        return Private::convertCImg<DIM,Result>::toCimg(img);
    }
    /*!
    \fn  void fromCImg(const cimg_library::CImg<typename TypeTraitsTypeScalar<F>::Result>& img)
    \param img CIMG image
    *
    *  Come back to my library also a good one ;)
    */
    template<I32 DIM,typename Result>
    static MatN<DIM,Result> fromCImg(const cimg_library::CImg<typename TypeTraitsTypeScalar<Result>::Result>& img){
        MatN<DIM,Result> temp = Private::convertCImg<DIM,Result>::fromCimg(img);
        return temp;
    }
    /*!
    \fn  void fromCImg(const cimg_library::CImg<typename TypeTraitsTypeScalar<F>::Result>& img)
    \param img CIMG image
    *
    *  Come back to my library also a good one ;)
    */
    template<I32 DIM,typename Result>
    static void fromCImg(const cimg_library::CImg<typename TypeTraitsTypeScalar<Result>::Result>& cimg,MatN<DIM,Result>& img){
        img = Private::convertCImg<DIM,Result>::fromCimg(cimg);
    }
};
}
#endif
#endif // CONVERTORCIMG_H
