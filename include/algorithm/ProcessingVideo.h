#ifndef PROCESSINGVIDEO_H
#define PROCESSINGVIDEO_H
#include<data/mat/MatN.h>
namespace pop
{


template<int DIM,typename TypePixel>
class  TrackingVideoYang
{
    /*!
    \class pop::Feature::TrackingVideoYang
    \brief Implementation of Yang et al Real-time and accurate segmentation of moving objects in dynamic scene
    *
    *
    *
    * \code
    string dir = "/home/vtariel/Bureau/DataBase/Acquisition/record/";
    Vec<string> files = BasicUtility::getFilesInDirectory(dir);
    std::sort(files.begin(),files.end());
    MatNDisplay windows;
    Feature::TrackingVideoYang<Mat2UI8> tracking;
    for(int i=0;i<files.size();i++){
        Mat2RGBUI8 img;
        string path = dir+files[i];

        img.load(path.c_str());
        Mat2UI8 grey(img);
        grey = Draw::mergeTwomatrixHorizontal(grey,tracking.processFrame(grey));
        windows.display(grey);
    }
    * \endcode
    *
    */


public:
    //init the image if the size of the frame changed!!!
    void init(){
        _first =true;
    }
    TrackingVideoYang(){
        init();
    }
    MatN<DIM,TypePixel> processFrame( const MatN<DIM,TypePixel>& in,F32 sigma=2,F32 Tf=3,typename MatN<DIM,TypePixel>::F lambda=10,F32 alpha=0.1,F32 T=10){
        MatN<DIM,TypePixel> img(in);
        img = pop::Processing::smoothGaussian(img,sigma);
        if(!(in.getDomain()==_background.getDomain()))
        {
            _first=true;
        }
        if(_first==true){
            _first = false;
            _background =img;
            _img_old =img;
            _dynamicmatric.resize(img.getDomain());
            _dynamicmatric = 0;
            return _dynamicmatric;
        }else{
            _dynamicMatrixModifiedTaoAlgorithm(img,_img_old,_dynamicmatric,_background,Tf,lambda,alpha);
            MatN<DIM,TypePixel> moving(img.getDomain());
            typename MatN<DIM,TypePixel>::IteratorEDomain it(_background.getIteratorEDomain());
            while(it.next()){
                if(normValue((F32)img(it.x())-_background(it.x()))>T){
                    moving(it.x())=255;
                }
                else
                    moving(it.x())=0;
            }
            _img_old = img;
            return moving;
        }
    }
private:
    MatN<DIM,TypePixel> _dynamicmatric;
    MatN<DIM,TypePixel> _img_old;
    MatN<DIM,TypePixel> _background;
    bool _first ;
    void _dynamicMatrixModifiedTaoAlgorithm(const MatN<DIM,TypePixel> & frame_timet,const MatN<DIM,TypePixel> & frame_timet_minus_deltat,MatN<DIM,TypePixel> & dynamic_matrix,MatN<DIM,TypePixel> &background,F32 Tf=10,typename MatN<DIM,TypePixel>::F lambda=10,F32 alpha=0.2)
    {

        //frameToFrameDifferenceImage
        MatN<DIM,TypePixel> frametoframefifferenceimage(frame_timet);
        typename MatN<DIM,TypePixel>::IteratorEDomain it = frametoframefifferenceimage.getIteratorEDomain();

        while(it.next()){
            F32 value1 = normValue(frame_timet(it.x()));
            F32 value2 = normValue(frame_timet_minus_deltat(it.x()));
            F32 value = value1 - value2;
            if(absolute(value)>Tf){
                dynamic_matrix(it.x())=lambda;
            }
            else{
                dynamic_matrix(it.x())=maximum(dynamic_matrix(it.x())-1,0);
            }
            if(dynamic_matrix(it.x())==0){
                background(it.x())=(1-alpha)*background(it.x())+alpha*frame_timet(it.x());
            }
        }
        //           display2.display(Draw::mergeTwoImageHorizontal(dynamic_matrix*10,background));
    }
};
}

#endif // PROCESSINGVIDEO_H
