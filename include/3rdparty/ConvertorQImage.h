#ifndef CONVERTORQIMAGE_H
#define CONVERTORQIMAGE_H
#include"PopulationConfig.h"
#if defined(HAVE_QT)
#include"data/mat/MatN.h"
#include<QImage>
#include<QColor>
namespace pop{
class ConvertorQImage
{
private:

    template<pop::I32 DIM,typename Result>
    static MatN<DIM,pop::UI8> fromQImage(const QImage &,Int2Type<DIM>,Type2Type<Result>)
    {
        std::cerr<<"In Convertor::fromQImage, Pixel/voxel type must be pop::UI8 or RGB";
    }


    static MatN<2,pop::UI8> fromQImage(const QImage & qimg,bool isfastconversion,Int2Type<2>,Type2Type<pop::UI8>);
    static MatN<2,RGBUI8> fromQImage(const QImage & qimg,bool isfastconversion,Int2Type<2>,Type2Type<RGBUI8>);
public:
    /*!
    \fn QImage toQImage(const MatN<DIM,Result> & img)
    \param img input population image
    \return QImage QImage object
    \exception  std::string ion Image must be bidimensionnel and its pixel/voxel type must be pop::UI8 or RGB
    *
    * Convert in QImage object
    */
    template<pop::I32 DIM,typename Result>
    static QImage toQImage(const MatN<DIM,Result> & img,bool isfastconversion=false){
        if(DIM!=2)
            std::cerr<<"In Convertor::toQImage, Image must be bidimensionnel";
        else {
            MatN<DIM,pop::UI8> temp (img);
            return ConvertorQImage::toQImage(temp,isfastconversion);
        }
    }
    template<pop::I32 DIM>
    static QImage toQImage(const MatN<DIM,pop::UI8> & img,bool isfastconversion=false){
        if(DIM!=2)
            std::cerr<<"In Convertor::toQImage, Image must be bidimensionnel";
        return toQImage(MatN<DIM,pop::RGBUI8>(img),isfastconversion);
    }
    template<pop::I32 DIM>
    static QImage toQImage(const MatN<DIM,RGBUI8> & img,bool isfastconversion=false){
        if(DIM!=2)
            std::cerr<<"In Convertor::toQImage, Image must be bidimensionnel";
        QImage qimg( img.getDomain()(1),img.getDomain()(0),QImage::Format_RGB888);
        if(isfastconversion==false){
            for (int i = 0; i < qimg.width(); ++i){
                for (int j = 0; j < qimg.height(); ++j){
                    QColor col(img(j,i).r(),img(j,i).g(),img(j,i).b());
                    qimg.setPixel(i,j,col.rgb());
                }
            }
        }else{
            std::copy(reinterpret_cast<const uchar *>(img.data()),reinterpret_cast<const uchar *>(img.data())+3*img.getDomain().multCoordinate(),qimg.bits());
        }

        return qimg;
    }
    /*!
    \fn MatN<DIM,Result> fromQImage(const QImage & qimg)
    \param qimg input QImage object
    \return
    \exception  std::string ion Image must be bidimensionnel, its pixel/voxel type must be pop::UI8 or RGBUI8 and qimg is not empty
    *
    * Convert in QImage object
    */
    template<pop::I32 DIM,typename Result>
    static MatN<DIM,Result> fromQImage(const QImage & qimg,bool isfastconversion=false)
    {
        if(DIM!=2)
            std::cerr<<"In Convertor::fromQImage, Image must be bidimensionnel";
        return ConvertorQImage::fromQImage(qimg,isfastconversion,Int2Type<DIM>(),Type2Type<Result>());
    }


};
}
#endif
#endif // CONVERTORQIMAGE_H
