#include "dependency/ConvertorQImage.h"
#include"PopulationConfig.h"
#if defined(HAVE_QT)
namespace pop{
MatN<2,pop::UI8> ConvertorQImage::fromQImage(const QImage & qimg,bool isfastconversion,Loki::Int2Type<2>,Loki::Type2Type<pop::UI8>)
{

    if(qimg.isNull()==false){
        MatN<2,pop::UI8> img(qimg.height(),qimg.width());
        if(qimg.format()==QImage::Format_Indexed8&&isfastconversion==true){
            std::copy(qimg.bits(),qimg.bits()+qimg.width()*qimg.height(),img.begin());
            return img;
        }else{
            for (int i = 0; i < qimg.width(); ++i)
            {
                for (int j = 0; j < qimg.height(); ++j)
                {
                    QRgb col = qimg.pixel(i, j);
                    int  gray = qGray(col);
                    img(j,i)=gray;
                }
            }

            return img;
        }
    }else{
        std::cerr<<"In Convertor::fromQImage, QIMage is null";
        return MatN<2,pop::UI8>();
    }
}
MatN<2,RGBUI8> ConvertorQImage::fromQImage(const QImage & temp,bool isfastconversion,Loki::Int2Type<2>,Loki::Type2Type<RGBUI8>)
{
    if(temp.isNull()==false){
        if(isfastconversion==true){
            QImage qimg = temp.convertToFormat(QImage::Format_RGB888);
            MatN<2,RGBUI8> img(qimg.height(),qimg.width());
            unsigned char * data=reinterpret_cast< unsigned char *>(img.data());
            std::copy(qimg.bits(),qimg.bits()+3*qimg.width()*qimg.height(),data);
            return img;
        }else{
            MatN<2,RGBUI8> img(temp.height(),temp.width());
            for (int i = 0; i < temp.width(); ++i){
                for (int j = 0; j < temp.height(); ++j){
                    QRgb col = temp.pixel(i, j);
                    img(j,i)=RGBUI8(qRed(col),qGreen(col),qBlue(col));
                }
            }
            return img;
        }
    }else{
        std::cerr<<"In Convertor::fromQImage, QIMage is null";
        return MatN<2,RGBUI8>();
    }
}

}
#endif
