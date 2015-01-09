#include"PopulationConfig.h"

#include"3rdparty/jpgd.h"
#include"3rdparty/jpge.h"
#include"3rdparty/lodepng.h"
#include"3rdparty/bipmap.h"

#include"data/mat/MatNInOut.h"

#if defined(HAVE_CIMG)
#include"3rdparty/ConvertorCImg.h"
#endif

//#include"data/mat/MatN.h"
namespace pop
{
// I used the implementation pattern to hide the CImg header !!!

void  MatNInOut::_save(const MatN<2, UI8 > &img, const char * filename){
    (void)img;
(void)filename;
#if defined(HAVE_CIMG)
    cimg_library::CImg< TypeTraitsTypeScalar<UI8>::Result> temp= ConvertorCImg::toCImg(img);
    temp.save(filename);
#endif
}

void  MatNInOut::_save(const MatN<2, UI16 > &img, const char * filename){
        (void)img;
    (void)filename;
#if defined(HAVE_CIMG)
    cimg_library::CImg< TypeTraitsTypeScalar<UI16>::Result> temp= ConvertorCImg::toCImg(img);
    temp.save(filename);
#endif
}
void  MatNInOut::_save(const MatN<2, UI32 > &img, const char * filename){
    (void)img;
(void)filename;
#if defined(HAVE_CIMG)
    cimg_library::CImg< TypeTraitsTypeScalar<UI32>::Result> temp= ConvertorCImg::toCImg(img);
    temp.save(filename);
#endif
}
void  MatNInOut::_save(const MatN<2, I32 > &img, const char * filename){
    (void)img;
(void)filename;
#if defined(HAVE_CIMG)
    cimg_library::CImg< TypeTraitsTypeScalar<I32>::Result> temp= ConvertorCImg::toCImg(img);
    temp.save(filename);
#endif
}
void  MatNInOut::_save(const MatN<2, F32 > &img, const char * filename){
    (void)img;
(void)filename;
#if defined(HAVE_CIMG)
    cimg_library::CImg< TypeTraitsTypeScalar<F32>::Result> temp=ConvertorCImg::toCImg(img);
    temp.save(filename);
#endif
}
void  MatNInOut::_save(const MatN<2, F64 > &img, const char * filename){
    (void)img;
(void)filename;
#if defined(HAVE_CIMG)
    cimg_library::CImg< TypeTraitsTypeScalar<F64>::Result> temp= ConvertorCImg::toCImg(img);
    temp.save(filename);
#endif
}
void  MatNInOut::_save(const MatN<2, RGBUI8 > &img, const char * filename){
    (void)img;
(void)filename;
#if defined(HAVE_CIMG)
    cimg_library::CImg< TypeTraitsTypeScalar<RGBUI8>::Result> temp= ConvertorCImg::toCImg(img);
    temp.save(filename);
#endif
}


void  MatNInOut::_load( MatN<2, UI8 > &img, const char * filename){
    (void)img;
(void)filename;
#if defined(HAVE_CIMG)
    cimg_library::CImg< TypeTraitsTypeScalar<UI8>::Result> temp;
    temp.load(filename);
    img = ConvertorCImg::fromCImg<2,UI8>(temp);
#endif
}

void  MatNInOut::_load( MatN<2, UI16 > &img, const char * filename){
    (void)img;
(void)filename;
#if defined(HAVE_CIMG)
    cimg_library::CImg< UI16> temp;
    temp.load(filename);
    img = ConvertorCImg::fromCImg<2,UI16>(temp);
#endif
}
void  MatNInOut::_load( MatN<2, UI32 > &img, const char * filename){
    (void)img;
(void)filename;
#if defined(HAVE_CIMG)
    cimg_library::CImg< UI32> temp;
    temp.load(filename);
    img = ConvertorCImg::fromCImg<2,UI32>(temp);
#endif
}
void  MatNInOut::_load( MatN<2, I32 > &img, const char * filename){
    (void)img;
(void)filename;
#if defined(HAVE_CIMG)
    cimg_library::CImg< I32> temp;
    temp.load(filename);
    img = ConvertorCImg::fromCImg<2,I32>(temp);
#endif
}
void  MatNInOut::_load( MatN<2, F32 > &img, const char * filename){
    (void)img;
(void)filename;
#if defined(HAVE_CIMG)
    cimg_library::CImg< F32> temp;
    temp.load(filename);
    img = ConvertorCImg::fromCImg<2,F32>(temp);
#endif
}
void  MatNInOut::_load( MatN<2, F64 > &img, const char * filename){
    (void)img;
(void)filename;
#if defined(HAVE_CIMG)
    cimg_library::CImg< F64> temp;
    temp.load(filename);
    img = ConvertorCImg::fromCImg<2,F64>(temp);
#endif
}
void  MatNInOut::_load( MatN<2, RGBUI8 > &img, const char * filename){
    (void)img;
(void)filename;
#if defined(HAVE_CIMG)
    cimg_library::CImg< TypeTraitsTypeScalar<RGBUI8>::Result> temp;
    temp.load(filename);
    img = ConvertorCImg::fromCImg<2,RGBUI8>(temp);
#endif
}


void  MatNInOut::_savePNG(const MatN<2, UI8 > &img, const char * filename){
    lodepng::encode(filename, img, img._domain(1), img._domain(0),LCT_GREY,8);
}

void  MatNInOut::_savePNG(const MatN<2, RGBUI8 > &img, const char * filename){

    std::vector<unsigned char> byte_array(img._domain(0)* img._domain(1)*3);
    std::vector<unsigned char>::iterator itin =byte_array.begin();
    std::vector<RGBUI8>::const_iterator itout =img.begin();
    for(;itin!=byte_array.end();){
        *itin =itout->r();
        itin++;
        *itin=itout->g();
        itin++;
        *itin=itout->b();
        itin++;
        itout++;
    }
    lodepng::encode(filename, byte_array, img._domain(1), img._domain(0),LCT_RGB,8);
}


void  MatNInOut::_loadPNG( MatN<2, UI8 > &img, const char * filename)
{
    MatN<2, RGBUI8 > img_color;
    _loadPNG(img_color,filename);
    img = img_color;
}
void  MatNInOut::_loadPNG( MatN<2, RGBUI8 > &img, const char * filename)
{
    unsigned width, height;

    std::vector<unsigned char> byte_array;
    lodepng::decode(byte_array, width, height, filename,LCT_RGB,8);
    if(byte_array.size()==0)
        std::cerr<<"In MatN::load, cannot open file: "+std::string(filename);
    img._domain(1)=width;
    img._domain(0)=height;
    img.std::vector<RGBUI8>::resize(width*height);
    std::vector<unsigned char>::iterator itin =byte_array.begin();
    std::vector<RGBUI8>::iterator itout =img.begin();
    for(;itin!=byte_array.end();){
        itout->r()=*itin;
        itin++;
        itout->g()=*itin;
        itin++;
        itout->b()=*itin;
        itin++;
        itout++;
    }
}
void  MatNInOut::_loadBMP( MatN<2, UI8 > &img, const char * filename){
    BIPMAP::CBitmap bimg;
    bimg.Load(filename);
    if(bimg.GetWidth()==0||bimg.GetHeight()==0)
        std::cerr<<"In MatN::load, cannot open file: "+std::string(filename);
    img._domain(1)=bimg.GetWidth();
    img._domain(0)=bimg.GetHeight();
    img.std::vector<UI8>::resize(bimg.GetWidth()*bimg.GetHeight());
    for(unsigned int i=0;i<bimg.GetWidth();i++)
        for(unsigned int j=0;j<bimg.GetHeight();j++){
            img(img.getDomain()(0)-1-j,i) =RGBUI8(static_cast<BIPMAP::RGBA*>(bimg.GetBits())[j*bimg.GetWidth()+i].Red,static_cast<BIPMAP::RGBA*>(bimg.GetBits())[j*bimg.GetWidth()+i].Green,static_cast<BIPMAP::RGBA*>(bimg.GetBits())[j*bimg.GetWidth()+i].Blue).lumi();
        }
}

void  MatNInOut::_loadBMP( MatN<2, RGBUI8 > &img, const char * filename){
    BIPMAP::CBitmap bimg;
    bimg.Load(filename);
    if(bimg.GetWidth()==0||bimg.GetHeight()==0)
        std::cerr<<"In MatN::load, cannot open file: "+std::string(filename);
    img._domain(1)=bimg.GetWidth();
    img._domain(0)=bimg.GetHeight();
    img.std::vector<RGBUI8>::resize(bimg.GetWidth()*bimg.GetHeight());
    for(unsigned int i=0;i<bimg.GetWidth();i++)
        for(unsigned int j=0;j<bimg.GetHeight();j++){
            img(img.getDomain()(0)-1-j,i).r()= static_cast<BIPMAP::RGBA*>(bimg.GetBits())[j*bimg.GetWidth()+i].Red;
            img(img.getDomain()(0)-1-j,i).g()= static_cast<BIPMAP::RGBA*>(bimg.GetBits())[j*bimg.GetWidth()+i].Green;
            img(img.getDomain()(0)-1-j,i).b()= static_cast<BIPMAP::RGBA*>(bimg.GetBits())[j*bimg.GetWidth()+i].Blue;
        }
}
void  MatNInOut::_saveBMP(const MatN<2, RGBUI8 > &img, const char * filename){
    BIPMAP::CBitmap bimg;
    bimg.m_BitmapHeader.Width = img.getDomain()(1);
    bimg.m_BitmapHeader.Height = img.getDomain()(0);
    bimg.m_BitmapHeader.BitCount = 32;
    bimg.m_BitmapHeader.Compression = 3;

    bimg.m_BitmapSize = bimg.GetWidth() * bimg.GetHeight();
    bimg.m_BitmapData = new BIPMAP::RGBA[bimg.m_BitmapSize];


    for(unsigned int i=0;i<bimg.GetWidth();i++)
        for(unsigned int j=0;j<bimg.GetHeight();j++){
            bimg.m_BitmapData[j*bimg.GetWidth()+i].Alpha = 255;
            bimg.m_BitmapData[j*bimg.GetWidth()+i].Red = img(img.getDomain()(0)-1-j,i).r();
            bimg.m_BitmapData[j*bimg.GetWidth()+i].Green = img(img.getDomain()(0)-1-j,i).g();
            bimg.m_BitmapData[j*bimg.GetWidth()+i].Blue = img(img.getDomain()(0)-1-j,i).b();
        }
    bimg.Save(filename);
}

void  MatNInOut::_saveBMP(const MatN<2, UI8 > &img, const char * filename){
    BIPMAP::CBitmap bimg;
    bimg.m_BitmapHeader.Width = img.getDomain()(1);
    bimg.m_BitmapHeader.Height = img.getDomain()(0);
    bimg.m_BitmapHeader.BitCount = 32;
    bimg.m_BitmapHeader.Compression = 3;

    bimg.m_BitmapSize = bimg.GetWidth() * bimg.GetHeight();
    bimg.m_BitmapData = new BIPMAP::RGBA[bimg.m_BitmapSize];


    for(unsigned int i=0;i<bimg.GetWidth();i++)
        for(unsigned int j=0;j<bimg.GetHeight();j++){
            bimg.m_BitmapData[j*bimg.GetWidth()+i].Alpha = 255;

            bimg.m_BitmapData[j*bimg.GetWidth()+i].Red = img(img.getDomain()(0)-1-j,i);
            bimg.m_BitmapData[j*bimg.GetWidth()+i].Green = img(img.getDomain()(0)-1-j,i);
            bimg.m_BitmapData[j*bimg.GetWidth()+i].Blue = img(img.getDomain()(0)-1-j,i);
        }
    bimg.Save(filename);
}


void  MatNInOut::_loadJPG( MatN<2, UI8 > &img, const char * filename){
    int width;
    int height;
    int actual_comps;
    unsigned char * dat = jpgd::decompress_jpeg_image_from_file(filename, &width, &height, &actual_comps, 1);
    if(dat==0)
        std::cerr<<"In MatN::load, cannot open file: "+std::string(filename);
    img._domain(1)=width;
    img._domain(0)=height;
    img.std::vector<UI8>::resize(img._domain[1]*img._domain[0]);
    unsigned char * ptr=dat;
    std::copy(ptr,ptr+img._domain[1]*img._domain[0],img.begin());
    free(dat);
}

void  MatNInOut::_loadJPG( MatN<2, RGBUI8 > &img, const char * filename){
    int width;
    int height;
    int actual_comps;
    unsigned char * dat = jpgd::decompress_jpeg_image_from_file(filename, &width, &height, &actual_comps, 3);
    if(dat==0)
        std::cerr<<"In MatN::load, cannot open file: "+std::string(filename);
    img._domain(1)=width;
    img._domain(0)=height;
    img.std::vector<RGBUI8>::resize(img._domain[0]*img._domain[1]);
    unsigned char * ptr=dat;
    for(std::vector<RGBUI8>::iterator itb = img.begin();itb!=img.end();itb++){
        itb->r()=* ptr;
        ptr++;
        itb->g()=* ptr;
        ptr++;
        itb->b()=* ptr;
        ptr++;
    }
    free(dat);
}
void  MatNInOut::_saveJPG(const MatN<2, RGBUI8 > &img, const char * filename){
    jpge::compress_image_to_jpeg_file(filename,img.getDomain()(1),img.getDomain()(0),3,(unsigned char*) &(*img.begin()));
}

void  MatNInOut::_saveJPG(const MatN<2, UI8 > &img, const char * filename){
    jpge::compress_image_to_jpeg_file(filename,img.getDomain()(1),img.getDomain()(0),1,&(*img.begin()));
}

//#if defined(HAVE_CIMG)

//#elif WITHQT
//QImage qimg=ConvertorQT::toQImage(in);
//if(qimg.save(file)==false){
//    std::string msg="Your image format is not accepted :"+std::string(file)+"\nThe Accepted formats are :\n";
//    QString err;
//    err=msg.c_str();
//    QList<QByteArray> list = QImageWriter::supportedImageFormats ();
//    QList<QByteArray>::Iterator it;
//    err+="*: pgm for a pgm format, you must add the extension .pgm in the suffix name file\n";
//    for(it = list.begin();it!=list.end();it++)
//        err+="*: "+*it+"\n";
//    std::cerr<<err.toStdString();
//}
//#endif



//#if defined(HAVE_CIMG)



//#elif WITHQT
//QImage img(file);
//if(img.isNull()==false){
//    in = ConvertorQT::fromQImage<DIM,Result>(img);
//}
//else{
//std::string msg="Your image is not accepted :"+std::string(file)+"\nThe Accepted formats are :\n";
//QString err;
//err=msg.c_str();
//QList<QByteArray> list = QImageReader::supportedImageFormats ();
//QList<QByteArray>::Iterator it;
//err+="*: pgm for a pgm format, you must add the extension .pgm in the suffix name file\n";
//for(it = list.begin();it!=list.end();it++)
//err+="*: "+*it+"\n";
//std::cerr<<err.toStdString();
//}
//#endif
}
