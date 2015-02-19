#include"Population.h"//Single header
using namespace pop;//Population namespace

template<typename PixelType>
void fill1(MatN<2,PixelType> & m, PixelType value){
    for(unsigned int i =0;i<m.sizeI();i++)
        for(unsigned int j =0;j<m.sizeJ();j++)
            m(i,j)=value;
}
template<typename PixelType>
void fill2(MatN<2,PixelType> & m, PixelType value){
    ForEachDomain2D(x,m){
        m(x)=value;
    }
}
template<int DIM,typename PixelType>
void fill3(MatN<DIM,PixelType> & m, PixelType value){
    typename MatN<DIM,PixelType>::IteratorEDomain it = m.getIteratorEDomain();//in C++11, you can write: auto it =  img.getIteratorEDomain();
    while(it.next()){
        m(it.x())=value;
    }
}
template<int DIM,typename PixelType>
MatN<DIM,PixelType> erosion1(const MatN<DIM,PixelType> & m, F32 radius, int norm){
    MatN<DIM,PixelType> m_ero(m.getDomain());
    typename MatN<DIM,PixelType>::IteratorEDomain it_global = m.getIteratorEDomain();//in C++11, you can write: auto it_global =  m.getIteratorEDomain();
    typename MatN<DIM,PixelType>::IteratorENeighborhood it_local = m.getIteratorENeighborhood(radius,norm);//in C++11, you can write: auto it_local =  m.getIteratorENeighborhood(radius,norm);
    //Global scan
    while(it_global.next()){
        PixelType value = pop::NumericLimits<PixelType>::maximumRange();
        it_local.init(it_global.x());
        //local scan
        while(it_local.next()){
            value = pop::minimum(value,m(it_local.x()));
        }
        m_ero(it_global.x())=value;
    }
    return m_ero;
}
template<int DIM,typename PixelType,typename IteratorEGlobal,typename IteratorELocal>
MatN<DIM,PixelType> erosion2(const MatN<DIM,PixelType> & m, IteratorEGlobal it_global, IteratorELocal it_local){
    MatN<DIM,PixelType> m_ero(m);
    //Global scan
    while(it_global.next()){
        PixelType value = pop::NumericLimits<PixelType>::maximumRange();
        it_local.init(it_global.x());
        //local scan
        while(it_local.next()){
            value = pop::minimum(value,m(it_local.x()));
        }
        m_ero(it_global.x())=value;
    }
    return m_ero;
}

int main()
{
    Mat2UI8 m(4,4);
    fill1(m,UI8(1));
    std::cout<<m<<std::endl;
    fill2(m,UI8(2));
    std::cout<<m<<std::endl;
    Mat3UI8 m3d(4,4,3);//3d matrix
    fill3(m3d,UI8(3));
    std::cout<<m3d<<std::endl;
    m.load(POP_PROJECT_SOURCE_DIR+std::string("/image/Lena.bmp"));
    erosion1(m,5,2).display("erosion1",false);
    erosion2(m,m.getIteratorERectangle(Vec2F32(m.getDomain())*0.25f,Vec2F32(m.getDomain())*0.75f),m.getIteratorENeighborhood(5,2)).display("erosion2");
}
