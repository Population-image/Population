#include "OperatorConvertImageGrid2ImageQT.h"

#include<DataImageGrid.h>
#include<DataImageQt.h>
#include<QColor>
OperatorMatN2ImageQT::OperatorMatN2ImageQT()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Format");
    this->setKey("PopulationOperatorImageGrid2ImageQTImageGrid");
    this->setName("convertToImageQT");
    this->setInformation("h(x)=f(x)\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"in.pgm");
    this->structurePlug().addPlugOut(DataImageQt::KEY,"out.bmp");
}

void OperatorMatN2ImageQT::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    foo func;
    BaseMatN * fc1= f1.get();
    typedef FilterKeepTlistTlist<TListImgGrid,0,Loki::Int2Type<2> >::Result ListFilter;
    try{Dynamic2Static<ListFilter>::Switch(func,fc1,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Input image must be 2D, used pattern");
        return;
    }
    if( MatN<2,RGBUI8 > * color =  dynamic_cast<MatN<2,RGBUI8 > * >(f1.get())  ){

        shared_ptr<QImage> h (new QImage(color->getDomain()[0], color->getDomain()[1], QImage::Format_RGB32));
        for(int i =0;i<h->width();i++){
            for(int j =0;j<h->height();j++){
                QColor c ;
                VecN<2,int> x;
                x(0)=i;
                x(1)=j;
                c.setRed( color->operator ()(x).r());
                c.setGreen(color->operator ()(x).g());
                c.setBlue(color->operator ()(x).b());
                h->setPixel(i,j,c.rgb());
            }
        }
        dynamic_cast<DataImageQt *>(this->plugOut()[0]->getData())->setData(shared_ptr<QImage>(h));
    }
    else{
        shared_ptr<QImage> h (new QImage( dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getFile().c_str()));

        dynamic_cast<DataImageQt *>(this->plugOut()[0]->getData())->setData(shared_ptr<QImage>(h));
    }

}

COperator * OperatorMatN2ImageQT::clone(){
    return new OperatorMatN2ImageQT();
}
