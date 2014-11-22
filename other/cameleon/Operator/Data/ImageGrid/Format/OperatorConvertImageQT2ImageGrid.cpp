#include "OperatorConvertImageQT2ImageGrid.h"

#include<DataImageGrid.h>
#include<DataImageQt.h>
#include"algorithm/Visualization.h"
#include"dependency/ConvertorQImage.h"
using namespace pop;
OperatorImageQT2MatN::OperatorImageQT2MatN()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Format");
    this->setKey("PopulationOperatorImageQT2ImageGrid");
    this->setName("convertFromImageQT");
    this->setInformation("h(x)=f(x)\n");
    this->structurePlug().addPlugIn(DataImageQt::KEY,"f.bmp");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorImageQT2MatN::exec(){
    shared_ptr<QImage> f = dynamic_cast<DataImageQt *>(this->plugIn()[0]->getData())->getData();
    QImage * fcast=f.get();
    VecN<2,int> x;

    x(0)=f->width();x(1)=f->height();
    if(f->isGrayscale()==true){
        MatN<2,pop::UI8 > * grey = new MatN<2, pop::UI8 >(x);
        * grey = pop::ConvertorQImage::fromQImage<2,pop::UI8>(*fcast);
        dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(grey));
    }else{
        MatN<2,RGBUI8 > * color = new MatN<2,RGBUI8 >(x);
        * color = pop::ConvertorQImage::fromQImage<2,RGBUI8 >(*fcast);
        dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(color));
    }
}

COperator * OperatorImageQT2MatN::clone(){
    return new OperatorImageQT2MatN();
}
