#include "OperatorConvertColor2RGBImageGrid.h"

#include<DataImageGrid.h>
OperatorConvertColor2RGBMatN::OperatorConvertColor2RGBMatN()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Format");
    this->setKey("PopulationOperatorConvertColor2RGBImageGrid");
    this->setName("convertColor2RGB");
    this->setInformation("r(x)=color(x).r(),  g(x)=color(x).g(), b(x)=color(x).b()\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"color.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"r.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"g.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"b.pgm");
}

void OperatorConvertColor2RGBMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    BaseMatN * r;
    BaseMatN * g;
    BaseMatN * b;
    foo func;

    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridRGB>::Switch(func,fc1,r,g,b,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(r));
    dynamic_cast<DataMatN *>(this->plugOut()[1]->getData())->setData(shared_ptr<BaseMatN>(g));
    dynamic_cast<DataMatN *>(this->plugOut()[2]->getData())->setData(shared_ptr<BaseMatN>(b));
}

COperator * OperatorConvertColor2RGBMatN::clone(){
    return new OperatorConvertColor2RGBMatN();
}
