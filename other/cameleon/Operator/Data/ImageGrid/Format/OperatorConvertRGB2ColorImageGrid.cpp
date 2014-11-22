#include "OperatorConvertRGB2ColorImageGrid.h"
#include<DataImageGrid.h>
OperatorConvertRGB2ColorMatN::OperatorConvertRGB2ColorMatN()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Format");
    this->setKey("PopulationOperatorConvertRGB2ColorImageGrid");
    this->setName("convertRGBToColor");
    this->setInformation("r(x)=color(x).r(),  g(x)=color(x).g(), b(x)=color(x).b()\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"r.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"g.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"b.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"color.pgm");
}

void OperatorConvertRGB2ColorMatN::exec(){
    shared_ptr<BaseMatN> r = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<BaseMatN> g = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();
    shared_ptr<BaseMatN> b = dynamic_cast<DataMatN *>(this->plugIn()[2]->getData())->getData();
    BaseMatN * color;
    foo func;

    BaseMatN * rc= r.get();
    BaseMatN * gc= g.get();
    BaseMatN * bc= b.get();
    try{Dynamic2Static<TListImgGrid1Byte>::Switch(func,rc,gc,bc,color,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        if(msg.what()[0]=='P')
            this->error("Pixel/voxel type of input image must be unsigned type used operator Convert1Byte");
        else
            this->error(msg.what());
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(color));
}

COperator * OperatorConvertRGB2ColorMatN::clone(){
    return new OperatorConvertRGB2ColorMatN();
}
