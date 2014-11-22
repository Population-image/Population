#include "OperatorConvertGrey2ColorImageGrid.h"

#include<DataImageGrid.h>
OperatorConvertGrey2ColorMatN::OperatorConvertGrey2ColorMatN()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Format");
    this->setKey("PopulationOperatorConvertGrey2ColorImageGrid");
    this->setName("convertGrey2Color");
    this->setInformation("color(x).r()=grey(x), color(x).g()=grey(x), color(x).b()=grey(x), for grey a color image color =grey\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"grey.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"color.pgm");
}

void OperatorConvertGrey2ColorMatN::exec(){
    shared_ptr<BaseMatN> grey= dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    BaseMatN * color;
    foo func;

    BaseMatN * greyc= grey.get();
    try{
        Dynamic2Static<TListImgGrid1Byte>::Switch(func,greyc,color,Loki::Type2Type<MatN<2,int> >());
        dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(color));
        return;
    }
    catch(pexception msg){
        try{
            Dynamic2Static<TListImgGridRGB>::Switch(func,greyc,color,Loki::Type2Type<MatN<2,int> >());
            dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(color));
            return;
        }
        catch(pexception msg){
            this->error("Pixel/voxel type of input image must be unsigned type used operator Convert1Byte");
            return;
        }
    }

}

COperator * OperatorConvertGrey2ColorMatN::clone(){
    return new OperatorConvertGrey2ColorMatN();
}
