#include "OperatorColorAverageImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
#include<DataBoolean.h>
OperatorColorAverageMatN::OperatorColorAverageMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Visualization");
    this->path().push_back("Label");
    this->setKey("PopulationOperatorColorAverageImageGrid");
    this->setName("labelAverageColor");
    this->setInformation("$h(x) = c_\\{label(x)\\}$ where $c_i = <f>_\\{label(y)=i\\}$");
    this->structurePlug().addPlugIn(DataMatN::KEY,"label.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}



void OperatorColorAverageMatN::exec(){
    shared_ptr<BaseMatN> label = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<BaseMatN> grey = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();


    BaseMatN *color;
    BaseMatN * labelc= label.get();
    BaseMatN * greyc= grey.get();
    foo func;
    try{Dynamic2Static<TListImgGridUnsigned>::Switch(func,labelc,greyc,color,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        if(msg.what()[0]=='P')
            this->error("Pixel/voxel type of input image must be unsigned Byte");
        else
            this->error(msg.what());
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(color));
}

COperator * OperatorColorAverageMatN::clone(){
    return new OperatorColorAverageMatN();
}
