#include "OperatorColorForegroundImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
#include<DataBoolean.h>
OperatorColorForegroundMatN::OperatorColorForegroundMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Visualization");
    this->path().push_back("Label");
    this->setKey("PopulationOperatorColorForegroundImageGrid");
    this->setName("labelForeground");
    this->setInformation("color(x) = ratio*grey(x) + (1-ratio)*c$_\\{label(x)\\}$ with with $c_i$ a collection of random colors and ratio=0.5 as default value");
    this->structurePlug().addPlugIn(DataMatN::KEY,"label.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"grey.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"ratio.num");
    this->structurePlug().addPlugOut(DataMatN::KEY,"color.pgm");
}

void OperatorColorForegroundMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    this->plugIn()[1]->setState(CPlug::EMPTY);
    if(this->plugIn()[2]->isConnected()==false)
        this->plugIn()[2]->setState(CPlug::OLD);
    else
        this->plugIn()[2]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}

void OperatorColorForegroundMatN::exec(){
    shared_ptr<BaseMatN> label = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<BaseMatN> grey = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();
    double ratio=0.5;
    if(this->plugIn()[2]->isDataAvailable()==true)
        ratio = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    BaseMatN *color;
    BaseMatN * labelc= label.get();
    BaseMatN * greyc= grey.get();

    foo func;
    try{Dynamic2Static<TListImgGridUnsigned>::Switch(func,labelc,greyc,ratio,color,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        if(msg.what()[0]=='P')
            this->error("Pixel/voxel type of input image must be unsigned Byte");
        else
            this->error(msg.what());
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(color));
}

COperator * OperatorColorForegroundMatN::clone(){
    return new OperatorColorForegroundMatN();
}
