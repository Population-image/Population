#include "OperatorSetPointImageGrid.h"

#include "CConnectorDirect.h"
#include<DataImageGrid.h>
#include<DataPoint.h>
#include<DataNumber.h>
OperatorSetPointMatN::OperatorSetPointMatN()
    :COperator(){
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Affect");
    this->setKey("PopulationOperatorSetPointImageGrid");
    this->setName("setValue");
    this->setInformation("Set the pixel value at the point x (for a scalar image, only the first input is updated for a color image image, the thre output representing rgb are updated\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataPoint::KEY    ,"x.v");
    this->structurePlug().addPlugIn(DataNumber::KEY  ,"scalar.point (or R for a color image)");
    this->structurePlug().addPlugIn(DataNumber::KEY, "G.v");
    this->structurePlug().addPlugIn(DataNumber::KEY ,"B.v");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}
void OperatorSetPointMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugIn()[2]->setState(CPlug::EMPTY);
    if(this->plugIn()[3]->isConnected()==false)
        this->plugIn()[3]->setState(CPlug::OLD);
    else
        this->plugIn()[3]->setState(CPlug::EMPTY);
    if(this->plugIn()[4]->isConnected()==false)
        this->plugIn()[4]->setState(CPlug::OLD);
    else
        this->plugIn()[4]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}

void OperatorSetPointMatN::exec(){
    shared_ptr<BaseMatN> f = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    VecF64   x = dynamic_cast<DataPoint *>(this->plugIn()[1]->getData())->getValue();
    double scalar = 0;
    if(this->plugIn()[2]->isDataAvailable()==true){
        scalar = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    }
    double green=0;
    if(this->plugIn()[3]->isDataAvailable()==true){
        green = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();
    }
    double blue=0;
    if(this->plugIn()[4]->isDataAvailable()==true){
        blue = dynamic_cast<DataNumber *>(this->plugIn()[4]->getData())->getValue();
    }

    BaseMatN * fc= f.get();
    try{
        foo func;
        Dynamic2Static<TListImgGridRGB>::Switch(func,fc,x,scalar,green,blue,Loki::Type2Type<MatN<2,int> >());
        dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(f);
    }
    catch(string){
        try{
            foo func;
            Dynamic2Static<TListImgGridScalar>::Switch(func,fc,x,scalar,Loki::Type2Type<MatN<2,int> >());
            dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(f);
        }
        catch(string){
            this->error("Pixel/voxel type of input image must be registered type");
            return;
        }
   }
}

COperator * OperatorSetPointMatN::clone(){
    return new OperatorSetPointMatN();
}
