#include "OperatorGetPointImageGrid.h"
#include<DataImageGrid.h>
#include<DataPoint.h>
#include<DataNumber.h>
OperatorGetPointMatN::OperatorGetPointMatN()
    :COperator(),_color(false){
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Affect");
    this->setKey("PopulationOperatorGetPointImageGrid");
    this->setName("getValue");
    this->setInformation("get the pixel value at the point x (for a scalar image, only the first input is updated for a color image image, the thre output representing rgb are updated\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataPoint::KEY    ,"x.v");
    this->structurePlug().addPlugOut(DataNumber::KEY  ,"scalar.point (or R for a color image)");
    this->structurePlug().addPlugOut(DataNumber::KEY, "G.v");
    this->structurePlug().addPlugOut(DataNumber::KEY ,"B.v");
}


void OperatorGetPointMatN::updateMarkingAfterExecution(){
    if(_color==false){
        this->plugOut()[0]->setState(CPlug::NEW);
        try{
            this->plugOut()[0]->send();
        }
        catch(pexception msg){
            this->error(msg.what());
        }
    }
    if(_color==true){
        this->plugOut()[0]->setState(CPlug::NEW);
        try{
            this->plugOut()[0]->send();
        }
        catch(pexception msg){
            this->error(msg.what());
        }
        this->plugOut()[1]->setState(CPlug::NEW);
        try{
            this->plugOut()[1]->send();
        }
        catch(pexception msg){
            this->error(msg.what());
        }
        this->plugOut()[2]->setState(CPlug::NEW);
        try{
            this->plugOut()[2]->send();
        }
        catch(pexception msg){
            this->error(msg.what());
        }

    }
    //All input plug becomes (NEW,OLD)->OLD
    vector<CPlugIn* >::iterator itin;
    for(itin =_v_plug_in.begin();itin!=_v_plug_in.end();itin++){
        (*itin)->setState(CPlug::OLD);
        (*itin)->send();
    }
    this->updateState();
}

void OperatorGetPointMatN::exec(){
    shared_ptr<BaseMatN> f = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    VecF64   x = dynamic_cast<DataPoint *>(this->plugIn()[1]->getData())->getValue();
    BaseMatN * fc= f.get();
    try{
        double r,g,b;
        foo func;
        Dynamic2Static<TListImgGridRGB>::Switch(func,fc,x,r,g,b,Loki::Type2Type<MatN<2,int> >());
        _color = true;
        dynamic_cast<DataNumber *>(this->plugOut()[0]->getData())->setValue(r);
        dynamic_cast<DataNumber *>(this->plugOut()[1]->getData())->setValue(g);
        dynamic_cast<DataNumber *>(this->plugOut()[2]->getData())->setValue(b);
    }
    catch(string){
        try{
            double scalar;
            foo func;
            Dynamic2Static<TListImgGridScalar>::Switch(func,fc,x,scalar,Loki::Type2Type<MatN<2,int> >());
            _color = false;
            dynamic_cast<DataNumber *>(this->plugOut()[0]->getData())->setValue(scalar);
        }
        catch(string){
            this->error("Pixel/voxel type of input image must be registered type");
            return;
        }
   }
}

COperator * OperatorGetPointMatN::clone(){
    return new OperatorGetPointMatN();
}
