#include "OperatorPermeabilityImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
#include<DataPoint.h>
#include <CConnectorDirect.h>
OperatorpermeabilityMatN::OperatorpermeabilityMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Physical");
    this->setKey("PopulationOperatorpermeabilityImageGrid");
    this->setName("permeability");
    this->setInformation("Apparent Permeability of the bulk={x:b(x)neq 0} in the given direction where delta x is the lenght of the image in the given direction (delta P =1 and mu = 1), the output images are the velocity field in the corresponding direction\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"b.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"direction.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"error.num(by default 0.01)");
    this->structurePlug().addPlugOut(DataPoint::KEY,"k.vec");
    this->structurePlug().addPlugOut(DataMatN::KEY,"vx.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"vy.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"vz.pgm");
}
void OperatorpermeabilityMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    this->plugIn()[1]->setState(CPlug::EMPTY);
    if(this->plugIn()[2]->isConnected()==false)
        this->plugIn()[2]->setState(CPlug::OLD);
    else
        this->plugIn()[2]->setState(CPlug::EMPTY);


    this->plugOut()[0]->setState(CPlug::EMPTY);
    this->plugOut()[1]->setState(CPlug::EMPTY);
    this->plugOut()[2]->setState(CPlug::EMPTY);
    this->plugOut()[3]->setState(CPlug::EMPTY);
}
void OperatorpermeabilityMatN::updateMarkingAfterExecution(){
    vector<CPlugOut* >::iterator itout;
    //All ouput plug becomes NEW
    int i =0;
    for(itout =_v_plug_out.begin();itout!=_v_plug_out.end();itout++){
        if(i!=3||dim==3){
            (*itout)->setState(CPlug::NEW);
            try{
                (*itout)->send();
            }
            catch(pexception msg){
                this->error(msg.what());
            }
        }
        i++;
    }
    //All input plug becomes (NEW,OLD)->OLD
    vector<CPlugIn* >::iterator itin;
    for(itin =_v_plug_in.begin();itin!=_v_plug_in.end();itin++){
        (*itin)->setState(CPlug::OLD);
        (*itin)->send();
    }
    this->updateState();
}

void OperatorpermeabilityMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    int direction =  dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();

    double error=0.01;
    if(this->plugIn()[2]->isDataAvailable()==true)
        error = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();



    VecF64 m;
    foo func;
    BaseMatN * fc1= f1.get();
    BaseMatN * vx,*vy,*vz;
    try{Dynamic2Static<TListImgGrid1Byte>::Switch(func, fc1, direction,error, m, vx, vy, vz,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be unsigned type used operator Convert1Byte");
        return;
    }
    dynamic_cast<DataPoint*>(this->plugOut()[0]->getData())->setValue(m);
    dynamic_cast<DataMatN *>(this->plugOut()[1]->getData())->setData(shared_ptr<BaseMatN>(vx));
    dynamic_cast<DataMatN *>(this->plugOut()[2]->getData())->setData(shared_ptr<BaseMatN>(vy));
    if(vz!=NULL)
    {
        dim = 3;
        dynamic_cast<DataMatN *>(this->plugOut()[3]->getData())->setData(shared_ptr<BaseMatN>(vz));
    }else
        dim = 2;
}
COperator * OperatorpermeabilityMatN::clone(){
    return new OperatorpermeabilityMatN();
}
