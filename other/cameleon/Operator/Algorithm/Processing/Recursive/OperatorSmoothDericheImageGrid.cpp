#include "OperatorSmoothDericheImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorSmoothDericheMatN::OperatorSmoothDericheMatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Recursive");
    this->setKey("PopulationOperatorSmoothDericheImageGrid");
    this->setName("smoothDeriche");
    this->setInformation("h(x)=smooth(g,sigma) with smooth is the Deriche's smooth operator with parameter sigma\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"sigma.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}








void OperatorSmoothDericheMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();

    double sigma;
    if(this->plugIn()[1]->isDataAvailable()==true){
        sigma = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    }
    else{
        sigma = 1;
    }

    BaseMatN * h;
    foo func;

    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1,sigma,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

void OperatorSmoothDericheMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
COperator * OperatorSmoothDericheMatN::clone(){
    return new OperatorSmoothDericheMatN();
}
