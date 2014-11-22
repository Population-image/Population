#include "OperatorScaleImageGrid.h"
#include<DataImageGrid.h>
#include<DataPoint.h>
#include<DataNumber.h>
OperatorScaleMatN::OperatorScaleMatN()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Tool");
    this->setKey("PopulationOperatorScaleImageGrid");
    this->setName("scale");
    this->setInformation("h(x)= f(x') with x'(i) = lambda(i) x(i)");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"lambdaX.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"lambdaY.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"lambdaZ.num");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}
void OperatorScaleMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);

    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);

    if(this->plugIn()[2]->isConnected()==false)
        this->plugIn()[2]->setState(CPlug::OLD);
    else
        this->plugIn()[2]->setState(CPlug::EMPTY);

    if(this->plugIn()[3]->isConnected()==false)
        this->plugIn()[3]->setState(CPlug::OLD);
    else
        this->plugIn()[3]->setState(CPlug::EMPTY);



    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorScaleMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    double lambdax=1;
    if(this->plugIn()[1]->isDataAvailable()==true)
        lambdax = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();

    double lambday=1;
    if(this->plugIn()[2]->isDataAvailable()==true)
        lambday = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();

    double lambdaz=1;
    if(this->plugIn()[3]->isDataAvailable()==true)
        lambdaz = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();

    BaseMatN * h;
    foo func;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1,lambdax,lambday,lambdaz,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}


COperator * OperatorScaleMatN::clone(){
    return new OperatorScaleMatN();
}
