#include "OperatorMinuxImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorMinusMatN::OperatorMinusMatN()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Arithmetic");
    this->setKey("PopulationOperatorMinusImageGrid");
    this->setName("opposite");
    this->setInformation("h=value-f1 (by default the value is equal to maxValue of the pixel range)");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f1.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"value.num");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}
void OperatorMinusMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorMinusMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();

    if(this->plugIn()[1]->isDataAvailable()==true){
        double value = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
        BaseMatN * h;
         foo func;
         BaseMatN * fc1= f1.get();
         try{Dynamic2Static<TListImgGridScalar>::Switch(func,fc1,value,h,Loki::Type2Type<MatN<2,int> >());}
         catch(pexception msg){
             this->error("Pixel/voxel type of input image must be scalar type");
             return;
         }
         dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
    }else{
        BaseMatN * h;
         foo func;
         BaseMatN * fc1= f1.get();
         try{Dynamic2Static<TListImgGrid>::Switch(func,fc1,h,Loki::Type2Type<MatN<2,int> >());}
         catch(pexception msg){
             this->error("Pixel/voxel type of input image must be registered type");
             return;
         }
         dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));

    }




}

COperator * OperatorMinusMatN::clone(){
    return new OperatorMinusMatN();
}
