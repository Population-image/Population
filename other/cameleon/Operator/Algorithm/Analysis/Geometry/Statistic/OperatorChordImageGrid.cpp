#include "OperatorChordImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
#include<DataMatrix.h>

OperatorChordMatN::OperatorChordMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Geometry");
    this->path().push_back("Statistic");
    this->setKey("PopulationOperatorChordImageGrid");
    this->setName("chord");
    this->setInformation("P(i,0)=i and P(i,1) = Proba(|c|=i) with c a random chord, To estimate this value, we random n chord\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"n.num(by default 5000000)");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"P.m");
}
void OperatorChordMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorChordMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    int number =  5000000;
    if(this->plugIn()[1]->isDataAvailable()==true)
        number = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    Mat2F64* m;
    foo func;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridScalar>::Switch(func,fc1,number,m,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(m));
}

COperator * OperatorChordMatN::clone(){
    return new OperatorChordMatN();
}
