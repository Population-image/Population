#include "OperatorCorrelationImageGrid.h"
#include<DataImageGrid.h>
#include<DataNumber.h>
#include<DataMatrix.h>
OperatorCorrelationMatN::OperatorCorrelationMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Geometry");
    this->path().push_back("Statistic");
    this->setKey("PopulationOperatorCorrelationImageGrid");
    this->setName("correlation");
    this->setInformation("P(i,0)=i and P(i,j) = P(f(x) = j-1 and f(x+r)neq j-1) with r any vector of size i. To estimate this value, we sample n*length*dim times.");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"n.num(by default 20000)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"length.num(by default domain(f)/2)");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"P.m");
}
void OperatorCorrelationMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);

    if(this->plugIn()[2]->isConnected()==false)
        this->plugIn()[2]->setState(CPlug::OLD);
    else
        this->plugIn()[2]->setState(CPlug::EMPTY);

    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorCorrelationMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    int number = 20000;
    if(this->plugIn()[1]->isDataAvailable()==true)
        number = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    int length =-1;
    if(this->plugIn()[2]->isDataAvailable()==true)
        number = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    Mat2F64* m;
    foo func;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridScalar>::Switch(func,fc1,number,length,m,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(m));
}

COperator * OperatorCorrelationMatN::clone(){
    return new OperatorCorrelationMatN();
}
