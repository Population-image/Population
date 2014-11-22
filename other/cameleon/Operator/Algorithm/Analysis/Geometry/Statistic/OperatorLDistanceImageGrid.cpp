#include "OperatorLDistanceImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
#include<DataMatrix.h>
OperatorLDistanceMatN::OperatorLDistanceMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Geometry");
    this->path().push_back("Statistic");
    this->setKey("PopulationOperatorLDistanceImageGrid");
    this->setName("LDistance");
    this->setInformation("M(i,0)=r and M(i,1)=P(B$_n$(x,r) where P(B$_n$(x,r)) is the probability to the radius of a maximal ball for a random point x in X. The maximal ball is the largest ball and included in X ");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"norm.num (default 1)");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"M.m");
    this->structurePlug().addPlugOut(DataMatN::KEY,"distancemap.pgm");
}
void OperatorLDistanceMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
    this->plugOut()[1]->setState(CPlug::EMPTY);
}

void OperatorLDistanceMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    int norm;
    if(this->plugIn()[1]->isDataAvailable()==true)
        norm = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    else
        norm =1;

    Mat2F64* m;
    foo func;
    BaseMatN * fc1= f1.get();
    BaseMatN * h;
    try{Dynamic2Static<TListImgGrid1Byte>::Switch(func,fc1,norm,m,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be 1Byte");
        return;
    }
    dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(m));
     dynamic_cast<DataMatN *>(this->plugOut()[1]->getData())->setData(shared_ptr<BaseMatN>(h));
}
COperator * OperatorLDistanceMatN::clone(){
    return new OperatorLDistanceMatN();
}
