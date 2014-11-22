#include "OperatorMatheronGranulometryImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
#include<DataMatrix.h>
OperatorMatheronGranulometryMatN::OperatorMatheronGranulometryMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Geometry");
    this->path().push_back("Statistic");
    this->setKey("PopulationOperatorMatheronGranulometryImageGrid");
    this->setName("granulometryMatheron");
    this->setInformation("M(i,0)=i and M(i,1) =|opening(f,norm,i)|, M(i,2)=M(i,1)-M(i-1,1) see granulometry(morphology) in wiki");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"norm.num (default 1)");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"M.m");
        this->structurePlug().addPlugOut(DataMatN::KEY,"granulomap.pgm");
}
void OperatorMatheronGranulometryMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
    this->plugOut()[1]->setState(CPlug::EMPTY);
}
void OperatorMatheronGranulometryMatN::exec(){
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
    try{Dynamic2Static<TListImgGridScalar>::Switch(func,fc1,norm,m,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(m));
     dynamic_cast<DataMatN *>(this->plugOut()[1]->getData())->setData(shared_ptr<BaseMatN>(h));
}
COperator * OperatorMatheronGranulometryMatN::clone(){
    return new OperatorMatheronGranulometryMatN();
}
