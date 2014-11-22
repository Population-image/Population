#include "OperatorVERPointPorosityImageGrid.h"

#include "OperatorVERPointPorosityImageGrid.h"

#include<DataImageGrid.h>
#include<DataPoint.h>
#include<DataMatrix.h>
#include<DataNumber.h>
OperatorVERPointPorosityMatN::OperatorVERPointPorosityMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Geometry");
    this->path().push_back("REV");
    this->setKey("PopulationOperatorVERPointPorosityImageGrid");
    this->setName("REVPointPorosity");
    this->setInformation("Caluated the grey-level Porosity inside the ball centered in x by progressively increased the radius from rmin to rmax\nA(i,0)=i and A(i,j) = |Sigma$^j_i$(x)|/|Sigma$^j$(x)| where Sigma$^j$(x)={x' in E: |x'-x|<rmin+j } and Sigma$^j_i$(x)={x' in E: |x'-x|<rmin+j and f(x')=i }\n The distance |.| uses the euclidean norm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataPoint::KEY,"x.v(by default the center of the image)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"rmax.num (by default max radius of a ball centered in x included in the domain of f)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"norm.num(by default 0)");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"A.m");
}

void OperatorVERPointPorosityMatN::initState(){
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
void OperatorVERPointPorosityMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    VecF64  x;
    if(this->plugIn()[1]->isDataAvailable()==true)
        x=dynamic_cast<DataPoint *>(this->plugIn()[1]->getData())->getValue();
    int rmax=2048;
    if(this->plugIn()[2]->isDataAvailable()==true)
        rmax = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();

    int norm=0;
    if(this->plugIn()[3]->isDataAvailable()==true)
        norm = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();
    Mat2F64* m;
    foo func;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridScalar>::Switch(func,fc1,x,rmax,norm,m,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(m));
}

COperator * OperatorVERPointPorosityMatN::clone(){
    return new OperatorVERPointPorosityMatN();
}
