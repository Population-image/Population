#include "OperatorVERPointHistogramImageGrid.h"

#include<DataImageGrid.h>
#include<DataPoint.h>
#include<DataMatrix.h>
#include<DataNumber.h>
OperatorVERPointHistogramMatN::OperatorVERPointHistogramMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Geometry");
    this->path().push_back("REV");
    this->setKey("PopulationOperatorVERPointHistogramImageGrid");
    this->setName("REVPointHistogram");
    string str;
    str ="Calculate the grey-level histogram inside the ball centered in x by progressively increased the radius from rmin to rmax\n";
    str +="M(i,0)=i and M(i,j) = |X$_{j-1}$ cap B(x,i)|/|B(x,i)| where B(x,i)={x': |x'-x|<i } the ball centered in x of radius i";
    str +="and X$_j$={x:f(x)=j} the level set of f";
    this->setInformation(str);
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataPoint::KEY,"x.v");
    this->structurePlug().addPlugIn(DataNumber::KEY,"rmax.num (by default max radius of a ball centered in x included in the domain of f)");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"A.m");


}


void OperatorVERPointHistogramMatN::initState(){
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

void OperatorVERPointHistogramMatN::exec(){
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

COperator * OperatorVERPointHistogramMatN::clone(){
    return new OperatorVERPointHistogramMatN();
}
