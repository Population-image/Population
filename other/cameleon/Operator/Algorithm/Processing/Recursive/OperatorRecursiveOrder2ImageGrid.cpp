#include "OperatorRecursiveOrder2ImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorRecursiveOrder2MatN::OperatorRecursiveOrder2MatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Recursive");
    this->setKey("PopulationOperatorRecursiveOrder2ImageGrid");
    this->setName("recursiveOrder2");
    this->setInformation("Apply this recursive formula h(x)= a0 f(x) + a1 f(x-1) + a2 f(x-2)  + b1 h(x-1) + b2 h(x-2);\n h(x=0) = $a^{b0}$ f(x=0)\n  h(x=1)= $a^{b1}_0$ f(x) + $a^{b1}_1$ f(x-1)  + $b^{b1}$ h(x-1) \n following the coordinate c and the way w\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"a0.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"a1.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"a2.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"b1.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"b2.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"ab0.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"ab10.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"ab11.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"bb1.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"c.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"w.num");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorRecursiveOrder2MatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    double a_0 = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    double a_1 = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    double a_2 = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();
    double b_1   = dynamic_cast<DataNumber *>(this->plugIn()[4]->getData())->getValue();
    double b_2   = dynamic_cast<DataNumber *>(this->plugIn()[5]->getData())->getValue();
    double a_b0 = dynamic_cast<DataNumber *>(this->plugIn()[6]->getData())->getValue();
    double a_b1_0 = dynamic_cast<DataNumber *>(this->plugIn()[7]->getData())->getValue();
    double a_b1_1 = dynamic_cast<DataNumber *>(this->plugIn()[8]->getData())->getValue();
    double b_b1   = dynamic_cast<DataNumber *>(this->plugIn()[9]->getData())->getValue();
    double c   = dynamic_cast<DataNumber *>(this->plugIn()[10]->getData())->getValue();
    double w   = dynamic_cast<DataNumber *>(this->plugIn()[11]->getData())->getValue();

    BaseMatN * h;
    foo func;
    FunctorFilterRecursiveOrder2 func1 (a_0,a_1,a_2,b_1,b_2,a_b0,a_b1_0,a_b1_1,b_b1);
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridScalar>::Switch(func,fc1,func1 ,c,w,   h,Loki::Type2Type<MatN<2,int> >());}
    catch(string msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorRecursiveOrder2MatN::clone(){
    return new OperatorRecursiveOrder2MatN();
}
