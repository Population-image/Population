#include "OperatorRecursiveOrder1ImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorRecursiveOrder1MatN::OperatorRecursiveOrder1MatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Recursive");
    this->setKey("PopulationOperatorRecursiveOrder1ImageGrid");
    this->setName("recursiveOrder1");
    this->setInformation("Apply this formula h(x)= a0 f(x) + a1 f(x-1)  + b h(x-1) and h(x=0) = ab f(x=0)  following the coordinate c and the way w\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"a0.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"a1.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"b.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"ab.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"c.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"w.num");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorRecursiveOrder1MatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    double a_0 = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    double a_1 = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    double b   = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();
    double a_b = dynamic_cast<DataNumber *>(this->plugIn()[4]->getData())->getValue();
    double c   = dynamic_cast<DataNumber *>(this->plugIn()[5]->getData())->getValue();
    double w   = dynamic_cast<DataNumber *>(this->plugIn()[6]->getData())->getValue();

    BaseMatN * h;
    foo func;
    FunctorFilterRecursiveOrder1 func1 (a_0,a_1,b,a_b);
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridScalar>::Switch(func,fc1,func1 ,c,w,   h,Loki::Type2Type<MatN<2,int> >());}
    catch(string msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorRecursiveOrder1MatN::clone(){
    return new OperatorRecursiveOrder1MatN();
}
