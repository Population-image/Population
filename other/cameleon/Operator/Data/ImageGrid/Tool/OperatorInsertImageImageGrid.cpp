#include "OperatorInsertImageImageGrid.h"

#include "OperatorInsertImageImageGrid.h"


#include<DataImageGrid.h>
#include<DataPoint.h>
#include<DataNumber.h>
OperatorInsertImageMatN::OperatorInsertImageMatN()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Tool");
    this->setKey("PopulationOperatorInsertImageImageGrid");
    this->setName("insertImage");
//    this->setInformation("h(x)= postit(x') for x' in domain(postit), f(x) otherwise with x= R_theta*(x'+t) \n");
        this->setInformation("h(x)= postit(x') for x' in domain(postit), f(x) otherwise with x= x'+t) \n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"postit.pgm");
    this->structurePlug().addPlugIn(DataPoint::KEY,"t.v");
//    this->structurePlug().addPlugIn(DataNumber::KEY,"theta.num");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorInsertImageMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<BaseMatN> f2 = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();
    VecF64  x =    dynamic_cast<DataPoint *>(this->plugIn()[2]->getData())->getValue();
//    double rot = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();

    BaseMatN * h;
    foo func;
    BaseMatN * fc1= f1.get();
    BaseMatN * fc2= f2.get();

//    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1,fc2,x,rot,h,Loki::Type2Type<MatN<2,int> >());}
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1,fc2,x,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        if(msg.what()[0]=='P')
            this->error("Pixel/voxel type of input image must be unsigned type used operator Convert1Byte");
        else
            this->error(msg.what());
        return;
    }

    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorInsertImageMatN::clone(){
    return new OperatorInsertImageMatN();
}

