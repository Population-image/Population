#include "OperatorBinaryFromLabelSelectionImageGrid.h"
#include "OperatorBinaryFromLabelSelectionImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorBinaryFromLabelSelectionMatN::OperatorBinaryFromLabelSelectionMatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Seed");
    this->setKey("PopulationOperatorBinaryFromLabelSelectionImageGrid");
    this->setName("labelFromValue");
    this->setInformation("h(x)=255 for f(x)=v1, 0 otherwise where 256 is the default value for v2 \n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"v1.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorBinaryFromLabelSelectionMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    double v1 = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    double v2 =v1+1;


    BaseMatN * h;
    foo func;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1,v1,v2,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorBinaryFromLabelSelectionMatN::clone(){
    return new OperatorBinaryFromLabelSelectionMatN();
}
