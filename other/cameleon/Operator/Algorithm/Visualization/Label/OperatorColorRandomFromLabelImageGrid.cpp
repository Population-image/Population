#include "OperatorColorRandomFromLabelImageGrid.h"

#include<DataImageGrid.h>
OperatorColorRandomFromLabelMatN::OperatorColorRandomFromLabelMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Visualization");
    this->path().push_back("Label");
    this->setKey("PopulationOperatorColorRandomFromLabelImageGrid");
    this->setName("labelToRandomColor");
    this->setInformation("$h(x) = c_\\{label(x)\\}$ with $c_i$ a collection of random colors\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"label.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}
void OperatorColorRandomFromLabelMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    BaseMatN * h;
    foo func;

    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridUnsigned>::Switch(func,fc1,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorColorRandomFromLabelMatN::clone(){
    return new OperatorColorRandomFromLabelMatN();
}
