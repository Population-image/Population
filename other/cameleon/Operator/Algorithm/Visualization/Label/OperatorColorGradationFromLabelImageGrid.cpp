#include "OperatorColorGradationFromLabelImageGrid.h"

#include<DataImageGrid.h>
OperatorColorGradationFromLabelMatN::OperatorColorGradationFromLabelMatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Visualization");
    this->path().push_back("Label");
    this->setKey("PopulationOperatorColorGradationFromLabelImageGrid");
    this->setName("labelToColorGradation");
    this->setInformation("$h(x) = c_\\{label(x)\\}$ with $c_\\{i=0\\}=0$ and $c_\\{i \\neq 0\\}$=blue*(maxValue(label)-i)/maxValue(label)+red*i/maxValue(label) ");
    this->structurePlug().addPlugIn(DataMatN::KEY,"label.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorColorGradationFromLabelMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    BaseMatN * h;
    foo func;

    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridUnsigned>::Switch(func,fc1,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be integer");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorColorGradationFromLabelMatN::clone(){
    return new OperatorColorGradationFromLabelMatN();
}
