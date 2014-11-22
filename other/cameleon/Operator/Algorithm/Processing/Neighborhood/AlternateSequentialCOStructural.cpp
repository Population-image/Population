#include "AlternateSequentialCOStructural.h"
#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorAlternateSequentialCOStructuralMatN::OperatorAlternateSequentialCOStructuralMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Neighborhood");
    this->setKey("PopulationOperatorAlternateSequentialCOStructuralImageGrid");
    this->setName("alternateSequentialCOStructuralElement");
    this->setInformation("h(x)=Closing(Opening(....Closing(Opening(h))...)) by progessively dilated by the current structural element by the original one\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"maxradius.num");
    this->structurePlug().addPlugIn(DataMatN::KEY,"structelt.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}


void OperatorAlternateSequentialCOStructuralMatN::exec(){
    shared_ptr<BaseMatN> f = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();

    double r = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    shared_ptr<BaseMatN> struc = dynamic_cast<DataMatN *>(this->plugIn()[2]->getData())->getData();

    BaseMatN * h;
    foo func;

    BaseMatN * fc= f.get();
    BaseMatN * strucc= struc.get();
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc,r,strucc,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorAlternateSequentialCOStructuralMatN::clone(){
    return new OperatorAlternateSequentialCOStructuralMatN();
}

