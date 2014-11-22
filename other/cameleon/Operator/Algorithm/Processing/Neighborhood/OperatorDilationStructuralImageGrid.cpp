#include "OperatorDilationStructuralImageGrid.h"

#include<DataImageGrid.h>
OperatorDilationStructuralMatN::OperatorDilationStructuralMatN()
    :COperator()
{
        this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Neighborhood");
    this->setKey("PopulationOperatorDilationStructuralImageGrid");
    this->setName("dilationStructuralElement");
    this->setInformation("$h(x)=min_{\\forall x'in N(x) }f(x)$ where N(x)={x'+x:strucelt(x'+xcenter) neq 0}\n for instance, for\n(0,1,0)\\\\\n(1,1,1)=strucelt, we have  N={(0,0),(1,0),(-1,0),(0,1),(0,-1)}\\\\\n(0,1,0)");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"structelt.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}


void OperatorDilationStructuralMatN::exec(){
    shared_ptr<BaseMatN> f = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<BaseMatN> struc = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();

    BaseMatN * h;
    foo func;

    BaseMatN * fc= f.get();
    BaseMatN * strucc= struc.get();
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc,strucc,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type and f and struct_elt must have the same dimension");
        return;
    }

    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorDilationStructuralMatN::clone(){
    return new OperatorDilationStructuralMatN();
}
