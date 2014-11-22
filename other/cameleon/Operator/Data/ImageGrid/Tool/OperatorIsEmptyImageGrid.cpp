#include "OperatorIsEmptyImageGrid.h"

#include<DataImageGrid.h>
#include<DataBoolean.h>
OperatorIsEmptyMatN::OperatorIsEmptyMatN()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Tool");
    this->setKey("PopulationOperatorIsEmptyImageGrid");
    this->setName("isEmpty");
    this->setInformation("bool =true for forall x in E h(x)=0, false otherwise\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugOut(DataBoolean::KEY,"bool.bool");
}

void OperatorIsEmptyMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    bool isempty;
    foo func;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1,isempty,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataBoolean *>(this->plugOut()[0]->getData())->setValue(isempty);
}

COperator * OperatorIsEmptyMatN::clone(){
    return new OperatorIsEmptyMatN();
}
