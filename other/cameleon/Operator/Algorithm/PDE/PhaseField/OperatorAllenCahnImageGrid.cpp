#include "OperatorAllenCahnImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorAllenCahnMatN::OperatorAllenCahnMatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("PDE");
    this->path().push_back("PhaseField");
    this->setKey("PopulationOperatorAllenCahnImageGrid");
    this->setName("allenCahn");
    this->setInformation("Multi phase field for Allen-cahn equation with neuman condition on the boundary\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"labelinit.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"bulk.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"nbstep.num");
    this->structurePlug().addPlugOut(DataMatN::KEY,"label.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"phasefield.pgm");
}
void OperatorAllenCahnMatN::exec(){
    shared_ptr<BaseMatN> labelinit = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<BaseMatN> bulk  = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();
    int nbrstep = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();

    BaseMatN * label;

    BaseMatN * phasefield;
    foo func;

    BaseMatN * phasec= labelinit.get();
    BaseMatN * bulkc= bulk.get();
    try{Dynamic2Static<TListImgGridUnsigned>::Switch(func,phasec,bulkc,nbrstep,label,phasefield,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        if(msg.what()[0]=='P')
            this->error("Pixel/voxel type of input image must be scalar type");
        else
            this->error(msg.what());

        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(label));
    dynamic_cast<DataMatN *>(this->plugOut()[1]->getData())->setData(shared_ptr<BaseMatN>(phasefield));
}

COperator * OperatorAllenCahnMatN::clone(){
    return new OperatorAllenCahnMatN();
}
