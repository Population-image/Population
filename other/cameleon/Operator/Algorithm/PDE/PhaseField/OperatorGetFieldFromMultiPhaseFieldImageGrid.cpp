#include "OperatorGetFieldFromMultiPhaseFieldImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorGetFieldFromMultiPhaseFieldMatN::OperatorGetFieldFromMultiPhaseFieldMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("PDE");
    this->path().push_back("PhaseField");
    this->setKey("PopulationOperatorGetFieldFromMultiPhaseFieldImageGrid");
    this->setName("getFieldFromMultiPhaseField");
    this->setInformation("Multi phase field for Allen-cahn equation with neuman condition on the boundary\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"scalarfield_multiphasefield.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"label_multiphasefield.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"label.num (by default 0)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"interfacewidthd.num(by default 5)");
    this->structurePlug().addPlugIn(DataMatN::KEY,"bulk.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"scalarfield_singlephasefield.pgm");
}
void OperatorGetFieldFromMultiPhaseFieldMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    this->plugIn()[1]->setState(CPlug::EMPTY);
    if(this->plugIn()[2]->isConnected()==false)
        this->plugIn()[2]->setState(CPlug::OLD);
    else
        this->plugIn()[2]->setState(CPlug::EMPTY);

    if(this->plugIn()[3]->isConnected()==false)
        this->plugIn()[3]->setState(CPlug::OLD);
    else
        this->plugIn()[3]->setState(CPlug::EMPTY);

    if(this->plugIn()[4]->isConnected()==false)
        this->plugIn()[4]->setState(CPlug::OLD);
    else
        this->plugIn()[4]->setState(CPlug::EMPTY);


    this->plugOut()[0]->setState(CPlug::EMPTY);
}

void OperatorGetFieldFromMultiPhaseFieldMatN::exec(){
    shared_ptr<BaseMatN> scalarfield_multiphasefield = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<BaseMatN> label_multiphasefield  = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();

    int label =0;
    if(this->plugIn()[2]->isDataAvailable()==true)
        label = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    int width =5;
    if(this->plugIn()[3]->isDataAvailable()==true)
        width = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();


    shared_ptr<BaseMatN> bulk(NULL);
    if(this->plugIn()[4]->isDataAvailable()==true){
        bulk  = dynamic_cast<DataMatN *>(this->plugIn()[4]->getData())->getData();
    }
    BaseMatN * scalarfield_singlephasefield;
    foo func;

//    try{Dynamic2Static<TListImgGridUnsigned>::Switch(func,label_multiphasefield.get(),scalarfield_multiphasefield.get(),label,width,bulk.get(),scalarfield_singlephasefield,Loki::Type2Type<MatN<2,pop::int32> >());}
//    catch(const pexception & e){
//        if(e.what()[0]=='P')
//            this->error("Pixel/voxel type of input image must be scalar type");
//        else
//            this->error(e.what());

//        return;
//    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(scalarfield_singlephasefield));

}

COperator * OperatorGetFieldFromMultiPhaseFieldMatN::clone(){
    return new OperatorGetFieldFromMultiPhaseFieldMatN();
}
