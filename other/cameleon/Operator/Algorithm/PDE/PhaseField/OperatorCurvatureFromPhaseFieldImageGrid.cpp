#include "OperatorCurvatureFromPhaseFieldImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorCurvatureFromPhaseFieldMatN::OperatorCurvatureFromPhaseFieldMatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("PDE");
    this->path().push_back("PhaseField");
    this->setKey("PopulationOperatorCurvatureFromPhaseFieldImageGrid");
    this->setName("curvatureFromPhaseField");
    this->setInformation("Multi phase field for Allen-cahn equation with neuman condition on the boundary\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"scalarfield_singlephasefield.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"bulk.num (by default the whole space)");
    this->structurePlug().addPlugOut(DataMatN::KEY,"curvature.pgm");
}
void OperatorCurvatureFromPhaseFieldMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);

    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);


    this->plugOut()[0]->setState(CPlug::EMPTY);
}

void OperatorCurvatureFromPhaseFieldMatN::exec(){
//    Image* scalarfield_multiphasefield = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();


//    Image* bulk=NULL;
//    if(this->plugIn()[1]->isDataAvailable()==true){
//         bulk = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();
//    }
//    Image * curvature;
//    foo func;
//    try{Dynamic2Static<TListImgGridFloat>::Switch(func,scalarfield_multiphasefield,bulk,curvature,Loki::Type2Type<MatN<2,pop::int32> >());}
//    catch(const pexception & e){
//        if(e.what()[0]=='P')
//            this->error("Pixel/voxel type of input image must be scalar type");
//        else
//            this->error(e.what());

//        return;
//    }
//    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData((curvature));

}

COperator * OperatorCurvatureFromPhaseFieldMatN::clone(){
    return new OperatorCurvatureFromPhaseFieldMatN();
}
