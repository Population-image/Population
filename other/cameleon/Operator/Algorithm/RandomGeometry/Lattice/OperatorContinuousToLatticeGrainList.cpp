#include "OperatorContinuousToLatticeGrainList.h"

#include <DataPoint.h>
#include <DataGrainList.h>
#include <DataImageGrid.h>
#include"algorithm/RandomGeometry.h"
using namespace pop;


OperatorContinuousToLatticeGermGrain::OperatorContinuousToLatticeGermGrain()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("RandomGeometry");
    this->path().push_back("Lattice");
    this->setKey("OperatorContinuousToLatticeGrainList");
    this->setName("continuousToDiscrete");
    this->setInformation("Lattice the continuous model where the domain is defined by its topcorner and buttoncorner and the defauld domain is the continuous model domain by default with periodic condition");
    this->structurePlug().addPlugIn(DataGermGrain::KEY,"phi.grainlist");
    this->structurePlug().addPlugIn(DataPoint::KEY,"buttoncorner.v");
    this->structurePlug().addPlugIn(DataPoint::KEY,"topcorner.v");
    this->structurePlug().addPlugOut(DataMatN::KEY,"lattice.pgm");
}

void OperatorContinuousToLatticeGermGrain::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);

    if(this->plugIn()[2]->isConnected()==false)
        this->plugIn()[2]->setState(CPlug::OLD);
    else
        this->plugIn()[2]->setState(CPlug::EMPTY);


    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorContinuousToLatticeGermGrain::exec(){


    shared_ptr<GermGrainMother> phi  = dynamic_cast<DataGermGrain *>(this->plugIn()[0]->getData())->getData();
    GermGrainMother * phic = phi.get();

    if(phi->dim==2)
    {
        if(this->plugIn()[1]->isDataAvailable()==true&&this->plugIn()[2]->isDataAvailable()==true){
            VecF64  v1 = dynamic_cast<DataPoint *>(this->plugIn()[1]->getData())->getValue();
            VecF64  v2 = dynamic_cast<DataPoint *>(this->plugIn()[2]->getData())->getValue();
            MatN<2,RGBUI8 > * lattice= new MatN<2,RGBUI8 >;
            *lattice =RandomGeometry::continuousToDiscrete(*dynamic_cast<GermGrain2 *>(phic),v1,v2);
            dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(lattice));
        }
        else if(this->plugIn()[1]->isDataAvailable()==false&&this->plugIn()[2]->isDataAvailable()==false)
        {
            MatN<2,RGBUI8 > * lattice= new MatN<2,RGBUI8 >;
            *lattice =RandomGeometry::continuousToDiscrete(*dynamic_cast<GermGrain2 *>(phic));
            dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(lattice));
        }
        else
        {
            this->error("plug1 and plug2 must be set or not");
        }
    }
    else if (phi->dim==3)
    {
        if(this->plugIn()[1]->isDataAvailable()==true&&this->plugIn()[2]->isDataAvailable()==true){
            VecF64  v1 = dynamic_cast<DataPoint *>(this->plugIn()[1]->getData())->getValue();
            VecF64  v2 = dynamic_cast<DataPoint *>(this->plugIn()[2]->getData())->getValue();
            MatN<3,RGBUI8 > * lattice= new MatN<3,RGBUI8 >;
            *lattice =RandomGeometry::continuousToDiscrete(*dynamic_cast<GermGrain3 *>(phic),v1,v2);
            dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(lattice));
        }
        else if(this->plugIn()[1]->isDataAvailable()==false&&this->plugIn()[2]->isDataAvailable()==false)
        {
            MatN<3,RGBUI8 > * lattice= new MatN<3,RGBUI8 >;
            *lattice =RandomGeometry::continuousToDiscrete(*dynamic_cast<GermGrain3 *>(phic));
            dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(lattice));
        }
        else
        {
            this->error("plug1 and plug2 must be set or not");
        }
    }
}

COperator * OperatorContinuousToLatticeGermGrain::clone(){
    return new OperatorContinuousToLatticeGermGrain;
}
