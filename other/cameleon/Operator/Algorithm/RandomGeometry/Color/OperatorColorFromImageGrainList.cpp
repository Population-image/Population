#include "OperatorColorFromImageGrainList.h"

#include <DataImageGrid.h>
#include <DataGrainList.h>
#include"algorithm/RandomGeometry.h"
using namespace pop;
OperatorColorFromImageGermGrain::OperatorColorFromImageGermGrain()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("RandomGeometry");
    this->path().push_back("Color");
    this->setKey("OperatorColorFromImageGrainList");
    this->setName("colorFromImage");
    this->setInformation("phi'=phi such that each grain take the color of the pixel located at the grain position  \n");
    this->structurePlug().addPlugIn(DataGermGrain::KEY,"phi.grainlist");
    this->structurePlug().addPlugIn(DataMatN::KEY,"img.pgm");
    this->structurePlug().addPlugOut(DataGermGrain::KEY,"phi'.grainlist");
}


void OperatorColorFromImageGermGrain::exec(){

    shared_ptr<GermGrainMother> phi  = dynamic_cast<DataGermGrain *>(this->plugIn()[0]->getData())->getData() ;
    shared_ptr<BaseMatN> img  = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData() ;

    if(phi->dim==2)
    {
        GermGrain2 * phiin = dynamic_cast<GermGrain2 * >(phi.get());
        if(MatN<2,RGBUI8 > *color = dynamic_cast<MatN<2,RGBUI8 > *>(img.get()) ){
            foo f;
            f(phiin, color);
            dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(phi);
        }else{
            this->error("the input image must be a color Image (used ConvertRGB2Color)");
        }
    }
    else if (phi->dim==3)
    {
        GermGrain3 * phiin = dynamic_cast<GermGrain3 * >(phi.get());
        if(MatN<3,RGBUI8 > *color = dynamic_cast<MatN<3,RGBUI8 > *>(img.get()) ){
            foo f;
            f(phiin, color);
            dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(phi);
        }else{
            this->error("the input image must be a color Image (used ConvertRGB2Color)");
        }
    }
}

COperator * OperatorColorFromImageGermGrain::clone(){
    return new OperatorColorFromImageGermGrain;
}
