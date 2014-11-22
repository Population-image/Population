#include "OperatorFractalBoxImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
#include<DataMatrix.h>
OperatorFractalBoxMatN::OperatorFractalBoxMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Geometry");
    this->path().push_back("Statistic");
    this->setKey("PopulationOperatorFractalBoxImageGrid");
    this->setName("fractalBox");
    this->setInformation("P(*,0)=ln(1/r) and P(*,1)=ln(N)  http://classes.yale.edu/fractals/fracanddim/boxdim/BoxDimDef/BoxDimDef.html and http://www.fast.u-psud.fr/~moisy/ml/boxcount/html/demo.html\\#11");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"P.m");
}

void OperatorFractalBoxMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();

    Mat2F64* m;
    foo func;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGrid1Byte>::Switch(func,fc1,m,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be unsigned type used operator Convert1Byte");
        return;
    }
    dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(m));
}
COperator * OperatorFractalBoxMatN::clone(){
    return new OperatorFractalBoxMatN();
}
