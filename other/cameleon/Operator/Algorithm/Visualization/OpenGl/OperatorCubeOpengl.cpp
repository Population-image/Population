#include "OperatorCubeOpengl.h"
#include "DataNumber.h"

OperatorCubeOpenGl::OperatorCubeOpenGl()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Visualization");
    this->path().push_back("OpenGl");
    this->setKey("PopulationOperatorCubeOpenGl");
    this->setName("cubel");
    this->setInformation("Boundary surface of the input 3D binary image with red lines");
    this->structurePlug().addPlugIn(DataMatN::KEY,"binary.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"width.pgm(by default 3)");
    this->structurePlug().addPlugOut(DataOpenGl::KEY,"h.pgm");
}

void OperatorCubeOpenGl::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorCubeOpenGl::exec(){
    shared_ptr<BaseMatN> f = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();

    double width=3;
    if(this->plugIn()[1]->isDataAvailable()==true){
        width = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    }


    foo func;
    Scene3d * out = new Scene3d();
    BaseMatN * fc= f.get();
    typedef  FilterKeepTlistTlist<TListImgGrid,0,Loki::Int2Type<3> >::Result ListFilter;
    typedef FilterRemoveTlistTlist<ListFilter, 1, Complex<pop::F64> >::Result ListFilter2;


    try{Dynamic2Static<ListFilter2>::Switch(func,fc,width,out,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Input image must be 3D");
        return;
    }
    dynamic_cast<DataOpenGl *>(this->plugOut()[0]->getData())->setData(shared_ptr<Scene3d>(out));

}

COperator * OperatorCubeOpenGl::clone(){
    return new OperatorCubeOpenGl();
}
