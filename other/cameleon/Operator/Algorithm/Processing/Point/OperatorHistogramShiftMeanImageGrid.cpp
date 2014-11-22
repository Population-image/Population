#include "OperatorHistogramShiftMeanImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorHistogramShiftMeanmageGrid::OperatorHistogramShiftMeanmageGrid()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Point");
    this->setKey("PopulationOperatorHistogramShiftMeanImageGrid");
    this->setName("greylevelTranslateMeanValue");
    this->setInformation("h(x)\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"m1.num (or red.num for color image");
    this->structurePlug().addPlugIn(DataNumber::KEY,"green.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"blue.num");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}
void OperatorHistogramShiftMeanmageGrid::initState(){
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

    this->plugOut()[0]->setState(CPlug::EMPTY);
}

void OperatorHistogramShiftMeanmageGrid::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    BaseMatN * h;
    foo func;

    double red = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    double green;
    if(this->plugIn()[2]->isDataAvailable()==true)
        green = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    else
        green = 125;

    double blue;
    if(this->plugIn()[3]->isDataAvailable()==true)
        blue = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();
    else
        blue = 125;
    BaseMatN * fc1= f1.get();

    try{Dynamic2Static<TListImgGridScalar>::Switch(func,fc1,red,h,Loki::Type2Type<MatN<2,int> >());}
    catch(string msg1){
        try{Dynamic2Static<TListImgGridRGB>::Switch(func,fc1,red,green,blue,h,Loki::Type2Type<MatN<2,int> >());}
        catch(string msg2){
            this->error("Error is \n *"+msg1+"\n *"+msg2);
            return;
        }
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}
COperator * OperatorHistogramShiftMeanmageGrid::clone(){
    return new OperatorHistogramShiftMeanmageGrid();
}
