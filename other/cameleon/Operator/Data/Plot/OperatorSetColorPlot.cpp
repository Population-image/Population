#include "OperatorSetColorPlot.h"

#include<CData.h>
#include<DataPlot.h>
#include<DataNumber.h>

#include "data/utility/PlotGraph.h"
OperatorSetColorPlot::OperatorSetColorPlot(){
        this->path().push_back("Data");
    this->path().push_back("Plot");
    this->setKey("OperatorSetColorPlot");
    this->setName("color");
    this->setInformation("set the color of the graph where red coresponds to (r=255,g=0,b=0), green (r=0,g=255,b=0) and blue (r=0,g=0,b=255) and any color a ratio of these three colors");
    this->structurePlug().addPlugIn(DataPlot::KEY,"g.plot");
    this->structurePlug().addPlugIn(DataNumber::KEY,"r.num(by default 0)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"g.num(by default 0)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"b.num(by default 0)");
    this->structurePlug().addPlugOut(DataPlot::KEY,"f.plot");
}
void OperatorSetColorPlot::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);

    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
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

void OperatorSetColorPlot::exec(){
    shared_ptr<PlotGraph> graph = dynamic_cast<DataPlot *>(this->plugIn()[0]->getData())->getData();

    double r=0;
    if(this->plugIn()[1]->isDataAvailable()==true)
        r = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();

    double g=0;
    if(this->plugIn()[2]->isDataAvailable()==true)
        g = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();

    double b=0;
    if(this->plugIn()[3]->isDataAvailable()==true)
        b = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();


    graph->VGraph()[0].setRGB(RGBUI8(r,g,b));

    dynamic_cast<DataPlot *>(this->plugOut()[0]->getData())->setData(graph);
}
COperator * OperatorSetColorPlot::clone(){
    return new OperatorSetColorPlot();
}
