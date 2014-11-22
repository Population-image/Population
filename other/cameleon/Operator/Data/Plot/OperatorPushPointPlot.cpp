#include "OperatorPushPointPlot.h"
#include<CData.h>
#include<DataPlot.h>
#include<DataNumber.h>

#include "data/utility/PlotGraph.h"

OperatorPushPointPlot::OperatorPushPointPlot(){
        this->path().push_back("Data");
    this->path().push_back("Plot");
    this->setKey("OperatorPushPointPlot");
    this->setName("pushPoint");
    this->setInformation("add the point (x,y) to the graph");
    this->structurePlug().addPlugIn(DataPlot::KEY,"g.plot");
    this->structurePlug().addPlugIn(DataNumber::KEY,"x.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"y.num");
    this->structurePlug().addPlugOut(DataPlot::KEY,"f.plot");
}
void OperatorPushPointPlot::exec(){
    shared_ptr<PlotGraph> g = dynamic_cast<DataPlot *>(this->plugIn()[0]->getData())->getData();
    double x = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    double y = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    g->VGraph()[0].X().push_back(x);
    g->VGraph()[0].Y().push_back(y);
    dynamic_cast<DataPlot *>(this->plugOut()[0]->getData())->setData(g);
}

COperator * OperatorPushPointPlot::clone(){
    return new OperatorPushPointPlot();
}
