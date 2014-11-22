#include "OperatorSetAlphaPlot.h"

#include<CData.h>
#include<DataPlot.h>
#include<DataNumber.h>

#include "data/utility/PlotGraph.h"

OperatorSetAlphaPlot::OperatorSetAlphaPlot(){
        this->path().push_back("Data");
    this->path().push_back("Plot");
    this->setKey("OperatorSetAlphaPlot");
    this->setName("alpha");
    this->setInformation("set the transparency of the brush with a value in the range [0,1]");
    this->structurePlug().addPlugIn(DataPlot::KEY,"g.plot");
    this->structurePlug().addPlugIn(DataNumber::KEY,"alpha.num");
    this->structurePlug().addPlugOut(DataPlot::KEY,"f.plot");
}


void OperatorSetAlphaPlot::exec(){
    shared_ptr<PlotGraph> graph = dynamic_cast<DataPlot *>(this->plugIn()[0]->getData())->getData();

    double alpha = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();



    graph->VGraph()[0].setAlpha(alpha);

    dynamic_cast<DataPlot *>(this->plugOut()[0]->getData())->setData(graph);
}
COperator * OperatorSetAlphaPlot::clone(){
    return new OperatorSetAlphaPlot();
}
