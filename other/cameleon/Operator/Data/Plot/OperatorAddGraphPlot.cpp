#include "OperatorAddGraphPlot.h"

#include<CData.h>
#include<DataPlot.h>
#include<DataNumber.h>

#include "data/utility/PlotGraph.h"

OperatorAddGraphPlot::OperatorAddGraphPlot(){
    this->path().push_back("Data");
    this->path().push_back("Plot");
    this->setKey("OperatorAddGraphPlot");
    this->setName("addGraph");
    this->setInformation("Add the two graph in order to plot both");
    this->structurePlug().addPlugIn(DataPlot::KEY,"gmaster.plot");
    this->structurePlug().addPlugIn(DataPlot::KEY,"h.plot");
    this->structurePlug().addPlugOut(DataPlot::KEY,"f.plot");
}
void OperatorAddGraphPlot::exec(){
    shared_ptr<PlotGraph> g = dynamic_cast<DataPlot *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<PlotGraph> h = dynamic_cast<DataPlot *>(this->plugIn()[1]->getData())->getData();

    for(int i =0;i<(int)h->VGraph().size();i++){
        g->VGraph().push_back(h->VGraph()[i]);
    }
    dynamic_cast<DataPlot *>(this->plugOut()[0]->getData())->setData(g);
}
COperator * OperatorAddGraphPlot::clone(){
    return new OperatorAddGraphPlot();
}
