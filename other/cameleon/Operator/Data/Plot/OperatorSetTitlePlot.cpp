#include "OperatorSetTitlePlot.h"

#include<CData.h>
#include<DataPlot.h>
#include<DataString.h>

#include "data/utility/PlotGraph.h"
OperatorSetTitlePlot::OperatorSetTitlePlot(){
        this->path().push_back("Data");
    this->path().push_back("Plot");
    this->setKey("OperatorSetTitlePlot");
    this->setName("title");
    this->setInformation("set the Title of the graph");
    this->structurePlug().addPlugIn(DataPlot::KEY,"g.plot");
    this->structurePlug().addPlugIn(DataString::KEY,"Title.str");;
    this->structurePlug().addPlugOut(DataPlot::KEY,"f.plot");
}
void OperatorSetTitlePlot::exec(){
    shared_ptr<PlotGraph> graph = dynamic_cast<DataPlot *>(this->plugIn()[0]->getData())->getData();

    string str = dynamic_cast<DataString *>(this->plugIn()[1]->getData())->getValue();
    graph->setTitle(str);

    dynamic_cast<DataPlot *>(this->plugOut()[0]->getData())->setData(graph);
}
COperator * OperatorSetTitlePlot::clone(){
    return new OperatorSetTitlePlot();
}
