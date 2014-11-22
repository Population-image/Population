#include "DataPlot.h"

#include "data/utility/PlotGraph.h"

DataPlot::DataPlot()
    :CDataByFile<PlotGraph>()
{
    _data=shared_ptr<PlotGraph>(new PlotGraph);
    this->_key = DataPlot::KEY;
    this->setExtension(".plot");
    this->setMode(CData::BYCOPY);
}
string DataPlot::KEY ="DATAPLOT";
DataPlot * DataPlot::clone(){
    return new DataPlot();
}
void DataPlot::setDataByFile(shared_ptr<PlotGraph> type){
    type->save(this->getFile());
}
void DataPlot::setDataByCopy(shared_ptr<PlotGraph> type){
    _data = shared_ptr<PlotGraph> (new PlotGraph(*(type.get())));
}

shared_ptr<PlotGraph> DataPlot::getDataByFile(){
    shared_ptr<PlotGraph> t(new PlotGraph);
    t->load(this->getFile());
    return t;
}
