#include "DataGraph.h"

DataGraph::DataGraph()
    :CDataByFile<GraphBase>()
{
    this->_key = DataGraph::KEY;
    this->setExtension(".graph");
}
string DataGraph::KEY ="DATAGraph";
DataGraph * DataGraph::clone(){
    return new DataGraph();
}
shared_ptr<GraphBase> DataGraph::getDataByFile(){
    shared_ptr<GraphBase> t(GraphBase::create(this->getFile()) );
    return t;
}
    void DataGraph::setDataByCopy(shared_ptr<GraphBase> type){
        this->_data = shared_ptr<GraphBase>(type->copy());
    }

void DataGraph::setDataByFile(shared_ptr<GraphBase> type){

    type->save(this->getFile());
}


