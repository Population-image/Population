#include "OperatorGraphOpenGl.h"
#include<DataGraph.h>
#include<DataOpenGl.h>
#include"data/notstable/graph/Edge.h"
#include"data/notstable/graph/Vertex.h"
#include"algorithm/Visualization.h"
using namespace pop;
OperatorGraphOpenGl::OperatorGraphOpenGl()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Visualization");
    this->path().push_back("OpenGl");
    this->setKey("PopulationOperatorGraphOpenGl");
    this->setName("graph");
    this->setInformation("");
    this->structurePlug().addPlugIn(DataGraph::KEY,"img.graph");
    this->structurePlug().addPlugOut(DataOpenGl::KEY,"h.pgm");
}
void OperatorGraphOpenGl::exec(){
    shared_ptr<GraphBase> f = dynamic_cast<DataGraph *>(this->plugIn()[0]->getData())->getData();

    if(GraphAdjencyList<VertexPosition,Edge> * fcast=  dynamic_cast<GraphAdjencyList<VertexPosition,Edge> *>(f.get())){
        Scene3d * out = new Scene3d();
        Visualization::graph(*out,* fcast);
        dynamic_cast<DataOpenGl *>(this->plugOut()[0]->getData())->setData(shared_ptr<Scene3d>(out));
    }


}

COperator * OperatorGraphOpenGl::clone(){
    return new OperatorGraphOpenGl();
}
