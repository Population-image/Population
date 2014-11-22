#include "OperatorLinkVertexWithEdgeImageGrid.h"

#include<DataImageGrid.h>
#include<DataGraph.h>
#include<DataNumber.h>
OperatorLinkEdgeVertexMatN::OperatorLinkEdgeVertexMatN()
    :COperator()
{
        this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Topology");
    this->path().push_back("Skeleton");

    this->setKey("PopulationOperatorLinkEdgeVertexImageGrid");
    this->setName("linkEdgeVertex");
    this->setInformation("Link edge and vertex to build the graph\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"vertex.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"edge.pgm");
    this->structurePlug().addPlugOut(DataGraph::KEY,"g.graph");
    this->structurePlug().addPlugOut(DataNumber::KEY,"tore.num");
}

void OperatorLinkEdgeVertexMatN::exec(){
    shared_ptr<BaseMatN> vertex = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<BaseMatN> edge   = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();
    foo func;
    GraphBase * graph;
    BaseMatN * vertexc= vertex.get();
     BaseMatN * edgec= edge.get();
     int tore;
    try{Dynamic2Static<TListImgGridUnsigned>::Switch(func,vertexc,edgec,graph,tore,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
         if(msg.what()[0]=='P')
             this->error("Pixel/voxel type of input image must be registered type");
         else
             this->error(msg.what());
        return;
    }
    dynamic_cast<DataGraph *>(this->plugOut()[0]->getData())->setData(shared_ptr<GraphBase>(graph));
         dynamic_cast<DataNumber *>(this->plugOut()[1]->getData())->setValue(tore);
}
COperator * OperatorLinkEdgeVertexMatN::clone(){
    return new OperatorLinkEdgeVertexMatN();
}
