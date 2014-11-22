#include "OperatorVertexAndEdgeFromSkeletonImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
#include<DataString.h>
#include<DataMatrix.h>
OperatorVertexAndEdgeFromSkeletonMatN::OperatorVertexAndEdgeFromSkeletonMatN()
    :COperator()
{
        this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Topology");
    this->path().push_back("Skeleton");
    this->setKey("PopulationOperatorVertexAndEdgeFromSkeletonImageGrid");
    this->setName("fromSkeletonToVertexAndEdge");
    this->setInformation("Thinning the binary image at constant topology\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"skeleton.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"vertex.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"edge.pgm");
}

void OperatorVertexAndEdgeFromSkeletonMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    foo func;

    BaseMatN * vertex;
    BaseMatN * edge;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGrid1Byte>::Switch(func,fc1,vertex,edge,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(vertex));
    dynamic_cast<DataMatN *>(this->plugOut()[1]->getData())->setData(shared_ptr<BaseMatN>(edge));
}

COperator * OperatorVertexAndEdgeFromSkeletonMatN::clone(){
    return new OperatorVertexAndEdgeFromSkeletonMatN();
}
