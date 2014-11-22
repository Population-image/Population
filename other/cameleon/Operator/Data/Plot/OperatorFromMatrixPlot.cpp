#include "OperatorFromMatrixPlot.h"

#include<CData.h>
#include<DataPlot.h>
#include<DataNumber.h>
#include<DataMatrix.h>

#include "data/utility/PlotGraph.h"

OperatorFromMatrixPlot::OperatorFromMatrixPlot(){

    this->path().push_back("Data");
    this->path().push_back("Plot");
    this->setKey("OperatorFromMatrixPlot");
    this->setName("fromMatrix");
    this->setInformation("Create a graph from a matrix where the points of x-axis are the column of IndexX and the points of y-axis are IndexY");

    this->structurePlug().addPlugIn(DataMatrix::KEY,"matrix.m");
    this->structurePlug().addPlugIn(DataNumber::KEY,"IndexX.num(by default 0)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"IndexY.num(by default 1)");
    this->structurePlug().addPlugOut(DataPlot::KEY,"g.plot");
}

void OperatorFromMatrixPlot::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);

    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);

    if(this->plugIn()[2]->isConnected()==false)
        this->plugIn()[2]->setState(CPlug::OLD);
    else
        this->plugIn()[2]->setState(CPlug::EMPTY);

    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorFromMatrixPlot::exec(){
    try{
    shared_ptr<Mat2F64> m1 = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();
    int colx =0;
    if(this->plugIn()[1]->isDataAvailable()==true)
        colx = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();

    int coly =1;
    if(this->plugIn()[2]->isDataAvailable()==true)
        coly = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();

    PlotGraph* m(new PlotGraph());
    Mat2F64*m1c = m1.get();
    * m = PlotGraphProcedureFromMatrix(*m1c,colx,coly);
    dynamic_cast<DataPlot *>(this->plugOut()[0]->getData())->setData(shared_ptr<PlotGraph>(m));
    }catch(pexception msg){
        this->error(msg.what());
    }
}
COperator * OperatorFromMatrixPlot::clone(){
    return new OperatorFromMatrixPlot();
}
