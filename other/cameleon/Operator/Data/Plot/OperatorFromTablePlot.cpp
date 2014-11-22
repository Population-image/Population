#include "OperatorFromTablePlot.h"

#include<CData.h>
#include<DataPlot.h>
#include<DataNumber.h>
#include<DataTable.h>
#include "data/utility/PlotGraph.h"


OperatorFromTablePlot::OperatorFromTablePlot(){

    this->path().push_back("Data");
    this->path().push_back("Plot");
    this->setKey("OperatorFromTablePlot");
    this->setName("fromTable");
    this->setInformation("Create a graph from a Table where the points of x-axis are the column of IndexX and the points of y-axis are the column of IndexY");

    this->structurePlug().addPlugIn(DataTable::KEY,"Table.m");
    this->structurePlug().addPlugIn(DataNumber::KEY,"IndexX.num(by default 0)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"IndexY.num(by default 1)");
    this->structurePlug().addPlugOut(DataPlot::KEY,"g.plot");
}

void OperatorFromTablePlot::initState(){
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
void OperatorFromTablePlot::exec(){
    shared_ptr<Table> m1 = dynamic_cast<DataTable *>(this->plugIn()[0]->getData())->getData();
    int colx =0;
    if(this->plugIn()[1]->isDataAvailable()==true)
        colx = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();

    int coly =1;
    if(this->plugIn()[2]->isDataAvailable()==true)
        coly = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();

    PlotGraph* m(new PlotGraph());
    Table *m1c = m1.get();
    if(colx>=0 && colx<m1c->sizeCol() && coly>=0 && coly<m1c->sizeCol()){
        vector<double> x(m1c->sizeRow()), y(m1c->sizeRow()); // initialize with 101 entries
        for(int i=0;i<m1c->sizeRow();i++){
             UtilityString::String2Any((*m1c)(colx,i),x[i]);
             UtilityString::String2Any((*m1c)(coly,i),y[i]);
        }
        PlotSingleGraph plot;
        plot.setPoints(x,y);
        *m = plot;
        dynamic_cast<DataPlot *>(this->plugOut()[0]->getData())->setData(shared_ptr<PlotGraph>(m));
    }
    else{
        this->error("Out of range in the operator OperatorFromTablePlot");
    }
}
COperator * OperatorFromTablePlot::clone(){
    return new OperatorFromTablePlot();
}
