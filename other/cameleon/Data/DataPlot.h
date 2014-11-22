#ifndef DATAPLOT_H
#define DATAPLOT_H

#include<CDataByFile.h>
#include"data/utility/PlotGraph.h"
using namespace pop;
class DataPlot : public CDataByFile<PlotGraph>
{
public:
    DataPlot();
    static string KEY;
    DataPlot * clone();
    void setDataByFile(shared_ptr<PlotGraph> type);
    void setDataByCopy(shared_ptr<PlotGraph> type);
    shared_ptr<PlotGraph> getDataByFile();
};

#endif // DATAPLOT_H
