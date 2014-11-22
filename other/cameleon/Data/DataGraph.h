#ifndef DATAGRAPH_H
#define DATAGRAPH_H

#include<CDataByFile.h>
#include"data/notstable/graph/Graph.h"
using namespace pop;
class DataGraph : public CDataByFile<GraphBase>
{
public:
    DataGraph();
    static string KEY;
    DataGraph * clone();
    shared_ptr<GraphBase> getDataByFile();
    void setDataByFile(shared_ptr<GraphBase> type);
    void setDataByCopy(shared_ptr<GraphBase> type);
};
#endif // DATAGRAPH_H
