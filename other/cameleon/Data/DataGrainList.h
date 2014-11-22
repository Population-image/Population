#ifndef DATAGRAINLIST_H
#define DATAGRAINLIST_H

#include<CDataByFile.h>
#include"data/graingerm/GrainGerm.h"
using namespace pop;
class DataGermGrain : public CDataByFile<GermGrainMother>
{
public:
    DataGermGrain();
     static string KEY;
     DataGermGrain * clone();
     shared_ptr<GermGrainMother> getDataByFile();
     void setDataByFile(shared_ptr<GermGrainMother> type);
     void setDataByCopy(shared_ptr<GermGrainMother> type);
};

#endif // DATAGRAINLIST_H
