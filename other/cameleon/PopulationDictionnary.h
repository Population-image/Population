#ifndef POPULATIONDICTIONNARY_H
#define POPULATIONDICTIONNARY_H
#include<CDictionnary.h>

class PopulationDictionnary : public CDictionnary
{
public:
    PopulationDictionnary();
    void collectData();
    void collectOperator();
    void collectControl();
};

#endif // POPULATIONDICTIONNARY_H
