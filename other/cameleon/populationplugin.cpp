#include "populationplugin.h"
#include "PopulationDictionnary.h"
CDictionnary* populationplugin::getDictionary(){
    return new PopulationDictionnary;
}

Q_EXPORT_PLUGIN2(population, populationplugin)
