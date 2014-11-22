#ifndef populationplugin_H
#define populationplugin_H
#include <QtCore>
#include <QtGui>
#include <QtPlugin>
#include "ICDictionary.h"
#include "CDictionnary.h"
class populationplugin : public QObject, public ICDictionary {
    Q_OBJECT
    Q_INTERFACES(ICDictionary)

public:
    CDictionnary*  getDictionary();
};

#endif // populationplugin_H
