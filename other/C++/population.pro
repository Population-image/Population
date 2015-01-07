POPULATIONPATH=/home/vincent/DEV2/Population #replace by yours
TEMPLATE = app
!include($${POPULATIONPATH}/populationconfig.pri)
SOURCES += $${POPULATIONPATH}/main.cpp # the main file of my project (you can replace by yours)
DESTDIR = $${POPULATIONPATH}/bin # directory target
DEPENDPATH=$${POPULATIONPATH}/lib
LIBS+=-L$${POPULATIONPATH}/lib # the path where the dynamic shared library is located for the linking
LIBS += -lpopulation











