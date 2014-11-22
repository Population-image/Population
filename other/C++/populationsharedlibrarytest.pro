CONFIG-=qt
CONFIG+=console
CONFIG  += debug_and_release
CONFIG(debug, debug|release){
DEFINES += Debug
}


#### application ####
PopulationPath=$${PWD}/../../
CONFIG += executable
TEMPLATE = app
INCLUDEPATH +=$${PopulationPath}
DESTDIR = $${PopulationPath}/bin
DEPENDPATH=$${PopulationPath}/bin
LIBS+=-L$${PopulationPath}/bin
LIBS += -lpopulation




#### main routine ####
SOURCES +=  $${PopulationPath}/main.cpp



