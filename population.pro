CONFIG -= qt
CONFIG -= app_bundle
CONFIG +=console
CONFIG += debug_and_release
CONFIG(debug, debug|release){
DEFINES += Debug
}

CONFIG += MAKE_EXE #comment this line to generate the shared library

MAKE_EXE{
    CONFIG += executable
    TEMPLATE = app
    SOURCES +=  $${PWD}/main.cpp
    DESTDIR=$${PWD}/bin
}else{
    CONFIG+=plugin
    TEMPLATE = lib
    DESTDIR=$${PWD}/lib
}

!include($${PWD}/populationconfig.pri)
!include($${PWD}/populationsrc.pri)











