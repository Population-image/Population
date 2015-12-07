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
    SOURCES +=  $${PWD}/../../tempmain.cpp
}else{
    CONFIG+=plugin
    TEMPLATE = lib
    DESTDIR=$${PWD}/../../lib
}

!include($${PWD}/populationconfigtemp.pri)
!include($${PWD}/populationsrctemp.pri)









