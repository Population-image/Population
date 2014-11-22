
CAMELEONPATH =../../../../
!include($${CAMELEONPATH}/configuration.pri)



TARGET        = $$qtLibraryTarget(population)
TEMPLATE = lib
CONFIG += plugin
DESTDIR = $${BUILDPATHLIBS}
LIBS += -L"$${BUILDPATH}"
LIBS += -lcvm
#MACHINE KERNEL INCLUDES (NEEDED BY BLANKLIB)
!include($${CAMELEONPATH}/machine/kernel/includekernel.pri)
!include($${CAMELEONPATH}/machine/scd/includescd.pri)


INCLUDEPATH += $${PWD}/../../

####### CONFIGURATION ######

CONFIG+=NOCIMG
CONFIG+=WITHQT
##### INTEGRATE POPULATION LIBRARY #########
POPULATIONPATH=.
#### FOR CIMG ####
unix:LIBS+= -lX11 -lpthread
win32:LIBS+= -lgdi32
#### FOR OPENGL ####
DEFINES+=WITHOPENGL
DEFINES+=NOGLUT
QT       += core gui opengl
!include($${PWD}/../../Common.pri)
############################################

!include($${PWD}/BindingCameleonInclude.pri)
!include($${PWD}/PopulationBindingCameleon.pri)


