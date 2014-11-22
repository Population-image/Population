TARGET=libpopulation
CONFIG -=qt
TEMPLATE = lib
CONFIG      += plugin
CONFIG  += plugin no_plugin_name_prefix
CONFIG  += debug_and_release
OBJECTS_DIR = $${PWD}/compile

DESTDIR = lib/java
#in ubuntu sudo apt-get install openjdk-6-jdk

unix:INCLUDEPATH +=  /usr/lib/jvm/java-6-openjdk-amd64/include/


!include(../../Common.pri)
INCLUDEPATH += ../../
DEFINES+=WITHSWIGJAVA
win32:SWIGEXE=D:/Users/Vincent/Downloads/swigwin-2.0.10/swigwin-2.0.10/swig.exe
SWIGPATH = $${PWD}/
win32:SWIG.commands = $${SWIGEXE}  -java -c++ -o $${SWIGPATH}/population_wrap.cxx $${SWIGPATH}/population.i
unix:SWIG.commands = swig2.0 -java -c++  -o  $${SWIGPATH}/population_wrap.cxx $${SWIGPATH}/population.i
SWIG.path =$${SWIGPATH}
SWIG.depends = FORCE
QMAKE_EXTRA_TARGETS += SWIG
INSTALLS += SWIG
PRE_TARGETDEPS+=SWIG


win32:DLLBUILD.commands = $$QMAKE_CXX -shared $${SWIGPATH}/compile/*.o -o $${SWIGPATH}/_population.pyd $$LIBS
win32:DLLBUILD.path =$${SWIGPATH}
win32:QMAKE_EXTRA_TARGETS += DLLBUILD
win32:INSTALLS += DLLBUILD
win32:POST_TARGETDEPS+=DLLBUILD

MV.commands = mv  $${SWIGPATH}/*.java $${SWIGPATH}/lib/java/
MV.path =$${SWIGPATH}
MV.depends = $${SWIGPATH}/population.py
QMAKE_EXTRA_TARGETS += MV
INSTALLS += MV
POST_TARGETDEPS+=MV


SOURCES   +=    $${SWIGPATH}/population_wrap.cxx
INCLUDEPATH += $${SWIGPATH}/

OTHER_FILES += \
    $${SWIGPATH}/population.i \
    $${SWIGPATH}/processing.i \
    $${SWIGPATH}/visualization.i \
    $${SWIGPATH}/PDE.i \
    $${SWIGPATH}/TempPopulation.i \
    $${SWIGPATH}/lib/python/test.py \
    data.i \
    analysis.i \
    randomgeometry.i \
    convertor.i \
    representation.i \
    draw.i \
    vision.i \
    geometricaltransformation.i

#    DEFINES+=LIBVLC
#    win32:INCLUDEPATH += $${PWD}/dependency/vlc/include
#    win32:LIBS += -L"$${PWD}/dependency/vlc/lib/"
#    win32-msvc2010 {
#        LIBS += -llibvlc
#    } else {
#        LIBS += -lvlc
#    }
