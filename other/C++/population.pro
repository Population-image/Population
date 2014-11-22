CONFIG-=qt
CONFIG+=console
CONFIG  += debug_and_release
CONFIG(debug, debug|release){
DEFINES += Debug
}



PopulationPath=$${PWD}/../../
CONFIG+=plugin
TEMPLATE = lib
DESTDIR=$${PopulationPath}/bin
TARGET = population



#### PLUG-IN #####
## Uncommented a line to use these plug-in
CONFIG += HAVE_OPENGL #opengl for 3d rendering
CONFIG += HAVE_CIMG #CIMG to display 2d image in windows
#CONFIG += HAVE_VLC  #VideoVLC to load stream video as rtsp
#CONFIG += HAVE_FFMPEG #VideoVLC to load stream video (as rtsp)  or file video
#CONFIG += HAVE_QT #convert QImage to pop::Mat2UI8 or pop::Mat2RGBUI8
#CONFIG += HAVE_OPENCV #convert cv::Mat to pop::Mat2UI8 or pop::Mat2RGBUI8

##### Common #####
##For linux,  install  glut (ubunto sudo apt-get install freeglut3-dev)
!include($${PWD}/../../population.pri)










