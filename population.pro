CONFIG+=qt
CONFIG+=console
CONFIG  += debug_and_release
CONFIG(debug, debug|release){
DEFINES += Debug
}

CONFIG += executable
TEMPLATE = app
SOURCES +=  $${PWD}/main.cpp
DESTDIR=$${PWD}/bin




#### PLUG-IN #####
## Uncommented a line to use these plug-in
CONFIG += HAVE_OPENGL #opengl for 3d rendering
CONFIG += HAVE_CIMG #CIMG to display 2d image in windows
#CONFIG += HAVE_VLC  #VideoVLC to load stream video as rtsp (if you use QT5 http://stackoverflow.com/questions/19175391/libvlc-qt-realloc-error-executing-example-program-of-libvlc-qt)
#CONFIG += HAVE_FFMPEG #VideoVLC to load stream video (as rtsp)  or file video
CONFIG += HAVE_QT #convert QImage to pop::Mat2UI8 or pop::Mat2RGBUI8
#CONFIG += HAVE_OPENCV #convert cv::Mat to pop::Mat2UI8 or pop::Mat2RGBUI8

##### Common #####
##For linux,  install  glut (ubunto sudo apt-get install freeglut3-dev)
CONFIG += HAVE_QMAKE
!include($${PWD}/population.pri)







#### C++ compiler #####
#QMAKE_CXX = clang++


# INSTALLATION VLC
##Install the vlc library as usual (ubuntu  sudo apt-get install libvlc-dev vlc-nox vlc)
##For windows, download http://www.videolan.org/vlc/download-windows.html and install C:/Program Files (x86)/VideoLAN


# INSTALLATION VIDEO
##For linux,  install the ffmpeg library as usual (ubuntu  sudo apt-get install ffmpeg)
##For windows, link your project with the ffmpeg library located in  $${PWD}/core/dependency/ffmpeg/, and add these files $${PWD}/core/dependency/ffmpeg/bin/* in your working directory (PS: with visual studio compiler, only in debub mode, the code works)

### OPENCV Convertor ###
##For linux,  install the opencv library as usual (ubuntu  sudo apt-get install libopencv-dev)






