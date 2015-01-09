#### C++ compiler #####
#CONFIG += c++11
#QMAKE_CXX = clang++

#### PLUG-IN #####
## Uncommented a line to use these plug-in
#CONFIG += HAVE_OPENGL  #opengl for 3d rendering
#CONFIG += HAVE_CIMG    #CIMG to display 2d image in windows
#CONFIG += HAVE_OPENMP  #openmp optimization
#CONFIG += HAVE_QT     #convert QImage to pop::Mat2UI8 or pop::Mat2RGBUI8
#CONFIG += HAVE_VLC    #VideoVLC to load stream video as rtsp (if you use QT5 http://stackoverflow.com/questions/19175391/libvlc-qt-realloc-error-executing-example-program-of-libvlc-qt)
#CONFIG += HAVE_FFMPEG #VideoVLC to load stream video (as rtsp)  or file video
#CONFIG += HAVE_OPENCV #convert cv::Mat to pop::Mat2UI8 or pop::Mat2RGBUI8

DEFINES += 'POP_PROJECT_SOURCE_DIR=\'\"$${PWD}\"\''#path to the population library
DEFINES+=HAVE_QMAKE#Do not include the popconfig.h generating by cmake

INCLUDEPATH +=$${PWD}/include/ # header path



HAVE_OPENGL{
    ##For linux,  install  glut (ubunto sudo apt-get install freeglut3-dev)
    DEFINES+=HAVE_OPENGL
    DEFINES+=HAVE_THREAD
    unix:LIBS+=-lglut -lGL -lGLU  -lpthread
    win32:LIBS += -lAdvapi32 -lgdi32 -luser32 -lshell32 -lopengl32 -lglu32
}
HAVE_CIMG{
    DEFINES+=HAVE_CIMG
    DEFINES*=HAVE_THREAD
    unix:LIBS*=-lX11 -lpthread
    win32:LIBS*=-lAdvapi32 -lgdi32 -luser32 -lshell32
}

HAVE_VLC{
# INSTALLATION VLC
##Install the vlc library as usual (ubuntu  sudo apt-get install libvlc-dev vlc-nox vlc)
##For windows, download http://www.videolan.org/vlc/download-windows.html and install C:/Program Files (x86)/VideoLAN
    DEFINES+=HAVE_VLC
    DEFINES*=HAVE_THREAD
    unix:LIBS*=-lX11 -lpthread
    win32:LIBS*=-lAdvapi32 -lgdi32 -luser32 -lshell32
    win32:INCLUDEPATH +="C:/Program Files (x86)/VideoLAN/VLC/sdk/include/"
    win32:LIBS += -L"C:/Program Files (x86)/VideoLAN/VLC"
    win32-msvc2010 {
        LIBS += -llibvlc
    } else {
        LIBS += -lvlc
    }
}
HAVE_QT {
    CONFIG*=qt
    DEFINES+= HAVE_QT
}
HAVE_FFMPEG {
##For linux,  install the ffmpeg library as usual (ubuntu  sudo apt-get install ffmpeg)
##For windows, link your project with the dev ffmpeg library http://ffmpeg.zeranoe.com/builds/win32/dev/
    DEFINES+= HAVE_FFMPEG
    DEFINES*=HAVE_THREAD
    unix:LIBS*=-lX11 -lpthread
    win32:LIBS*=-lAdvapi32 -lgdi32 -luser32 -lshell32
    win32:FFMPEGPATH+=D:/Users/vtariel/DEV/ffmpeg-20150106-git-a79ac73-win32-dev/  #replace by yours
    win32:INCLUDEPATH+=$${POPULATIONPATH}/include
    win32:LIBS+=-L$${POPULATIONPATH}/lib
    LIBS += -lavcodec
    LIBS += -lavformat
    LIBS += -lavutil
    LIBS += -lswscale
}

HAVE_OPENCV {
##For linux,  install the opencv library as usual (ubuntu  sudo apt-get install libopencv-dev). For windows, you have to do the job ;)
    DEFINES+= HAVE_OPENCV
    unix:CONFIG += link_pkgconfig
    unix:PKGCONFIG += opencv
}
HAVE_OPENMP {
    DEFINES+= HAVE_OPENMP
    unix:QMAKE_CXXFLAGS+= -fopenmp
    unix:QMAKE_LFLAGS +=  -fopenmp
    win32:QMAKE_CXXFLAGS+= -openmp
    win32:QMAKE_LFLAGS +=  -openmp
}else {
    QMAKE_CXXFLAGS+= -Wunknown-pragmas
 }
