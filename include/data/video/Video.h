#ifndef VIDEO_H
#define VIDEO_H

#include"data/mat/MatN.h"
#include"data/mat/Mat2x.h"
/// @cond DEV
namespace pop
{

/*! \ingroup Other
* \defgroup Video Video
* \brief  video player frame by frame (ip-camera, avi)
*
* \section InstallExternal External library vlc
*
* OS
* - for linux,  install the vlc library as usual (ubuntu  sudo apt-get install libvlc-dev vlc-nox vlc)
* - for windows, execute the installer package http://www.videolan.org/vlc/download-windows.html and you can find the SDK files in the install directory, i.e. development headers and import libraries.
*
* IDE
* - in qt-creator, you have to uncomment the CONFIG += vlc to use vlv player implementation.
* - in others,
*   -# add preprocessor definitions -DLIBVLC
*   -# add include path C:/Program Files (x86)/VideoLAN/VLC/sdk/include/ for windows
*   -# add library path C:/Program Files (x86)/VideoLAN/VLC for windows
*   -# link shared library -lvlc
*/



class POP_EXPORTS Video
{
public:
    /*!
     \brief The different implementations of Video.
    */
    enum VideoImpl{
        VLC,
        FFMPEG
    };

    /*!
  \class pop::Video
  \ingroup Video
  \brief simple class to deal with video
  \author Tariel Vincent

   The implementation uses the ffmpeg library or vlc library. In qtcreator, you uncomment this line  CONFIG += vlc or this one CONFIG += ffmpeg. For visual studio,
   you download QT 5.0 and the Visual Studio Add-in 1.2.0 for Qt5 http://qt-project.org/downloads . Then, you can open the project population.pro with the uncommented line.

\code
    Video video;
//    video.open( "rtsp://192.168.30.142/channel1");
    video.open( "/home/vtariel/Bureau/IPCameraRecord.avi");
    MatNDisplay disp;
    while(video.grabMatrixGrey()){
        disp.display(video.retrieveMatrixGrey());
    }
    video.open( "/home/vtariel/Bureau/IPCameraRecord.avi");
    MatNDisplay disp;
    while(video.grabMatrixRGB()){
        disp.display(video.retrieveMatrixRGB());
    }

\endcode
  */

    /*!
     \brief generic constructor of the different implementations of Video, e.g., VideoVLC if impl==VLC or VideoFFMPEG if imple==FFMPEG.
    */
    static Video* create(VideoImpl impl);

    /*!
    \brief destructor
    *
    */
    inline virtual ~Video(){}
    /*!
    \param   filename IP adress or file path
    \brief open the file or the network stream
    *
    */
    virtual bool open(const std::string & filename)throw(pexception)=0;
    /*!
    \return false  no  frame anymore
    \brief  grab the next frame
    *
    */
    virtual bool grabMatrixGrey()=0;

    /*!
    \return grey Matrix frame
    \brief reads the frame
     *
    */
    virtual Mat2UI8 &retrieveMatrixGrey()=0;


    virtual bool grabMatrixRGB()=0;
    virtual Mat2RGBUI8 &retrieveMatrixRGB()=0;
    virtual bool tryOpen(const std::string & filename)=0;
};
}
/// @endcond
#endif // VIDEO_H
