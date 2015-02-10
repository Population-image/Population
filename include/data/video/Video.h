#ifndef VIDEO_H
#define VIDEO_H

#include"data/mat/MatN.h"
#include"data/mat/Mat2x.h"
/// @cond DEV
namespace pop
{
/*!
 * \brief The ConvertRV32ToGrey struct is used as a rainbow table to speed up the call to lumi()
 */
struct ConvertRV32ToGrey{
    static bool init;
    static UI8 _look_up_table[256][256][256];
    static UI8 lumi(const pop::VecN<4,UI8> &rgb){
        if(init==false){
            init= true;
            for(unsigned int i=0;i<256;i++){
                for(unsigned int j=0;j<256;j++){
                    for(unsigned int k=0;k<256;k++){
                        _look_up_table[i][j][k]=ArithmeticsSaturation<pop::UI8,F64>::Range(0.299*i + 0.587*j + 0.114*k+0.000001);
                    }
                }
            }
        }
        return _look_up_table[rgb(2)][rgb(1)][rgb(0)];
    }
};

/*!
 * \brief The ConvertRV32ToRGBUI8 struct is used to speed up the call to lumi()
 */
struct ConvertRV32ToRGBUI8
{
    static RGBUI8 lumi(const pop::VecN<4,UI8> &rgb){
        return RGBUI8(rgb(2),rgb(1),rgb(0));
    }
};

/*! \ingroup Other
* \defgroup Video Video
* \brief  video player frame by frame (ip-camera, avi)
*/
class POP_EXPORTS Video
{
public:
    /*!
     \brief The different implementations of Video.
    */
    enum VideoImpl{
        OLDVLC,
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
    virtual bool open(const std::string & filename)=0;
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

    /*!
     * \brief isFile
     * \return true iff the video stream comes from a file (i.e., not from the net)
     */
    virtual bool isFile() const=0;

    virtual bool grabMatrixRGB()=0;
    virtual Mat2RGBUI8 &retrieveMatrixRGB()=0;
};
}
/// @endcond
#endif // VIDEO_H
