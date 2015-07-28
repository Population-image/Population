#ifndef VIDEO_H
#define VIDEO_H

#include"data/mat/MatN.h"
#include"data/mat/Mat2x.h"
namespace pop
{

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
        VLCDEPRECATED,
        VLC,
        FFMPEG
    };

    /*!
  \class pop::Video
  \ingroup Video
  \brief simple class to read  video (rtsp stream, avi file ...)
  \author Tariel Vincent

   The implementation uses the ffmpeg library or vlc library. In qtcreator, you uncomment this line  CONFIG += HAVE_VLC in populationconfig.pri. For CMake, in CMakeList.txt, you set WITH_VLC at ON.

\code
        Video * video =Video::create(Video::VLC);
        video->open( "/home/tariel/flame.avi");//http://www.engr.colostate.edu/me/facil/dynamics/avis.htm
        MatNDisplay disp;
        MatNDisplay disp2;
        while(video->grabMatrixGrey()){
            Mat2UI8 m = video->retrieveMatrixGrey();
            int value;
            Mat2UI8 m_threshold = Processing::thresholdOtsuMethod(m,value);
            disp.display(m);
            disp2.display(m_threshold);
        }
        return 1;

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
#endif // VIDEO_H
