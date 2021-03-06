#include "3rdparty/VideoVLC.h"
#if defined(HAVE_VLC)
#include <string.h>
#include"3rdparty/tinythread.h"

namespace pop
{
struct ctx
{
    bool playing_started;
    bool end_reached;
    bool encoutered_error;
    int height;
    int width;
    int index;
    pop::MatN<2,pop::VecN<4,UI8> > * image_RV32;
    tthread::mutex * pmutex;
};

static const char DEF_CHROMA[] = "RV32";
static const int DEF_PIXEL_BYTES = 4;


static unsigned video_setup_cb(void **opaque, char *chroma, unsigned *width, unsigned *height, unsigned *pitches, unsigned *lines) {
    struct ctx *context = static_cast<ctx*>(*opaque);

    context->height = *height;
    context->width = *width;
    context->image_RV32 = new pop::MatN<2, pop::VecN<4, pop::UI8> > (context->height, context->width);

    memcpy(chroma, DEF_CHROMA, sizeof(DEF_CHROMA)-1);
    (*pitches) = context->width * DEF_PIXEL_BYTES;
    (*lines)   = context->height;

    return 1;
}

static void video_cleanup_cb(void *opaque) {
    struct ctx *context = static_cast<ctx*>(opaque);
    delete context->image_RV32;
    context->image_RV32 = NULL;
}

static void *lock_vlc(void *data, void**p_pixels)
{
    ctx *context = static_cast<ctx*>(data);
    context->pmutex->lock();
    *p_pixels = (unsigned char *)context->image_RV32->data();
    return NULL;
}

static void display_vlc(void *data, void *id)
{
    (void) data;
    assert(id == NULL);
}

static void unlock_vlc(void *data, void *id, void *const *){
    ctx *context = static_cast<ctx*>(data);
    context->index++;
    context->pmutex->unlock();
    assert(id == NULL); // picture identifier, not needed here
}

static void on_event_vlc(const libvlc_event_t* event, void* data) {
    ctx *context = static_cast<ctx*>(data);

    switch (event->type) {
    case libvlc_MediaPlayerPlaying:
        context->playing_started = true;
        std::cout << "media player is playing" << std::endl;
        break;
    case libvlc_MediaPlayerEndReached:
        std::cout << "media player end reached" << std::endl;
        context->end_reached = true;
        break;
    case libvlc_MediaPlayerEncounteredError:
        std::cout << "media player encountered error" << std::endl;
        context->encoutered_error = true;
        break;
    default:
        break;
    }
}

VideoVLC::VideoVLC(bool vlc_debug)
{
    char const *vlc_argv[] =
    {
        "--no-audio", /* skip any audio track */
        "--no-xlib", /* tell VLC to not use Xlib */
        "--network-caching=450",
        /* Logging things must be at the end of the array */
        "--verbose=3",
        "--extraintf=logger",
    };
    int vlc_argc = sizeof(vlc_argv) / sizeof(*vlc_argv);
    if (!vlc_debug) {
        vlc_argc -= 2;
    }

    if ((instance = libvlc_new(vlc_argc, vlc_argv)) == NULL) {
        std::cerr << "[VideoVLC::new] libvlc_new() error" << std::endl;
    }

    mediaPlayer = NULL;

    context = new  ctx;
    context->pmutex = new tthread::mutex();
    context->playing_started = false;
    context->end_reached = false;
    context->index = -1;
    context->encoutered_error = false;
    my_index = -1;
    isplaying = false;
    _isfile = true;
}

VideoVLC::VideoVLC(const VideoVLC & v)
{
    this->open(v.file_playing);
}

void VideoVLC::release(){
    context->playing_started = false;
    context->end_reached = false;
    context->index = -1;
    context->encoutered_error = false;
    my_index = -1;
    isplaying = false;

    if(mediaPlayer!=NULL){
        if(libvlc_media_player_is_playing(mediaPlayer)) {
            libvlc_media_player_stop(mediaPlayer);
        }
        libvlc_media_player_release(mediaPlayer);
        mediaPlayer = NULL;
    }
}

VideoVLC::~VideoVLC()
{
    release();
    delete context->pmutex;
    delete context;
    libvlc_release(instance);
}

bool VideoVLC::open(const std::string & path){
    release();
    if(path=="") {
        return false;
    }

    libvlc_media_t* media = NULL;
    _isfile = BasicUtility::isFile(path);
    file_playing = path;
    if(_isfile){
#if Pop_OS==2
        file_playing = BasicUtility::replaceSlashByAntiSlash(path);
#endif
        media = libvlc_media_new_path(instance, file_playing.c_str());
    } else{
        media = libvlc_media_new_location(instance, file_playing.c_str());
    }
    if (media == NULL) {
        std::cerr << "[VideoVLC::open] libvlc_media_new_path() error" << std::endl;
        return false;
    }

    if ((mediaPlayer = libvlc_media_player_new_from_media(media)) == NULL) {
        std::cerr << "[VideoVLC::open] libvlc_media_player_new_from_media() error" << std::endl;
        return false;
    }
    libvlc_media_release(media);

    libvlc_video_set_callbacks(mediaPlayer, lock_vlc, unlock_vlc, display_vlc, this->context);
    libvlc_video_set_format_callbacks(mediaPlayer, video_setup_cb, video_cleanup_cb);
    eventManager = libvlc_media_player_event_manager(mediaPlayer);
    libvlc_event_attach(eventManager, libvlc_MediaPlayerPlaying, on_event_vlc, this->context);
    libvlc_event_attach(eventManager, libvlc_MediaPlayerEndReached, on_event_vlc, this->context);
    libvlc_event_attach(eventManager, libvlc_MediaPlayerEncounteredError, on_event_vlc, this->context);

    if ((libvlc_media_player_play(mediaPlayer)) == -1) {
        std::cerr << "[VideoVLC::open] libvlc_media_player_play() error" << std::endl;
        return false;
    }

    isplaying = true;
    my_index = 0;

    pop::BasicUtility::sleep_ms(100);
    return (!context->encoutered_error);
}

bool VideoVLC::grabMatrixGrey(){
    bool ret = false;

    while (!context->playing_started && !context->encoutered_error) {
        pop::BasicUtility::sleep_ms(10);
    }

    if (context->encoutered_error) {
        //std::cerr << "VIDEO ERROR ENCOUNTERED" << std::endl;
        return false;
    }

    if (isplaying) {
        while(context->index <= 0 || my_index == context->index){
            if(!isPlaying()){
                isplaying = false;
                //std::cout << "VIDEO STOPPED PLAYING!!!" << std::endl;
                return false;
            }
            pop::BasicUtility::sleep_ms(10);
        }

        my_index = context->index;
        ret = true;
    } else {
        //std::cout << "VIDEO NOT PLAYING!!!" << std::endl;
    }

    return ret;
}

Mat2UI8& VideoVLC::retrieveMatrixGrey(){
    context->pmutex->lock();
    //the vlc cleanup callback deletes context->image_RV32 once the end of the stream is reached
    if (context->image_RV32) {
        imggrey.resize(context->image_RV32->getDomain());
        std::transform(context->image_RV32->begin(),context->image_RV32->end(),imggrey.begin(),ConvertRV32ToGrey::lumi);
    }
    context->pmutex->unlock();
    return imggrey;
}

bool VideoVLC::isFile()const{
    return _isfile;
}

bool VideoVLC::grabMatrixRGB(){
    return grabMatrixGrey();
}

Mat2RGBUI8& VideoVLC::retrieveMatrixRGB(){
    context->pmutex->lock();
    //the vlc cleanup callback deletes context->image_RV32 once the end of the stream is reached
    if (context->image_RV32) {
        imgcolor.resize(context->image_RV32->getDomain());
        pop::MatN<2,pop::VecN<4,UI8> >::iterator it =  context->image_RV32->begin();
        pop::MatN<2,pop::VecN<4,UI8> >::iterator it_end =  context->image_RV32->end();
        Mat2RGBUI8::iterator it_out = imgcolor.begin();
        while(it!=it_end){
            it_out->r()=it->operator [](2);
            it_out->g()=it->operator [](1);
            it_out->b()=it->operator [](0);
            it++;
            it_out++;
        }
    }
    context->pmutex->unlock();
    return imgcolor;
}

bool VideoVLC::isPlaying() const {
    return (!context->playing_started) || (libvlc_media_player_is_playing(mediaPlayer) && !context->end_reached);
}
bool ConvertRV32ToGrey::init =false;
UI8 ConvertRV32ToGrey::_look_up_table[256][256][256];
pop::UI8 ConvertRV32ToGrey::lumi(const pop::VecN<4,pop::UI8> &rgb){
    if(init==false){
        init= true;
        for(unsigned int i=0;i<256;i++){
            for(unsigned int j=0;j<256;j++){
                for(unsigned int k=0;k<256;k++){
                    _look_up_table[i][j][k]=ArithmeticsSaturation<pop::UI8,pop::F64>::Range(0.299*i + 0.587*j + 0.114*k+0.000001);
                }
            }
        }
    }
    return _look_up_table[rgb(2)][rgb(1)][rgb(0)];
}

pop::RGBUI8 ConvertRV32ToRGBUI8::lumi(const pop::VecN<4,pop::UI8> &rgb){
    return pop::RGBUI8(rgb(2),rgb(1),rgb(0));
}

}

#endif
