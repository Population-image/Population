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
        break;
    case libvlc_MediaPlayerEndReached:
        context->end_reached = true;
        break;
    case libvlc_MediaPlayerEncounteredError:
        context->encoutered_error = true;
        break;
    default:
        break;
    }
}

VideoVLC::VideoVLC()
{
    char const *vlc_argv[] =
    {
        "--no-audio", /* skip any audio track */
        "--no-xlib", /* tell VLC to not use Xlib */
    };
    int vlc_argc = sizeof(vlc_argv) / sizeof(*vlc_argv);

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

#if Pop_OS==2
    Sleep(100);
#elif Pop_OS==1
    usleep(100000);
#endif
    return (!context->encoutered_error);
}

bool VideoVLC::grabMatrixGrey(){
    bool ret = false;

    while (!context->playing_started && !context->encoutered_error) {
#if Pop_OS==2
        Sleep(10);
#elif Pop_OS==1
        usleep(10000);
#endif
    }

    if (context->encoutered_error) {
        //std::cerr << "VIDEO ERROR ENCOUNTERED" << std::endl;
        return false;
    }

    if (isplaying) {
        while(context->index <= 0 || my_index == context->index){
            if(_isfile && !isPlaying()){
                isplaying = false;
                //std::cout << "VIDEO STOPPED PLAYING!!!" << std::endl;
                return false;
            } else if(!_isfile){
#if Pop_OS==2
                Sleep(10);
#endif
#if Pop_OS==1
                usleep(10000);
#endif
            }
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
    bool ret = false;

    if (isplaying) {
        while(context->index <= 0 || my_index == context->index){
#if Pop_OS==2
            Sleep(40);
#endif
#if Pop_OS==1
            usleep(40000);
#endif
        }

        my_index = context->index;
        ret = true;
    }

    return ret;
}

Mat2RGBUI8& VideoVLC::retrieveMatrixRGB(){
    context->pmutex->lock();
    //the vlc cleanup callback deletes context->image_RV32 once the end of the stream is reached
    if (context->image_RV32) {
        imgcolor.resize(context->image_RV32->getDomain());
        std::transform(context->image_RV32->begin(),context->image_RV32->end(),imgcolor.begin(),ConvertRV32ToRGBUI8::lumi);
        context->pmutex->unlock();
    }
    return imgcolor;
}

bool VideoVLC::isPlaying() const {
    return (!context->playing_started) || (libvlc_media_player_is_playing(mediaPlayer) && !context->end_reached);
}

}

#endif
