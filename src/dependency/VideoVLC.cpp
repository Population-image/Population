#include "dependency/VideoVLC.h"
#if defined(HAVE_VLC)
#include "vlc/libvlc.h"
#include "vlc/vlc.h"
#include"dependency/tinythread.h"
namespace pop
{
struct ctx
{
    pop::MatN<2,pop::RGBAUI8> * image;
    tthread::mutex * pmutex;
    int * index;
};
void *lock_vlc(void *data, void**p_pixels)
{
    ctx *context = static_cast<ctx*>(data);
    context->pmutex->lock();
    *p_pixels = (unsigned char *)context->image->data();
    return NULL;
}
void display_vlc(void *data, void *id)
{
    (void) data;
    assert(id == NULL);
}
void unlock_vlc(void *data, void *id, void *const *){
    ctx *context = static_cast<ctx*>(data);
    if(*context->index>=0){
        (*context->index)++;
    }
    context->pmutex->unlock();
    assert(id == NULL); // picture identifier, not needed here
}


//libvlc_instance_t* VideoVLC::impl::instance = libvlc_new(0,NULL);
VideoVLC::VideoVLC()
{
    instance = libvlc_new(0,NULL);
    mediaPlayer = NULL;
    context = new  ctx;
    context->pmutex = new tthread::mutex();
    context->image = new pop::MatN<2,pop::RGBAUI8>(10,10);
    context->index = new int(-1);
    my_index = -1;
    isplaying = false;
    _isfile =true;
}
VideoVLC::VideoVLC(const VideoVLC & v)
{
    this->open(v.file_playing);
}

void VideoVLC::release(){
    if(mediaPlayer!=NULL){
        if(libvlc_media_player_is_playing(mediaPlayer))
            libvlc_media_player_stop(mediaPlayer);
        libvlc_media_player_release(mediaPlayer);
        mediaPlayer = NULL;
    }
}

VideoVLC::~VideoVLC()
{
    release();
    delete context->pmutex;
    delete context->image;
    delete context->index;
    delete context;
    libvlc_release(instance);
}
bool VideoVLC::open(const std::string & path)throw(pexception){
    release();
    if(path=="")
        return false;
    libvlc_media_t* media = NULL;
    bool isfile = BasicUtility::isFile(path);
    if(isfile==true){
#if Pop_OS==2
        media = libvlc_media_new_path(instance, BasicUtility::replaceSlashByAntiSlash(path).c_str());
#else
        media = libvlc_media_new_path(instance, path.c_str());
#endif
        _isfile=true;
    }
    else{
        media = libvlc_media_new_location(instance,path.c_str() );
        _isfile=false;
    }
    if(media!=NULL){
        file_playing = path;
        mediaPlayer = libvlc_media_player_new(instance);
        libvlc_media_player_set_media( mediaPlayer, media);
        libvlc_media_release (media);
        if(libvlc_media_player_play(mediaPlayer)==-1)
            return false;
        libvlc_video_set_callbacks(mediaPlayer, lock_vlc, unlock_vlc, display_vlc, context);
        libvlc_video_set_format(mediaPlayer, "RV32", context->image->sizeJ(), context->image->sizeI(), context->image->sizeJ()*4);

        unsigned int w=0,h=0;
        bool find=false;
        int numbertest=0;
        do{

            for(int i=0;i<10;i++){
                if(libvlc_video_get_size( mediaPlayer, i, &w, &h )==0)
                    if(w>0&&h>0){
                        i=5;
                        find =true;

                    }
            }
            numbertest++;
            if(find==false){
#if Pop_OS==2
                Sleep(2000);
#endif
#if Pop_OS==1
                sleep(2);
#endif
            }
        }while(find==false&&numbertest<10);
        if(numbertest<10){
            isplaying    = true;
            libvlc_media_player_stop(mediaPlayer);
            libvlc_media_player_release(mediaPlayer);
            mediaPlayer =NULL;
            std::cout<<h<<std::endl;
            std::cout<<w<<std::endl;
            context->image->resize(h,w);
            media = libvlc_media_new_path(instance, path.c_str());
            if(isfile==true){
#if Pop_OS==2
                media = libvlc_media_new_path(instance, BasicUtility::replaceSlashByAntiSlash(path).c_str());
#else
                media = libvlc_media_new_path(instance, path.c_str());
#endif
            }
            else
                media = libvlc_media_new_location(instance,path.c_str() );
            mediaPlayer = libvlc_media_player_new(instance);
            libvlc_media_player_set_media( mediaPlayer, media);
            libvlc_media_release (media);
            libvlc_media_player_play(mediaPlayer);
            libvlc_video_set_callbacks(mediaPlayer, lock_vlc, unlock_vlc, display_vlc, context);
            libvlc_video_set_format(mediaPlayer, "RV32", context->image->sizeJ(), context->image->sizeI(), context->image->sizeJ()*4);
            *(context->index) =0;
//                std::cout<<"return true"<<std::endl;
#if Pop_OS==2
                Sleep(1000);
#endif
#if Pop_OS==1
                sleep(1);
#endif
            return true;
        }else{
            libvlc_media_player_stop(mediaPlayer);
            libvlc_media_player_release(mediaPlayer);
            mediaPlayer =NULL;
            return false;
        }
    }else{
        return false;
    }
}
bool VideoVLC::tryOpen(const std::string & path){
    if(path=="")
        return false;
    libvlc_media_t* media = NULL;
    bool isfile = BasicUtility::isFile(path);
    if(isfile==true){
#if Pop_OS==2
        media = libvlc_media_new_path(instance, BasicUtility::replaceSlashByAntiSlash(path).c_str());
#else
        media = libvlc_media_new_path(instance, path.c_str());
#endif
        _isfile =true;

    }
    else{
        media = libvlc_media_new_location(instance,path.c_str() );
        _isfile =true;
    }
    if(media!=NULL){
        file_playing = path;
        isplaying    = true;
        mediaPlayer = libvlc_media_player_new(instance);
        libvlc_media_player_set_media( mediaPlayer, media);
        libvlc_media_release (media);
        media =NULL;
        libvlc_media_player_play(mediaPlayer);
        libvlc_video_set_callbacks(mediaPlayer, lock_vlc, unlock_vlc, display_vlc, context);
        libvlc_video_set_format(mediaPlayer, "RV32", context->image->sizeJ(), context->image->sizeI(), context->image->sizeJ()*4);
#if Pop_OS==2
        Sleep(2000);
#endif
#if Pop_OS==1
        sleep(2);
#endif
        if(libvlc_media_player_is_playing(mediaPlayer)){
            libvlc_media_player_stop(mediaPlayer);
            libvlc_media_player_release(mediaPlayer);
            mediaPlayer =NULL;
            return true;
        }
        else{
            libvlc_media_player_stop(mediaPlayer);
            libvlc_media_player_release(mediaPlayer);
            mediaPlayer =NULL;
            return false;
        }
    }else
        return false;
}
bool VideoVLC::grabMatrixGrey(){
    if(isplaying==true){
        while(my_index==*context->index){
            if(_isfile==true&&libvlc_media_player_is_playing(mediaPlayer)==false){
                isplaying = false;
                return false;
            }
            if(_isfile==false){
#if Pop_OS==2
                Sleep(10);
#endif
#if Pop_OS==1
                usleep(10000);
#endif
            }
        }
        my_index=*context->index;
        return true;
    }else
    {
        return false;
    }
}
Mat2UI8& VideoVLC::retrieveMatrixGrey(){
    context->pmutex->lock();
    imggrey = *context->image;
    context->pmutex->unlock();
    return imggrey;
}
bool VideoVLC::isFile()const{
    return _isfile;
}
bool VideoVLC::grabMatrixRGB(){
    if(isplaying==true){
        while(my_index==*context->index){
#if Pop_OS==2
            Sleep(40);
#endif
#if Pop_OS==1
            usleep(40000);
#endif
        }
        my_index=*context->index;
        return true;
    }else
    {
        return false;
    }
}
Mat2RGBUI8& VideoVLC::retrieveMatrixRGB(){
    context->pmutex->lock();
    imgcolor = *context->image;
    context->pmutex->unlock();
    return imgcolor;
}
}
#endif
