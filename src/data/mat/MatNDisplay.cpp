#include"data/mat/MatNDisplay.h"
#include"Population.h"
#if defined(HAVE_CIMG)
#include"3rdparty/MatNDisplayCImg.h"
#endif
namespace pop{

MatNDisplayInterface::~MatNDisplayInterface (){

}

MatNDisplay::~MatNDisplay (){
    if(_impl!=NULL){
        delete _impl;
    }
}

MatNDisplay::MatNDisplay ()
    :_impl(NULL)
{
#if defined(HAVE_CIMG)
    _impl = new MatNDisplayCImg();
#endif
}
MatNDisplay::MatNDisplay(const MatNDisplay &disp)
    :_impl(NULL)
{
    (void)disp;
#if defined(HAVE_CIMG)
    _impl = new MatNDisplayCImg(*dynamic_cast<MatNDisplayCImg*>(disp._impl));
#endif
}
MatNDisplay & 	MatNDisplay::operator= (const MatNDisplay &disp){
    (void)disp;
    if(_impl!=NULL){
        delete _impl;
    }
#if defined(HAVE_CIMG)
    _impl = new MatNDisplayCImg(*dynamic_cast<MatNDisplayCImg*>(disp._impl));
#endif
    return *this;
}

MatNDisplay & MatNDisplay::display(const MatN<2, RGBUI8 > & m){
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";}else _impl->display(m);
    return *this;
}

//MatNDisplay & MatNDisplay::display(const MatN<2, RGBAUI8 > &m){
//    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";}else _impl->display(m);
//    return *this;
//}

MatNDisplay & MatNDisplay::display(const MatN<2, UI8 > &m){
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";}else _impl->display(m);
    return *this;
}



void waitKey(std::string text){
#if defined(HAVE_CIMG)
    Mat2UI8 m(100,350);
    if(text=="")
        text = "Press this windows to end the infinite loop";
    Draw::text(m,text,Vec2I32(m.sizeI()/3,0),255,1);
    MatNDisplayCImg disp;
    disp.display(m);
    do{
        if(disp.is_key()){
            break;
        }
        if (disp.button()&1) { // Left button clicked.
            break;
        }
        if (disp.button()&2) { // Right button clicked.
            break;
        }
        if (disp.button()&4) { // Middle button clicked.
            break;
        }
        if (disp.is_closed()) { // Middle button clicked.
            break;
        }

    }while(1==1);
#endif
}




MatNDisplay::operator bool () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->operator bool();
}
bool 	MatNDisplay::is_empty () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_empty();
}
bool 	MatNDisplay::is_closed () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return true;}else return _impl->is_closed();
}
bool 	MatNDisplay::is_resized () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_resized();
}
bool 	MatNDisplay::is_moved () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_moved();
}
bool 	MatNDisplay::is_event () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_event();
}
bool 	MatNDisplay::is_fullscreen () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_fullscreen();
}
bool 	MatNDisplay::is_key () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_key();
}
bool 	MatNDisplay::is_key (const unsigned int keycode_value) const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_key(keycode_value);
}
bool 	MatNDisplay::is_key (const char *const keycode_value) const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_key(keycode_value);
}
bool 	MatNDisplay::is_key_sequence (const unsigned int *const keycodes_sequence, const unsigned int length, const bool remove_sequence){
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_key_sequence(keycodes_sequence,length,remove_sequence);
}
bool 	MatNDisplay::is_keyESC () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyESC();
}
bool 	MatNDisplay::is_keyF1 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyF1();
}
bool 	MatNDisplay::is_keyF2 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyF2();
}
bool 	MatNDisplay::is_keyF3 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyF3();
}
bool 	MatNDisplay::is_keyF4 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyF4();
}
bool 	MatNDisplay::is_keyF5 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyF5();
}
bool 	MatNDisplay::is_keyF6 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyF6();
}
bool 	MatNDisplay::is_keyF7 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyF7();
}
bool 	MatNDisplay::is_keyF8() const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyF8();
}
bool 	MatNDisplay::is_keyF9 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyF9();
}
bool 	MatNDisplay::is_keyF10 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyF10();
}
bool 	MatNDisplay::is_keyF11 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyF11();
}
bool 	MatNDisplay::is_keyF12 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyF12();
}
bool 	MatNDisplay::is_keyPAUSE () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyPAUSE();
}
bool 	MatNDisplay::is_key1 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_key1();
}
bool 	MatNDisplay::is_key2 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_key2();
}
bool 	MatNDisplay::is_key3 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_key3();
}
bool 	MatNDisplay::is_key4 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_key4();
}
bool 	MatNDisplay::is_key5 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_key5();
}
bool 	MatNDisplay::is_key6 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_key6();
}
bool 	MatNDisplay::is_key7 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_key7();
}
bool 	MatNDisplay::is_key8 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_key8();
}
bool 	MatNDisplay::is_key9 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_key9();
}
bool 	MatNDisplay::is_key0 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_key0();
}
bool 	MatNDisplay::is_keyBACKSPACE () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyBACKSPACE();
}
bool 	MatNDisplay::is_keyINSERT () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyINSERT();
}
bool 	MatNDisplay::is_keyHOME () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyHOME();
}
bool 	MatNDisplay::is_keyPAGEUP () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyPAGEUP();
}
bool 	MatNDisplay::is_keyTAB () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyTAB();
}
bool 	MatNDisplay::is_keyQ () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyQ();
}
bool 	MatNDisplay::is_keyW () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyW();
}
bool 	MatNDisplay::is_keyE () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyE();
}
bool 	MatNDisplay::is_keyR () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyR();
}
bool 	MatNDisplay::is_keyT () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyT();
}
bool 	MatNDisplay::is_keyY () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyY();
}
bool 	MatNDisplay::is_keyU () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyU();
}
bool 	MatNDisplay::is_keyI () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyI();
}
bool 	MatNDisplay::is_keyO () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyO();
}
bool 	MatNDisplay::is_keyP () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyP();
}
bool 	MatNDisplay::is_keyDELETE () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyDELETE();
}
bool 	MatNDisplay::is_keyEND () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyEND();
}
bool 	MatNDisplay::is_keyPAGEDOWN () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyPAGEDOWN();
}
bool 	MatNDisplay::is_keyCAPSLOCK () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyCAPSLOCK();
}
bool 	MatNDisplay::is_keyA () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyA();
}
bool 	MatNDisplay::is_keyS () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyS();
}
bool 	MatNDisplay::is_keyD () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyD();
}
bool 	MatNDisplay::is_keyF () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyF();
}
bool 	MatNDisplay::is_keyG () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyG();
}
bool 	MatNDisplay::is_keyH () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyH();
}
bool 	MatNDisplay::is_keyJ () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyJ();
}
bool 	MatNDisplay::is_keyK () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyK();
}
bool 	MatNDisplay::is_keyL () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyL();
}
bool 	MatNDisplay::is_keyENTER () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyENTER();
}
bool 	MatNDisplay::is_keySHIFTLEFT () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keySHIFTLEFT();
}
bool 	MatNDisplay::is_keyZ () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyZ();
}
bool 	MatNDisplay::is_keyX () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyX();
}
bool 	MatNDisplay::is_keyC () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyC();
}
bool 	MatNDisplay::is_keyV () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyV();
}
bool 	MatNDisplay::is_keyB () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyB();
}
bool 	MatNDisplay::is_keyN () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyN();
}
bool 	MatNDisplay::is_keyM () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyM();
}
bool 	MatNDisplay::is_keySHIFTRIGHT () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keySHIFTRIGHT();
}
bool 	MatNDisplay::is_keyARROWUP () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyARROWUP();
}
bool 	MatNDisplay::is_keyCTRLLEFT () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyCTRLLEFT();
}
bool 	MatNDisplay::is_keyAPPLEFT () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyAPPLEFT();
}
bool 	MatNDisplay::is_keyALT () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyALT();
}
bool 	MatNDisplay::is_keySPACE () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keySPACE();
}
bool 	MatNDisplay::is_keyALTGR () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyALTGR();
}
bool 	MatNDisplay::is_keyAPPRIGHT () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyAPPRIGHT();
}
bool 	MatNDisplay::is_keyMENU () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyMENU();
}
bool 	MatNDisplay::is_keyCTRLRIGHT () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyCTRLRIGHT();
}
bool 	MatNDisplay::is_keyARROWLEFT () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyARROWLEFT();
}
bool 	MatNDisplay::is_keyARROWDOWN () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyARROWDOWN();
}
bool 	MatNDisplay::is_keyARROWRIGHT () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyARROWRIGHT();
}
bool 	MatNDisplay::is_keyPAD0 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyPAD0();
}
bool 	MatNDisplay::is_keyPAD1 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyPAD1();
}
bool 	MatNDisplay::is_keyPAD2 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyPAD2();
}
bool 	MatNDisplay::is_keyPAD3 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyPAD3();
}
bool 	MatNDisplay::is_keyPAD4 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyPAD4();
}
bool 	MatNDisplay::is_keyPAD5 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyPAD5();
}
bool 	MatNDisplay::is_keyPAD6 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyPAD6();
}
bool 	MatNDisplay::is_keyPAD7 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyPAD7();
}
bool 	MatNDisplay::is_keyPAD8 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyPAD8();
}
bool 	MatNDisplay::is_keyPAD9 () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyPAD9();
}
bool 	MatNDisplay::is_keyPADADD () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyPADADD();
}
bool 	MatNDisplay::is_keyPADSUB () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyPADSUB();
}
bool 	MatNDisplay::is_keyPADMUL () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyPADMUL();
}
bool 	MatNDisplay::is_keyPADDIV () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->is_keyPADDIV();
}
int 	MatNDisplay::width () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->width();
}
int 	MatNDisplay::height () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->height();
}
unsigned int 	MatNDisplay::normalization () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->normalization();
}
const char * 	MatNDisplay::title () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return "";}else return _impl->title();
}
int 	MatNDisplay::window_width () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->window_width();
}
int 	MatNDisplay::window_height () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->window_height();
}
int 	MatNDisplay::window_x () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->window_x();
}
int 	MatNDisplay::window_y () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->window_y();
}
int 	MatNDisplay::mouse_x () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->mouse_x();
}
int 	MatNDisplay::mouse_y () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->mouse_y();
}
unsigned int 	MatNDisplay::button () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->button();
}
int 	MatNDisplay::wheel () const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->wheel();
}
unsigned int 	MatNDisplay::key (const unsigned int pos) const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->key(pos);
}
unsigned int 	MatNDisplay::released_key (const unsigned int pos) const{
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->released_key(pos);
}
float 	MatNDisplay::frames_per_second (){
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->frames_per_second();
}
int 	MatNDisplay::screen_width (){
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->screen_width();
}
int 	MatNDisplay::screen_height (){
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->screen_height();
}
unsigned int 	MatNDisplay::keycode (const char *const keycode_value){
    if(_impl==NULL){std::cerr<<"No visual display because CImg is not included. Add it in CMake !";return false;}else return _impl->keycode(keycode_value);
}
MatNDisplay & 	MatNDisplay::show (){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->show();
    return *this;
}
MatNDisplay & 	MatNDisplay::close (){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->close();
    return *this;
}
MatNDisplay & 	MatNDisplay::move (const int pos_x, const int pos_y){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->move(pos_x,pos_y);
    return *this;
}
MatNDisplay & 	MatNDisplay::resize (const bool force_redraw){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->resize(force_redraw);
    return *this;
}
MatNDisplay & 	MatNDisplay::resize (const int width_value, const int height_value, const bool force_redraw){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->resize(width_value,height_value,force_redraw);
    return *this;
}


MatNDisplay & 	MatNDisplay::set_normalization (const unsigned int normalization_value){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->set_normalization(normalization_value);
    return *this;
}
MatNDisplay & 	MatNDisplay::set_title (const char *const format,...){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->set_title(format);
    return *this;
}
MatNDisplay & 	MatNDisplay::set_fullscreen (const bool is_fullscreen_value, const bool force_redraw){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->set_fullscreen(is_fullscreen_value,force_redraw);
    return *this;
}
MatNDisplay & 	MatNDisplay::toggle_fullscreen (const bool force_redraw){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->toggle_fullscreen(force_redraw);
    return *this;
}
MatNDisplay & 	MatNDisplay::show_mouse (){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->show_mouse();
    return *this;
}
MatNDisplay & 	MatNDisplay::hide_mouse (){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->hide_mouse();
    return *this;
}
MatNDisplay & 	MatNDisplay::set_mouse (const int pos_x, const int pos_y){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->set_mouse(pos_x,pos_y);
    return *this;
}
MatNDisplay & 	MatNDisplay::set_button (){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->set_button();
    return *this;
}
MatNDisplay & 	MatNDisplay::set_button (const unsigned int button_value, const bool is_pressed){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->set_button(button_value,  is_pressed);
    return *this;
}
MatNDisplay & 	MatNDisplay::set_wheel (){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->set_wheel();
    return *this;
}
MatNDisplay & 	MatNDisplay::set_wheel (const int amplitude){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->set_wheel(amplitude);
    return *this;
}
MatNDisplay & 	MatNDisplay::set_key (){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->set_key();
    return *this;
}
MatNDisplay & 	MatNDisplay::set_key (const unsigned int keycode_value, const bool is_pressed){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->set_key( keycode_value,  is_pressed);
    return *this;
}
MatNDisplay & 	MatNDisplay::flush (){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->flush();
    return *this;
}

MatNDisplay & 	MatNDisplay::waitTime (){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->waitTime();
    return *this;
}
MatNDisplay & 	MatNDisplay::waitTime (const unsigned int milliseconds){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->waitTime(milliseconds);
    return *this;
}

MatNDisplay & 	MatNDisplay::paint (){
    if(_impl==NULL)std::cerr<<"No visual display because CImg is not included. Add it in CMake !";else _impl->paint();
    return *this;
}



}
