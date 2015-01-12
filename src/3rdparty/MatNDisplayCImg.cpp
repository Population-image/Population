#include "3rdparty/MatNDisplayCImg.h"
#if defined(HAVE_CIMG)
#include"3rdparty/CImg.h"
namespace pop{

class MatNDisplayCImg::impl
{
public:
    cimg_library::CImgDisplay * display;
};
MatNDisplayCImg::~MatNDisplayCImg (){
    delete _pImpl->display;
    delete _pImpl;
}

MatNDisplayCImg::MatNDisplayCImg ()

{
    _pImpl = new impl();
    _pImpl->display = new cimg_library::CImgDisplay();
}



MatNDisplayCImg & MatNDisplayCImg::display(const MatN<2, RGBUI8 > &img){
    cimg_library::CImg<UI8> temp(img.getDomain()(1),img.getDomain()(0),1,3);
    for(int i =0;i<img.getDomain()[0];i++)
        for(int j =0;j<img.getDomain()[1];j++){
            temp.operator ()(j,i,0,0)=img.operator ()(i,j).r();
            temp.operator ()(j,i,0,1)=img.operator ()(i,j).g();
            temp.operator ()(j,i,0,2)=img.operator ()(i,j).b();
        }

    _pImpl->display->display(temp);
    return *this;
}

MatNDisplayCImg & MatNDisplayCImg::display(const MatN<2, UI16 > &img){
    Mat2UI8 temp = pop::Processing::greylevelRange(img,0,255);
    return display(temp);
}

MatNDisplayCImg & MatNDisplayCImg::display(const MatN<2, UI32 > &img){
    Mat2UI8 temp = pop::Processing::greylevelRange(img,0,255);
    return display(temp);
}

//MatNDisplayCImg & MatNDisplayCImg::display(const MatN<2, RGBAUI8 > &img){
//    cimg_library::CImg<UI8> temp(img.getDomain()(1),img.getDomain()(0),1,4);
//    for(int i =0;i<img.getDomain()[0];i++)
//        for(int j =0;j<img.getDomain()[1];j++){
//            temp.operator ()(j,i,0,0)=img.operator ()(i,j).r();
//            temp.operator ()(j,i,0,1)=img.operator ()(i,j).g();
//            temp.operator ()(j,i,0,2)=img.operator ()(i,j).b();
//            temp.operator ()(j,i,0,3)=img.operator ()(i,j).a();
//        }

//    _pImpl->display->display(temp);
//    return *this;
//}
MatNDisplayCImg & MatNDisplayCImg::display(const MatN<2, UI8 > &img){
    cimg_library::CImg<UI8> temp(img.getDomain()(1),img.getDomain()(0));
    for(int i =0;i<img.getDomain()[0];i++)
        for(int j =0;j<img.getDomain()[1];j++){
            temp.operator ()(j,i)=img.operator ()(i,j);
        }
    _pImpl->display->display(temp);
    return *this;
}
MatNDisplayCImg & MatNDisplayCImg::display(const MatN<2, F64 > &imgf){
     MatN<2, UI8 > img =Processing::greylevelRange(imgf,0,255);
    cimg_library::CImg<UI8> temp(img.getDomain()(1),img.getDomain()(0));
    for(int i =0;i<img.getDomain()[0];i++)
        for(int j =0;j<img.getDomain()[1];j++){
            temp.operator ()(j,i)=img.operator ()(i,j);
        }
    _pImpl->display->display(temp);
    return *this;
}
MatNDisplayCImg::MatNDisplayCImg(const MatNDisplayCImg &disp){
    _pImpl = new impl();
    _pImpl->display = new cimg_library::CImgDisplay(*disp._pImpl->display);
}




MatNDisplayCImg & 	MatNDisplayCImg::operator= (const MatNDisplayCImg &disp){
    _pImpl->display->assign(*(disp._pImpl->display));
    return *this;
}

MatNDisplayCImg::operator bool () const{
    return _pImpl->display->operator bool();
}
bool 	MatNDisplayCImg::is_empty () const{
    return _pImpl->display->is_empty();
}
bool 	MatNDisplayCImg::is_closed () const{
    return _pImpl->display->is_closed();
}
bool 	MatNDisplayCImg::is_resized () const{
    return _pImpl->display->is_resized();
}
bool 	MatNDisplayCImg::is_moved () const{
    return _pImpl->display->is_moved();
}
bool 	MatNDisplayCImg::is_event () const{
    return _pImpl->display->is_event();
}
bool 	MatNDisplayCImg::is_fullscreen () const{
    return _pImpl->display->is_fullscreen();
}
bool 	MatNDisplayCImg::is_key () const{
    return _pImpl->display->is_key();
}
bool 	MatNDisplayCImg::is_key (const unsigned int keycode) const{
    return _pImpl->display->is_key(keycode);
}
bool 	MatNDisplayCImg::is_key (const char *const keycode) const{
    return _pImpl->display->is_key(keycode);
}
bool 	MatNDisplayCImg::is_key_sequence (const unsigned int *const keycodes_sequence, const unsigned int length, const bool remove_sequence){
    return _pImpl->display->is_key_sequence(keycodes_sequence,length,remove_sequence);
}
bool 	MatNDisplayCImg::is_keyESC () const{
    return _pImpl->display->is_keyESC();
}
bool 	MatNDisplayCImg::is_keyF1 () const{
    return _pImpl->display->is_keyF1();
}
bool 	MatNDisplayCImg::is_keyF2 () const{
    return _pImpl->display->is_keyF2();
}
bool 	MatNDisplayCImg::is_keyF3 () const{
    return _pImpl->display->is_keyF3();
}
bool 	MatNDisplayCImg::is_keyF4 () const{
    return _pImpl->display->is_keyF4();
}
bool 	MatNDisplayCImg::is_keyF5 () const{
    return _pImpl->display->is_keyF5();
}
bool 	MatNDisplayCImg::is_keyF6 () const{
    return _pImpl->display->is_keyF6();
}
bool 	MatNDisplayCImg::is_keyF7 () const{
    return _pImpl->display->is_keyF7();
}
bool 	MatNDisplayCImg::is_keyF8() const{
    return _pImpl->display->is_keyF8();
}
bool 	MatNDisplayCImg::is_keyF9 () const{
    return _pImpl->display->is_keyF9();
}
bool 	MatNDisplayCImg::is_keyF10 () const{
    return _pImpl->display->is_keyF10();
}
bool 	MatNDisplayCImg::is_keyF11 () const{
    return _pImpl->display->is_keyF11();
}
bool 	MatNDisplayCImg::is_keyF12 () const{
    return _pImpl->display->is_keyF12();
}
bool 	MatNDisplayCImg::is_keyPAUSE () const{
    return _pImpl->display->is_keyPAUSE();
}
bool 	MatNDisplayCImg::is_key1 () const{
    return _pImpl->display->is_key1();
}
bool 	MatNDisplayCImg::is_key2 () const{
    return _pImpl->display->is_key2();
}
bool 	MatNDisplayCImg::is_key3 () const{
    return _pImpl->display->is_key3();
}
bool 	MatNDisplayCImg::is_key4 () const{
    return _pImpl->display->is_key4();
}
bool 	MatNDisplayCImg::is_key5 () const{
    return _pImpl->display->is_key5();
}
bool 	MatNDisplayCImg::is_key6 () const{
    return _pImpl->display->is_key6();
}
bool 	MatNDisplayCImg::is_key7 () const{
    return _pImpl->display->is_key7();
}
bool 	MatNDisplayCImg::is_key8 () const{
    return _pImpl->display->is_key8();
}
bool 	MatNDisplayCImg::is_key9 () const{
    return _pImpl->display->is_key9();
}
bool 	MatNDisplayCImg::is_key0 () const{
    return _pImpl->display->is_key0();
}
bool 	MatNDisplayCImg::is_keyBACKSPACE () const{
    return _pImpl->display->is_keyBACKSPACE();
}
bool 	MatNDisplayCImg::is_keyINSERT () const{
    return _pImpl->display->is_keyINSERT();
}
bool 	MatNDisplayCImg::is_keyHOME () const{
    return _pImpl->display->is_keyHOME();
}
bool 	MatNDisplayCImg::is_keyPAGEUP () const{
    return _pImpl->display->is_keyPAGEUP();
}
bool 	MatNDisplayCImg::is_keyTAB () const{
    return _pImpl->display->is_keyTAB();
}
bool 	MatNDisplayCImg::is_keyQ () const{
    return _pImpl->display->is_keyQ();
}
bool 	MatNDisplayCImg::is_keyW () const{
    return _pImpl->display->is_keyW();
}
bool 	MatNDisplayCImg::is_keyE () const{
    return _pImpl->display->is_keyE();
}
bool 	MatNDisplayCImg::is_keyR () const{
    return _pImpl->display->is_keyR();
}
bool 	MatNDisplayCImg::is_keyT () const{
    return _pImpl->display->is_keyT();
}
bool 	MatNDisplayCImg::is_keyY () const{
    return _pImpl->display->is_keyY();
}
bool 	MatNDisplayCImg::is_keyU () const{
    return _pImpl->display->is_keyU();
}
bool 	MatNDisplayCImg::is_keyI () const{
    return _pImpl->display->is_keyI();
}
bool 	MatNDisplayCImg::is_keyO () const{
    return _pImpl->display->is_keyO();
}
bool 	MatNDisplayCImg::is_keyP () const{
    return _pImpl->display->is_keyP();
}
bool 	MatNDisplayCImg::is_keyDELETE () const{
    return _pImpl->display->is_keyDELETE();
}
bool 	MatNDisplayCImg::is_keyEND () const{
    return _pImpl->display->is_keyEND();
}
bool 	MatNDisplayCImg::is_keyPAGEDOWN () const{
    return _pImpl->display->is_keyPAGEDOWN();
}
bool 	MatNDisplayCImg::is_keyCAPSLOCK () const{
    return _pImpl->display->is_keyCAPSLOCK();
}
bool 	MatNDisplayCImg::is_keyA () const{
    return _pImpl->display->is_keyA();
}
bool 	MatNDisplayCImg::is_keyS () const{
    return _pImpl->display->is_keyS();
}
bool 	MatNDisplayCImg::is_keyD () const{
    return _pImpl->display->is_keyD();
}
bool 	MatNDisplayCImg::is_keyF () const{
    return _pImpl->display->is_keyF();
}
bool 	MatNDisplayCImg::is_keyG () const{
    return _pImpl->display->is_keyG();
}
bool 	MatNDisplayCImg::is_keyH () const{
    return _pImpl->display->is_keyH();
}
bool 	MatNDisplayCImg::is_keyJ () const{
    return _pImpl->display->is_keyJ();
}
bool 	MatNDisplayCImg::is_keyK () const{
    return _pImpl->display->is_keyK();
}
bool 	MatNDisplayCImg::is_keyL () const{
    return _pImpl->display->is_keyL();
}
bool 	MatNDisplayCImg::is_keyENTER () const{
    return _pImpl->display->is_keyENTER();
}
bool 	MatNDisplayCImg::is_keySHIFTLEFT () const{
    return _pImpl->display->is_keySHIFTLEFT();
}
bool 	MatNDisplayCImg::is_keyZ () const{
    return _pImpl->display->is_keyZ();
}
bool 	MatNDisplayCImg::is_keyX () const{
    return _pImpl->display->is_keyX();
}
bool 	MatNDisplayCImg::is_keyC () const{
    return _pImpl->display->is_keyC();
}
bool 	MatNDisplayCImg::is_keyV () const{
    return _pImpl->display->is_keyV();
}
bool 	MatNDisplayCImg::is_keyB () const{
    return _pImpl->display->is_keyB();
}
bool 	MatNDisplayCImg::is_keyN () const{
    return _pImpl->display->is_keyN();
}
bool 	MatNDisplayCImg::is_keyM () const{
    return _pImpl->display->is_keyM();
}
bool 	MatNDisplayCImg::is_keySHIFTRIGHT () const{
    return _pImpl->display->is_keySHIFTRIGHT();
}
bool 	MatNDisplayCImg::is_keyARROWUP () const{
    return _pImpl->display->is_keyARROWUP();
}
bool 	MatNDisplayCImg::is_keyCTRLLEFT () const{
    return _pImpl->display->is_keyCTRLLEFT();
}
bool 	MatNDisplayCImg::is_keyAPPLEFT () const{
    return _pImpl->display->is_keyAPPLEFT();
}
bool 	MatNDisplayCImg::is_keyALT () const{
    return _pImpl->display->is_keyALT();
}
bool 	MatNDisplayCImg::is_keySPACE () const{
    return _pImpl->display->is_keySPACE();
}
bool 	MatNDisplayCImg::is_keyALTGR () const{
    return _pImpl->display->is_keyALTGR();
}
bool 	MatNDisplayCImg::is_keyAPPRIGHT () const{
    return _pImpl->display->is_keyAPPRIGHT();
}
bool 	MatNDisplayCImg::is_keyMENU () const{
    return _pImpl->display->is_keyMENU();
}
bool 	MatNDisplayCImg::is_keyCTRLRIGHT () const{
    return _pImpl->display->is_keyCTRLRIGHT();
}
bool 	MatNDisplayCImg::is_keyARROWLEFT () const{
    return _pImpl->display->is_keyARROWLEFT();
}
bool 	MatNDisplayCImg::is_keyARROWDOWN () const{
    return _pImpl->display->is_keyARROWDOWN();
}
bool 	MatNDisplayCImg::is_keyARROWRIGHT () const{
    return _pImpl->display->is_keyARROWRIGHT();
}
bool 	MatNDisplayCImg::is_keyPAD0 () const{
    return _pImpl->display->is_keyPAD0();
}
bool 	MatNDisplayCImg::is_keyPAD1 () const{
    return _pImpl->display->is_keyPAD1();
}
bool 	MatNDisplayCImg::is_keyPAD2 () const{
    return _pImpl->display->is_keyPAD2();
}
bool 	MatNDisplayCImg::is_keyPAD3 () const{
    return _pImpl->display->is_keyPAD3();
}
bool 	MatNDisplayCImg::is_keyPAD4 () const{
    return _pImpl->display->is_keyPAD4();
}
bool 	MatNDisplayCImg::is_keyPAD5 () const{
    return _pImpl->display->is_keyPAD5();
}
bool 	MatNDisplayCImg::is_keyPAD6 () const{
    return _pImpl->display->is_keyPAD6();
}
bool 	MatNDisplayCImg::is_keyPAD7 () const{
    return _pImpl->display->is_keyPAD7();
}
bool 	MatNDisplayCImg::is_keyPAD8 () const{
    return _pImpl->display->is_keyPAD8();
}
bool 	MatNDisplayCImg::is_keyPAD9 () const{
    return _pImpl->display->is_keyPAD9();
}
bool 	MatNDisplayCImg::is_keyPADADD () const{
    return _pImpl->display->is_keyPADADD();
}
bool 	MatNDisplayCImg::is_keyPADSUB () const{
    return _pImpl->display->is_keyPADSUB();
}
bool 	MatNDisplayCImg::is_keyPADMUL () const{
    return _pImpl->display->is_keyPADMUL();
}
bool 	MatNDisplayCImg::is_keyPADDIV () const{
    return _pImpl->display->is_keyPADDIV();
}
int 	MatNDisplayCImg::width () const{
    return _pImpl->display->width();
}
int 	MatNDisplayCImg::height () const{
    return _pImpl->display->height();
}
unsigned int 	MatNDisplayCImg::normalization () const{
    return _pImpl->display->normalization();
}
const char * 	MatNDisplayCImg::title () const{
    return _pImpl->display->title();
}
int 	MatNDisplayCImg::window_width () const{
    return _pImpl->display->window_width();
}
int 	MatNDisplayCImg::window_height () const{
    return _pImpl->display->window_height();
}
int 	MatNDisplayCImg::window_x () const{
    return _pImpl->display->window_x();
}
int 	MatNDisplayCImg::window_y () const{
    return _pImpl->display->window_y();
}
int 	MatNDisplayCImg::mouse_x () const{
    return _pImpl->display->mouse_x();
}
int 	MatNDisplayCImg::mouse_y () const{
    return _pImpl->display->mouse_y();
}
unsigned int 	MatNDisplayCImg::button () const{
    return _pImpl->display->button();
}
int 	MatNDisplayCImg::wheel () const{
    return _pImpl->display->wheel();
}
unsigned int 	MatNDisplayCImg::key (const unsigned int pos) const{
    return _pImpl->display->key(pos);
}
unsigned int 	MatNDisplayCImg::released_key (const unsigned int pos) const{
    return _pImpl->display->released_key(pos);
}
float 	MatNDisplayCImg::frames_per_second (){
    return _pImpl->display->frames_per_second();
}
int 	MatNDisplayCImg::screen_width (){
    return _pImpl->display->screen_width();
}
int 	MatNDisplayCImg::screen_height (){
    return _pImpl->display->screen_height();
}
unsigned int 	MatNDisplayCImg::keycode (const char *const keycode){
    return _pImpl->display->keycode(keycode);
}
MatNDisplayCImg & 	MatNDisplayCImg::show (){
    _pImpl->display->show();
    return *this;
}
MatNDisplayCImg & 	MatNDisplayCImg::close (){
    _pImpl->display->close();
    return *this;
}
MatNDisplayCImg & 	MatNDisplayCImg::move (const int pos_x, const int pos_y){
    _pImpl->display->move(pos_x,pos_y);
    return *this;
}
MatNDisplayCImg & 	MatNDisplayCImg::resize (const bool force_redraw){
    _pImpl->display->resize(force_redraw);
    return *this;
}
MatNDisplayCImg & 	MatNDisplayCImg::resize (const int width, const int height, const bool force_redraw){
    _pImpl->display->resize(width,height,force_redraw);
    return *this;
}

MatNDisplayCImg & 	MatNDisplayCImg::resize (const MatNDisplayCImg &disp, const bool force_redraw){
    _pImpl->display->resize(disp,force_redraw);
    return *this;
}
MatNDisplayCImg & 	MatNDisplayCImg::set_normalization (const unsigned int normalization){
    _pImpl->display->set_normalization(normalization);
    return *this;
}
MatNDisplayCImg & 	MatNDisplayCImg::set_title (const char *const format,...){
    _pImpl->display->set_title(format);
    return *this;
}
MatNDisplayCImg & 	MatNDisplayCImg::set_fullscreen (const bool is_fullscreen, const bool force_redraw){
    _pImpl->display->set_fullscreen(is_fullscreen,force_redraw);
    return *this;
}
MatNDisplayCImg & 	MatNDisplayCImg::toggle_fullscreen (const bool force_redraw){
    _pImpl->display->toggle_fullscreen(force_redraw);
    return *this;
}
MatNDisplayCImg & 	MatNDisplayCImg::show_mouse (){
    _pImpl->display->show_mouse();
    return *this;
}
MatNDisplayCImg & 	MatNDisplayCImg::hide_mouse (){
    _pImpl->display->hide_mouse();
    return *this;
}
MatNDisplayCImg & 	MatNDisplayCImg::set_mouse (const int pos_x, const int pos_y){
    _pImpl->display->set_mouse(pos_x,pos_y);
    return *this;
}
MatNDisplayCImg & 	MatNDisplayCImg::set_button (){
    _pImpl->display->set_button();
    return *this;
}
MatNDisplayCImg & 	MatNDisplayCImg::set_button (const unsigned int button, const bool is_pressed){
    _pImpl->display->set_button(button,  is_pressed);
    return *this;
}
MatNDisplayCImg & 	MatNDisplayCImg::set_wheel (){
    _pImpl->display->set_wheel();
    return *this;
}
MatNDisplayCImg & 	MatNDisplayCImg::set_wheel (const int amplitude){
    _pImpl->display->set_wheel(amplitude);
    return *this;
}
MatNDisplayCImg & 	MatNDisplayCImg::set_key (){
    _pImpl->display->set_key();
    return *this;
}
MatNDisplayCImg & 	MatNDisplayCImg::set_key (const unsigned int keycode, const bool is_pressed){
    _pImpl->display->set_key( keycode,  is_pressed);
    return *this;
}
MatNDisplayCImg & 	MatNDisplayCImg::flush (){
    _pImpl->display->flush();
    return *this;
}

MatNDisplayCImg & 	MatNDisplayCImg::waitTime (){
    _pImpl->display->wait();
    return *this;
}
MatNDisplayCImg & 	MatNDisplayCImg::waitTime (const unsigned int milliseconds){
    _pImpl->display->wait(milliseconds);
    return *this;
}

MatNDisplayCImg & 	MatNDisplayCImg::paint (){
    _pImpl->display->paint();
    return *this;
}

void 	MatNDisplayCImg::waitTime (MatNDisplayCImg &disp1){
    _pImpl->display->wait(*disp1._pImpl->display);
}
void 	MatNDisplayCImg::waitTime (MatNDisplayCImg &disp1, MatNDisplayCImg &disp2){
    _pImpl->display->wait(*disp1._pImpl->display,*disp2._pImpl->display);
}
void 	MatNDisplayCImg::waitTime (MatNDisplayCImg &disp1, MatNDisplayCImg &disp2, MatNDisplayCImg &disp3){
    _pImpl->display->wait(*disp1._pImpl->display,*disp2._pImpl->display,*disp3._pImpl->display);
}
void 	MatNDisplayCImg::waitTime (MatNDisplayCImg &disp1, MatNDisplayCImg &disp2, MatNDisplayCImg &disp3, MatNDisplayCImg &disp4){
    _pImpl->display->wait(*disp1._pImpl->display,*disp2._pImpl->display,*disp3._pImpl->display,*disp4._pImpl->display);
}
void 	MatNDisplayCImg::waitTime (MatNDisplayCImg &disp1, MatNDisplayCImg &disp2, MatNDisplayCImg &disp3, MatNDisplayCImg &disp4, MatNDisplayCImg &disp5){
    _pImpl->display->wait(*disp1._pImpl->display,*disp2._pImpl->display,*disp3._pImpl->display,*disp4._pImpl->display,*disp5._pImpl->display);
}
void 	MatNDisplayCImg::wait_all (){
    _pImpl->display->wait_all();
}
}
#endif
