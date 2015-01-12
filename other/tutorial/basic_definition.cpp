#include"Population.h"//Single header
using namespace pop;//Population namespace
void test0(){
    std::cout<<"test0"<<std::endl;
    Mat2UI8 m(10,10);//create a 2d matrix with 10-row and the 10-columns and with UI8 pixel type (0,1,2...,255)
    int i=4;
    int j=6;
    UI8 intensity = 200;//set the intensity at 200
    m(i,j) = intensity;//set the pixel value at 200
    intensity = m(i,j);//access the pixel value at 4-row and the 6-column
    std::cout<<m<<std::endl;
}
void test1(){
    std::cout<<"\ntest1"<<std::endl;
    Mat2F64 m(4,3);//create a 2d matrix with 4-row and the 3-columns and with float pixel type
    Vec2I32 x(2,1);
    m(x) = 2.45;//set the pixel value at 2.45
    std::cout<<m<<std::endl;
}
void test2(){
    std::cout<<"\ntest2"<<std::endl;
    Mat2RGBUI8 m(10,8);//create a 2d matrix with rgb pixel type
    Vec2I32 x(4,6);
    RGBUI8 value;
    value.r()=200;//set the red channel at 200
    value.g()=100;//set the green channel at 100
    value.b()=210;//set the blue channel at 210
    m(x) = value;//set the pixel value
    m.display("test 2",false);
}
void test3(){
    std::cout<<"\ntest3"<<std::endl;
    Mat2UI8 m(6,5);//construct a 2d matrix with 1 byte pixel type with 6 rows and 5 columns
    m(0,0)= 20;m(0,1)= 20;m(0,2)= 20;m(0,3)= 20;m(0,4)= 20;//set the pixel values of the first row
    m(1,0)= 20;m(1,1)=255;m(1,2)= 20;m(1,3)=255;m(1,4)= 20;//set the pixel values of the second row
    m(2,0)= 20;m(2,1)= 20;m(2,2)=255;m(2,3)= 20;m(2,4)= 20;//set the pixel values of the third row
    m(3,0)= 20;m(3,1)= 20;m(3,2)= 20;m(3,3)= 20;m(3,4)= 20;//set the pixel values of the fourth row
    m(4,0)= 20;m(4,1)=150;m(4,2)=150;m(4,3)=150;m(4,4)= 20;//set the pixel values of the fifth row
    m(5,0)= 20;m(5,1)= 20;m(5,2)= 20;m(5,3)= 20;m(5,4)= 20;//set the pixel values of the sixth row
    std::cout<<m<<std::endl;//display the array (left figure)
    m.display("test 3");//display the image (right figure)
}
void test4(){
    std::cout<<"\n test4"<<std::endl;
    Mat3UI8 m(3,3,4);//create a 3d matrix with 3-row, 3-columns and 4-depth and UI8 voxel type
    Vec3I32 x(1,0,1);
    m(x) = 200;//set the pixel value
    std::cout<<m<<std::endl;
}
int main()
{
    test0();
    test1();
    test2();
    test4();
    test3();

}
