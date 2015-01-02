/******************************************************************************\
|*                   Population library for C++ X.X.X                         *|
|*----------------------------------------------------------------------------*|
The Population License is similar to the MIT license in adding this clause:
for any writing public or private that has resulted from the use of the
software population, the reference of this book "Population library, 2012,
Vincent Tariel" shall be included in it.

So, the terms of the Population License are:

Copyright © 2012-2015, Tariel Vincent

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software and for any writing
public or private that has resulted from the use of the software population,
the reference of this book "Population library, 2012, Vincent Tariel" shall
be included in it.

The Software is provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising
from, out of or in connection with the software or the use or other dealings
in the Software.
\***************************************************************************/

#ifndef POPTEST_H
#define POPTEST_H
#include"Population.h"
namespace pop
{

struct PopTest
{
   std::string _name_operator;
   bool _bool_write;
   clock_t _start_time, _end_time;
   PopTest(){
       _bool_write =false;
   }


   void start(std::string name_operator, std::string param=std::string()){
       _name_operator = name_operator;
       _start_time = clock();
   }
  template<int DIM,typename Type>
   void end(MatN<DIM,Type> out){
       MatN<DIM,Type> out_algo;

       _end_time = clock();
       std::string name_file =  std::string(POP_PROJECT_SOURCE_DIR)+"/other/test/"+_name_operator+".png";
       if(_bool_write==true){
           out.save(name_file.c_str());
       }else{
            if(pop::BasicUtility::isFile(name_file)){
                out_algo.load(name_file);
                if(out != out_algo){
                    std::cout<<"[ERROR]["+ _name_operator +"] Result is differente and execute in " << (double) (_end_time - _start_time) / CLOCKS_PER_SEC<<std::endl;
                    return;
                }

            }else {
                std::cout<<"[ERROR]["+ _name_operator +"] Cannot load the file"<<std::endl;
                return;
            }
            std::cout<<"[GOOD]["+ _name_operator +"] input size image= "<<out.getDomain()<<" and execute in " << (double) (_end_time - _start_time) / CLOCKS_PER_SEC<<std::endl;
       }
   }
   inline void end(){
       _end_time = clock();
      std::cout<<"[GOOD]["+ _name_operator +"]  execute in " << (double) (_end_time - _start_time) / CLOCKS_PER_SEC<<std::endl;
   }

};
}

#endif // POPTEST_H
