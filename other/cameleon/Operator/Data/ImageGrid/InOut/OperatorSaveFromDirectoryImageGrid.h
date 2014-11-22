#ifndef OPERATORSAVEFROMDIRECTORYMatN_H
#define OPERATORSAVEFROMDIRECTORYMatN_H

#include"COperator.h"
#include<DataImageGrid.h>
using namespace pop;
class OperatorSaveFromDirectoryMatN: public COperator
{
public:
    OperatorSaveFromDirectoryMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<typename Type>
        void operator()(MatN<3,Type> * in1cast,string path, string file, string extension)
        {
            in1cast->saveFromDirectory(path.c_str(),file.c_str(),extension.c_str());


        }
    };
};
#endif // OPERATORSAVEFROMDIRECTORYMatN_H
