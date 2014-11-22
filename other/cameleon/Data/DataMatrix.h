#ifndef DATAMATRIX_H
#define DATAMATRIX_H

#include<CDataByFile.h>
#include"data/mat/MatN.h"
using namespace pop;
class DataMatrix: public CDataByFile<Mat2F64>
{
public:
    DataMatrix();
    static string KEY;
    DataMatrix* clone();
    void setDataByFile(shared_ptr<Mat2F64> type);
    void setDataByCopy(shared_ptr<Mat2F64> type);
    shared_ptr<Mat2F64> getDataByFile();
};

#endif // DATAMATRIX_H
