#include "DataMatrix.h"
DataMatrix::DataMatrix()
    :CDataByFile<Mat2F64>()
{
    _data = shared_ptr<Mat2F64>(new Mat2F64);
    this->_key = DataMatrix::KEY;
    this->setExtension(".pgm");
    this->setMode(CData::BYCOPY);
}
string DataMatrix::KEY ="DATAMATRIX";
DataMatrix *DataMatrix::clone(){
    return new DataMatrix();
}
void DataMatrix::setDataByFile(shared_ptr<Mat2F64> type){

    type->save(this->getFile().c_str());
}
void DataMatrix::setDataByCopy(shared_ptr<Mat2F64> type){
    Mat2F64* m = new Mat2F64(*(type.get()));
    _data = shared_ptr<Mat2F64> (m);
}

shared_ptr<Mat2F64> DataMatrix::getDataByFile(){
    shared_ptr<Mat2F64> t(new Mat2F64);
    t->load(this->getFile().c_str());
    return t;
}
