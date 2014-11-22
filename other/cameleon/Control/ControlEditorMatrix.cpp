#include "ControlEditorMatrix.h"
#include<DataMatrix.h>
ControlEditorMatrix::ControlEditorMatrix(QWidget *parent)
    : EditorTable(parent)
{
    this->path().clear();
    this->path().push_back("Matrix");
    this->setName("EditorMatrix");
    this->setKey("EditorMatrix");
    this->structurePlug().plugOut().clear();
    this->structurePlug().addPlugOut(DataMatrix::KEY,"out.m");
//    this->addRaw();this->addRaw();this->addRaw();
//    this->addColumn();this->addColumn();this->addColumn();
}
QString ControlEditorMatrix::defaultString(){
    return "0";
}

QString ControlEditorMatrix::defaultHeaderColumn(int col){
    return QString::number(col);
}

QString ControlEditorMatrix::defaultHeaderRaw(int raw){
    return QString::number(raw);
}
CControl * ControlEditorMatrix::clone(){
    return new ControlEditorMatrix;
}

void ControlEditorMatrix::apply(){
    if(test==true&&this->isPlugOutConnected(0)==true)
    {
        int rcount = box->rowCount();
        int ccount = box->columnCount();
        Mat2F64* t = new Mat2F64(rcount,ccount);
        for(int i=0;i<rcount;i++){
            for(int j=0;j<ccount;j++){
                QTableWidgetItem* item = box->item(i,j);
                t->operator ()(i,j)=item->text().toDouble();
            }
        }
        DataMatrix* data = new DataMatrix;
        shared_ptr<Mat2F64> st(t);
        data->setData(st);
        this->sendPlugOutControl(0,data,CPlug::NEW);
    }
}
