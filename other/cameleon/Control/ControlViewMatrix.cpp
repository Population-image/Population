#include "ControlViewMatrix.h"

#include "DataMatrix.h"
#include "data/utility/UtilitySTL.h"
ControlViewMatrix::ControlViewMatrix(QWidget * parent)
    :CControl(parent)
{
    this->path().push_back("Matrix");
    this->setName("ViewMatrix");
    this->setKey("ControlViewMatrix");
    this->structurePlug().addPlugIn(DataMatrix::KEY,"in.m");

    box = new QTableWidget;
    box->setRowCount(0);
    box->setColumnCount(0);

    QVBoxLayout *lay = new QVBoxLayout;
    lay->addWidget(box);
    this->setLayout(lay);
    this->setMinimumWidth(300);
}


CControl * ControlViewMatrix::clone(){
    return new ControlViewMatrix();
}



void ControlViewMatrix::updatePlugInControl(int, CData* data){

        box->clear();

        if(DataMatrix* matrix = dynamic_cast<DataMatrix*>(data)){
            shared_ptr<Mat2F64> st = matrix->getData();
            box->setRowCount(st->sizeI());
            box->setColumnCount(st->sizeJ());
            for(int i=0;i<st->sizeI();i++){
                for(int j=0;j<st->sizeJ();j++){
                    QTableWidgetItem* item = new QTableWidgetItem(QString::number(st->operator ()(i,j)));
                    box->setItem(i,j,item);
                }
            }
            for(int j=0;j<st->sizeJ();j++){
                string header = "j_"+UtilityString::Any2String(j);
                QTableWidgetItem* itemName = new QTableWidgetItem(header.c_str());
                box->setHorizontalHeaderItem(j,itemName);
            }
            for(int j=0;j<st->sizeI();j++){
                string header = "i_"+UtilityString::Any2String(j);
                QTableWidgetItem* itemName = new QTableWidgetItem(header.c_str());
                box->setVerticalHeaderItem(j,itemName);
            }
        }
        this->update();

}
