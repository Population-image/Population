#include "ControlViewPoint.h"

#include "DataPoint.h"
#include "data/utility/UtilitySTL.h"
ControlViewPoint::ControlViewPoint(QWidget * parent)
    :CControl(parent)
{
    this->path().push_back("Point");
    this->setName("ViewPoint");
    this->setKey("ControlViewPoint");
    this->structurePlug().addPlugIn(DataPoint::KEY,"in.v");
    box = new QTableWidget;
    box->setRowCount(1);
    box->setColumnCount(0);

    QVBoxLayout *lay = new QVBoxLayout;
    lay->addWidget(box);
    this->setLayout(lay);
    this->setMinimumWidth(300);
}


CControl * ControlViewPoint::clone(){
    return new ControlViewPoint();
}



void ControlViewPoint::updatePlugInControl(int, CData* data){

        box->clear();

        if(DataPoint * Point = dynamic_cast<DataPoint *>(data)){
            VecF64  v = Point->getValue();
            box->setColumnCount(v.size());
            for(int i=0;i<v.size();i++){

                    QTableWidgetItem* item = new QTableWidgetItem(QString::number(v(i)));
                    box->setItem(0,i,item);

            }
            for(int j=0;j<v.size();j++){
                string header = "x_"+UtilityString::Any2String(j);
                QTableWidgetItem* itemName = new QTableWidgetItem(header.c_str());
                box->setHorizontalHeaderItem(j,itemName);
            }
        }
        this->update();

}
