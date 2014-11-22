#include "ControlViewImageGridValue.h"

#include "DataImageGrid.h"
#include "data/utility/UtilitySTL.h"
ControlViewMatNValue::ControlViewMatNValue(QWidget * parent)
    :CControl(parent)
{
    this->path().push_back("ImageGrid");
    this->setName("ViewMatNValue");
    this->setKey("ControlViewMatNValue");
    this->structurePlug().addPlugIn(DataMatN::KEY,"in.pgm");

    box = new QTableWidget;
    box->setRowCount(0);
    box->setColumnCount(0);

    QVBoxLayout *lay = new QVBoxLayout;
    lay->addWidget(box);
    this->setLayout(lay);
    this->setMinimumWidth(300);
}


CControl * ControlViewMatNValue::clone(){
    return new ControlViewMatNValue();
}



void ControlViewMatNValue::updatePlugInControl(int, CData* data){

        box->clear();

        if(DataMatN * datac = dynamic_cast<DataMatN *>(data)){
            shared_ptr<BaseMatN> f = datac->getData();
            BaseMatN * fc = f.get();
            foo func;
            typedef  FilterKeepTlistTlist<TListImgGrid,0,Loki::Int2Type<2> >::Result ListFilter;

            try{Dynamic2Static<ListFilter>::Switch(func, fc, box, Loki::Type2Type<MatN<2,int> >());}
            catch(pexception msg){
                this->error("Pixel/voxel type of input image must be registered type");
                return;
            }
        }
        this->update();

}
