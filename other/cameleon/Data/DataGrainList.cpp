#include "DataGrainList.h"



DataGermGrain::DataGermGrain()
    :CDataByFile<GermGrainMother>()
{
    this->_key = DataGermGrain::KEY;
    this->setExtension(".grain");
    this->setMode(CData::BYFILE);
}
string DataGermGrain::KEY ="DATAGrainList";
DataGermGrain * DataGermGrain::clone(){
    return new DataGermGrain();
}


shared_ptr<GermGrainMother> DataGermGrain::getDataByFile(){
    std::ifstream  in(this->getFile().c_str());
    return shared_ptr<GermGrainMother>(GermGrainMother::create(in));
}

void DataGermGrain::setDataByFile(shared_ptr<GermGrainMother> type){
    std::ofstream  out(this->getFile().c_str());
    type->GermGrainMother::save(out);
    type->save(out);
    out.close();
}
void DataGermGrain::setDataByCopy(shared_ptr<GermGrainMother> type){
    if(type->dim==2){
        GermGrain2 * phiin = new GermGrain2;
        * phiin= *dynamic_cast<GermGrain2 * >(type.get());
        this->_data = shared_ptr<GermGrainMother>(phiin);

    }else{
        GermGrain3 * phiin = new GermGrain3;
        * phiin= *dynamic_cast<GermGrain3 * >(type.get());
        this->_data = shared_ptr<GermGrainMother>(phiin);
    }

}
