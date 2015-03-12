#include "data/utility/XML.h"

#include <sstream>
namespace pop{

XMLNode::XMLNode()
{
}
XMLNode::XMLNode(const XMLNode & node)
{
    this->_node = node._node;
}
XMLNode& XMLNode::operator ()(const XMLNode & node){
    this->_node = node._node;
    return *this;
}
void XMLNode::setName(std::string name){
    this->_node.set_name(name.c_str());
}
void XMLNode::setValue(std::string value){
    //    this->_node.set_value(value.c_str());
    if(this->_node.first_child().set_value(value.c_str()) == false)
    {
        this->_node.append_child(pugi::node_pcdata).set_value(value.c_str());
    }
}
XMLNode XMLNode::addChild(std::string name){
    XMLNode nod;
    nod._node=  this->_node.append_child(name.c_str());
    return nod;
}
bool XMLNode::rmChild(const XMLNode& node){
    return this->_node.remove_child(node._node);
}
std::string XMLNode::getName()const{
    return this->_node.name();
}
std::string XMLNode::getValue()const{
    return this->_node.first_child().value();
}
XMLNode XMLNode::getChild(std::string childname)const{
    XMLNode nod;
    nod._node= this->_node.child(childname.c_str());
    return nod;
}
XMLNode XMLNode::firstChild()const{
    XMLNode nod;
    nod._node= this->_node.first_child();
    return nod;
}
XMLNode::operator bool() const {
    return! this->_node.operator !();
}


XMLNode XMLNode::nextSibling()const{
    XMLNode nod;
    nod._node= this->_node.next_sibling();
    return nod;
}
bool XMLNode::hasAttribute(std::string attributename)const{
    if(this->_node.attribute(attributename.c_str()))
        return true;
    else
        return false;
}

std::string XMLNode::getAttribute(std::string attributename) const{
    std::string test = this->_node.attribute(attributename.c_str()).value();;
    return test;
}
void XMLNode::setAttribute(std::string name,std::string  value){
    this->_node.attribute(name.c_str()).set_value(value.c_str());
}
void XMLNode::addAttribute(std::string name,std::string  value){
    this->_node.append_attribute(name.c_str()) = value.c_str();
}
bool XMLNode::rmAttribute(std::string name){
    return this->_node.remove_attribute(name.c_str());
}


XMLDocument::XMLDocument()
{
}


void XMLDocument::load(std::string file){
    this->_doc.load_file(file.c_str());
}
void  XMLDocument::loadFromByteArray(const char *  file){
    this->_doc.load(file);
}

//void XMLDocument::load(std::istream & is){
//    std::string str;
//    while(is >> str);
//    this->_doc.load_buffer(str.c_str(),str.size());
//}

void XMLDocument::save(std::string file)const {
    this->_doc.save_file(file.c_str());
}
//void save(std::ostream file)const {

//}

XMLNode XMLDocument::getChild(std::string name){
    XMLNode node;
    node._node = this->_doc.child(name.c_str());
    return node;
}

XMLNode XMLDocument::addChild(std::string name){
    XMLNode node;
    node._node =  this->_doc.append_child(name.c_str());
    return node;
}

}
