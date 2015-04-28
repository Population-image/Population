#include"Population.h"
#include<iostream>
#include"neuralnetworkmatrix.h"
#include"data/notstable/graph/Graph.h"
#include"popconfig.h"
using namespace pop;

struct Line{

    static Vec<Vec2F32> pointInLine(Vec2F32 xmin,Vec2F32 xmax, F32 step){
        Vec2F32 direction = xmax-xmin;
        F32 distance = direction.norm(2);
        direction/=distance;
        Vec2F32 x=xmin;
        Vec<Vec2F32> v_points;
        for(unsigned int i = 0;i<distance/step;i++){
            v_points.push_back(x);
            x +=direction;
        }
        return v_points;
    }
    template<typename PixelType>
    static F32 mean(const Vec<Vec2F32>& line,const MatN<2,PixelType> & m  ){
        F32 sum = 0;
        for(unsigned int i=0;i<line.size();i++){
            if(m.isValid(line(i))){
                sum+=m.interpolationBilinear(line(i));
            }
        }
        return sum/line.size();
    }
    template<typename PixelType>
    static Vec2F32  barycentre(const Vec<Vec2F32>& line,const MatN<2,PixelType> & m  ){
        Vec2F32 barycentre(0,0);
        F32 sum = 0;
        for(unsigned int i=0;i<line.size();i++){
            if(m.isValid(line(i))){
                sum+=m.interpolationBilinear(line(i));
                barycentre += (sum*line(i));
            }
        }
        return barycentre/sum;
    }



};

template<typename PixelType>
void findLine( MatN<2,PixelType>  m){
    int value;
    MatN<2,PixelType> m_thresh = Processing::thresholdOtsuMethod(m,value);
    //    m_thresh = Processing::opening(m_thresh,1);
    Mat2UI32 imglabel =Processing::clusterToLabel(m_thresh);

    GeometricalTransformation::scale(Visualization::labelToRandomRGB(imglabel),Vec2F32(3,3)).display("false",true,false);

    Vec<int> v_count;
    Vec<Vec2F32> v_bary;
    ForEachDomain2D(x,imglabel){
        int label = imglabel(x);
        if(label>0){
            if(label>v_count.size()){
                v_count.resize(label);
                v_bary.resize(label);
            }
            v_count(label-1)++;
            v_bary(label-1)+=x;
        }
    }

    Vec<LinearLeastSquareRANSACModel::Data> data;

    for(unsigned int i=0;i<v_bary.size();i++){
        v_bary(i)=v_bary(i)/v_count(i);
        VecF32 x(2);F32 y;
        x(0)=1;x(1)=v_bary(i)(1);y=v_bary(i)(0);data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        std::cout<<v_bary(i)(1)<<" "<<v_bary(i)(0)<<std::endl;

        m(v_bary(i))=255;
    }
    GeometricalTransformation::scale(m,Vec2F32(3,3)).display("point ",true,false);


    //    F32 mean_value =Analysis::meanValue(m);
    //    Vec<LinearLeastSquareRANSACModel::Data> data;

    //    for(unsigned int j=0;j<m.sizeJ();j++){
    //        F32 sum = 0;
    //        F32 barycentre = 0;
    //        for(unsigned int i=0;i<m.sizeI();i++){
    //            barycentre+=i*m(i,j);
    //            sum+=m(i,j);
    //        }
    //        barycentre/=sum;
    //        sum/=m.sizeI();
    //        if(sum>mean_value){

    //            VecF32 x(2);F32 y;
    //            x(0)=1;x(1)=j;y=barycentre;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
    //            std::cout<<j<<" "<<barycentre<<std::endl;
    //        }
    //    }
    LinearLeastSquareRANSACModel model;
    Vec<LinearLeastSquareRANSACModel::Data> dataconsencus;
    ransac(data,1000,1,15,model,dataconsencus);
    std::cout<<model.getBeta()<<std::endl;
    std::cout<<model.getError()<<std::endl;
    std::cout<<dataconsencus<<std::endl;

    Vec2I32 x1(model.getBeta()(0),0);
    Vec2I32 x2(model.getBeta()(0)+m.sizeJ()*model.getBeta()(1),m.sizeJ());
    Draw::line(m,x1,x2,128);
    GeometricalTransformation::scale(m,Vec2F32(3,3)).display("rotate",true,false);
    ;
    double angle = std::atan(model.getBeta()(1));
    std::cout<<angle<<std::endl;
    m= GeometricalTransformation::scale(m,Vec2F32(3,3));
    GeometricalTransformation::rotate(m,-angle).display("rotate",true,false);
}



struct NeuralLayer
{
    /** @brief Using the CPU device, compute the output values . */
    virtual void forwardCPU(const NeuralLayer& layer_previous) = 0;

    /** @brief Using the CPU device, compute the error of the output values of the layer prrevious. */
    virtual void backwardCPU(NeuralLayer& layer_previous) = 0;
    virtual void learn()=0;


    /** @brief get output value */
    virtual const VecF32& X()const=0;
    virtual VecF32& X()=0;
    /** @brief get output value */
    virtual VecF32& d_E_X()=0;
    /** @brief set the layer to be trainable */
    virtual void setTrainable(bool istrainable)=0;


    void setLearnableParameter(F32 mu){
        _mu = mu;
    }
protected:
    F32 _mu;
};


struct NeuronSigmoid
{
    F32 activation(F32 y){ return 1.7159*tanh(0.66666667*y);}
    F32 derivedActivation(F32 ,F32 x){ return 0.666667f/1.7159f*(1.7159f+(x))*(1.7159f-(x));}  // derivative of the sigmoid as a function of the sigmoid's output

};

struct NeuralLayerInput : public NeuralLayer
{

     void forwardCPU(const NeuralLayer& layer_previous) {}
     void backwardCPU(NeuralLayer& layer_previous) {}
     void learn(){}
     const VecF32& X()const{return _X;}
     VecF32& X(){return _X;}
     VecF32& d_E_X(){return _X;}
     void setTrainable(bool ){}
private:
    VecF32 _X;
};
struct NeuralLayerInputMatrix : public NeuralLayerMatrix
{

     void forwardCPU(const NeuralLayer& layer_previous) {}
     void backwardCPU(NeuralLayer& layer_previous) {}
     void learn(){}
     const VecF32& X()const{return _X;}
     VecF32& X(){return _X;}
     VecF32& d_E_X(){return _X;}
     void setTrainable(bool ){}
private:
    VecF32 _X;
};


template<typename Neuron=NeuronSigmoid>
struct NeuralLayerFullyConnected : public Neuron,public NeuralLayer
{
    NeuralLayerFullyConnected(int nbr_neurons_previous,int nbr_neurons)
        :_Y(nbr_neurons),_X(nbr_neurons),_W(nbr_neurons,nbr_neurons_previous+1),_X_biais(nbr_neurons_previous+1,1)
    {

        DistributionNormal n(0,1./std::sqrt(nbr_neurons_previous+1));
        for(unsigned int i=0;i<_W.size();i++){
            _W(i)=n.randomVariable();
        }
    }
    void setTrainable(bool istrainable){
        if(istrainable==true){
            _d_E_Y = _Y;
            _d_E_W = _W;
            _d_E_X = _X;
        }else{
            _d_E_Y.clear();
            _d_E_W.clear();
            _d_E_X.clear();
        }
    }

    virtual void forwardCPU(const NeuralLayer& layer_previous){
        std::copy(layer_previous.X().begin(),layer_previous.X().end(),_X_biais.begin());
        _Y = _W * _X_biais;
        for(unsigned int i=0;i<_Y.size();i++){
            _X(i) = Neuron::activation(_Y(i));
        }

    }
    virtual void backwardCPU(NeuralLayer& layer_previous){

        VecF32& d_E_X_previous= layer_previous.d_E_X();
        for(unsigned int i=0;i<_Y.size();i++){
            _d_E_Y(i) = _d_E_X(i)*Neuron::derivedActivation(_Y(i),_X(i));
        }
        for(unsigned int i=0;i<_W.sizeI();i++){
            for(unsigned int j=0;j<_W.sizeJ();j++){
                _d_E_W(i,j)=_X_biais(j)*_d_E_Y(i);
            }
        }
        for(unsigned int j=0;j<d_E_X_previous.size();j++){
            d_E_X_previous(j)=0;
            for(unsigned int i=0;i<_W.sizeI();i++){
                d_E_X_previous(j)+=_d_E_Y(i)*_W(i,j);
            }
        }
    }
    void learn(){
        for(unsigned int i=0;i<_W.sizeI();i++){
            for(unsigned int j=0;j<_W.sizeJ();j++){
                _W(i,j)= _W(i,j) -  _mu*_d_E_W(i,j);
            }
        }
    }


    VecF32& X(){return _X;}
    const VecF32& X()const{return _X;}
    VecF32& d_E_X(){return _d_E_X;}
    //private:

    VecF32 _Y;
    VecF32 _X;
    Mat2F32 _W;
    VecF32 _X_biais;

    VecF32 _d_E_Y;
    Mat2F32 _d_E_W;
    VecF32 _d_E_X;
};

struct NeuralNet
{
    Vec<NeuralLayer*> _v_layer;

    void addLayerFullyConnected(int nbr_neurons){
        if(_v_layer.size()==0){
            this->_v_layer.push_back(new NeuralLayerFullyConnected<>(0,nbr_neurons));
        }else{
            this->_v_layer.push_back(new NeuralLayerFullyConnected<>((*(_v_layer.rbegin()))-> X().size(),nbr_neurons));
        }
    }
    void setLearnableParameter(F32 mu){
        for(unsigned int i=0;i<_v_layer.size();i++){
            _v_layer(i)->setLearnableParameter(mu);
        }
    }
    void setTrainable(bool istrainable){
        for(unsigned int i=0;i<_v_layer.size();i++){
            _v_layer(i)->setTrainable(istrainable);
        }
    }
    void learn(){
        for(unsigned int i=0;i<_v_layer.size();i++){
            _v_layer(i)->learn();
        }
    }
    void forwardCPU(const VecF32& X_in, VecF32& X_out){
        std::copy(X_in.begin(),X_in.end(), (*(_v_layer.begin()))->X().begin());
        for(unsigned int i=1;i<_v_layer.size();i++){
            _v_layer(i)->forwardCPU(*_v_layer(i-1));
        }
        std::copy((*(_v_layer.rbegin()))->X().begin(),(*(_v_layer.rbegin()))->X().end(),X_out.begin());
    }

    void backwardCPU(VecF32& X_expected){

        //first output layer
        NeuralLayer* layer_last = _v_layer[_v_layer.size()-1];
        for(unsigned int j=0;j<X_expected.size();j++){
            layer_last->d_E_X()(j) = ( layer_last->X()(j)-X_expected(j));
        }

        for( int index_layer=_v_layer.size()-1;index_layer>0;index_layer--){
            NeuralLayer* layer = _v_layer[index_layer];
            NeuralLayer* layer_previous = _v_layer[index_layer-1];
            layer->backwardCPU(* layer_previous);
        }
    }
};

int main(){
    NeuralNetworkFeedForward n_ref;
    n_ref.addInputLayer(2);
    n_ref.addLayerFullyConnected(3);
    n_ref.addLayerFullyConnected(1);
    n_ref.setLearningRate(0.1);


    NeuralNet neural;
    neural.addLayerFullyConnected(2);
    neural.addLayerFullyConnected(3);
    neural.addLayerFullyConnected(1);
    neural.setTrainable(true);
    neural.setLearnableParameter(0.1);



    Vec<VecF32> v_in(4,VecF32(2)),v_out(4,VecF32(1));
    v_in(0)(0)=-1;v_in(0)(1)=-1;v_out(0)(0)= 1;
    v_in(1)(0)= 1;v_in(1)(1)=-1;v_out(1)(0)=-1;
    v_in(2)(0)=-1;v_in(2)(1)= 1;v_out(2)(0)=-1;
    v_in(3)(0)= 1;v_in(3)(1)= 1;v_out(3)(0)= 1;

    VecF32 v_out_net(1);


    for(unsigned int i=1;i<=2;i++){
        NNLayer* layer_neural = n_ref.layers()(i);
        if(NeuralLayerFullyConnected<>* layer_new = dynamic_cast<NeuralLayerFullyConnected<> *>(neural._v_layer(i))){
            //        std::cout<<layer_neural->_weights.size()<<std::endl;
            //        std::cout<<test._v_layer(i)._W.size()<<std::endl;
            //fully connected
            for(unsigned int i=0;i<layer_new->_W.sizeI();i++){
                for(unsigned int j=0;j<layer_new->_W.sizeJ();j++){
                    if(j<layer_new->_W.sizeJ()-1){
                        layer_new->_W(i,j)=layer_neural->_weights(j+i*layer_new->_W.sizeJ()+1)->_Wn;
                    }else{
                        layer_new->_W(i,j)=layer_neural->_weights(i*layer_new->_W.sizeJ())->_Wn;
                    }
                }
            }
        }


        //        for(unsigned int index_weight=0;index_weight<layer_neural->_weights.size();index_weight++){
        //            if(index_weight==0){
        //                layer_new->_W(layer_neural->_weights.size()-1)=layer_neural->_weights(index_weight)->_Wn;
        //            }else{
        //                 layer_new->_W(index_weight-1)=layer_neural->_weights(index_weight)->_Wn;
        //            }
        //        }

    }


    Vec<int> v_global_rand(4);
    for(unsigned int i=0;i<v_global_rand.size();i++)
        v_global_rand(i)=i;

    for(unsigned int i=0;i<100;i++){
        std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,Distribution::irand());
        for(unsigned int j=0;j<v_global_rand.size();j++){

            n_ref.propagateFront(v_in(v_global_rand(j)),v_out_net);
            std::cout<<v_out_net<<std::endl;
            neural.forwardCPU(v_in(v_global_rand(j)),v_out_net);
            std::cout<<v_out_net<<std::endl;
            //            exit(0);
            neural.backwardCPU(v_out(v_global_rand(j)));
            n_ref.propagateBackFirstDerivate(v_out(v_global_rand(j)));
            neural.learn();
            n_ref.learningFirstDerivate();
            neural.forwardCPU(v_in(v_global_rand(j)),v_out_net);
            std::cout<<"neural1 "<<v_out_net<<std::endl;
            n_ref.propagateFront(v_in(v_global_rand(j)),v_out_net);
            std::cout<<"neural2 "<<v_out_net<<std::endl;
            std::cout<<"expected "<<v_out(v_global_rand(j))<<std::endl;
        }
    }
    return 0;
    //    {
    //        Mat2UI8 m;
    //        m.load("plate1.pgm");
    //        //        m.display();
    //        findLine(m);
    //       return 0;

    //    }
    {

        Mat2F32 m(2,2);
        m(0,0)=1;m(0,1)=1;
        m(1,0)=3;m(1,1)=5;
        Mat2F32 m_lign(2,1);
        m_lign(0,0)=3;
        m_lign(1,0)=2;
        VecF32 & v = m_lign;
        std::cout<<m*v<<std::endl;
        return 1;


        //        Vec<LinearLeastSquareRANSACModel::Data> data;
        //        VecF32 x(2);F32 y;
        //        x(0)=1;x(1)=1;y=5.2;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        //        x(0)=1;x(1)=2;y=5.8;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        //        x(0)=1;x(1)=3;y=6.8;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        //        x(0)=1;x(1)=4;y=8.2;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        //        x(0)=1;x(1)=5;y=13; data.push_back(LinearLeastSquareRANSACModel::Data(x,y));

        //        LinearLeastSquareRANSACModel m;
        //        Vec<LinearLeastSquareRANSACModel::Data> dataconsencus;
        //        ransac(data,10,1,2,m,dataconsencus);
        //        std::cout<<m.getBeta()<<std::endl;
        //        std::cout<<m.getError()<<std::endl;
        //        std::cout<<dataconsencus<<std::endl;

        //                Vec<LinearLeastSquareRANSACModel::Data> data;
        //                VecF32 x(2);F32 y;
        //                x(0)=1;x(1)=1;y=5.2;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        //                x(0)=1;x(1)=2;y=5.8;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        //                x(0)=1;x(1)=3;y=6.8;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        //                x(0)=1;x(1)=4;y=8.2;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        //                x(0)=1;x(1)=5;y=13; data.push_back(LinearLeastSquareRANSACModel::Data(x,y));

        //                LinearLeastSquareRANSACModel m;
        //                Vec<LinearLeastSquareRANSACModel::Data> dataconsencus;
        //                ransac(data,10,1,3,m,dataconsencus);
        //                std::cout<<m.getBeta()<<std::endl;
        //                std::cout<<m.getError()<<std::endl;
        //                std::cout<<dataconsencus<<std::endl;
        //        return 1;
    }
    //    {
    //        Mat2UI8 m;
    //        m.load(POP_PROJECT_SOURCE_DIR+std::string("/image/barriere.png"));
    //        Mat2UI8 edge = Processing::edgeDetectorCanny(m,2,0.5,5);
    //        edge.display("edge",false);
    //        Mat2F32 hough = Feature::transformHough(edge);
    //        hough.display("hough",false);
    //        std::vector< std::pair<Vec2F32, Vec2F32 > > v_lines = Feature::HoughToLines(hough,edge ,0.5);
    //        Mat2RGBUI8 m_hough(m);
    //        for(unsigned int i=0;i<v_lines.size();i++){
    //            Draw::line(m_hough,v_lines[i].first,v_lines[i].second,  RGBUI8(255,0,0),2);
    //        }
    //        m_hough.display();
    //    }
    Mat2UI8 m_init;
    std::string dir="/home/vincent/Desktop/passeport/";
    std::vector<std::string> v_files =BasicUtility::getFilesInDirectory(dir);

    for(unsigned int i=3;i<v_files.size();i++){
        std::cout<<i<<std::endl;
        m_init.load( dir+v_files[i]);
        F32 scale_init =1200./m_init.sizeJ();
        m_init = GeometricalTransformation::scale(m_init,Vec2F32(scale_init,scale_init));
        //    m_init.load("/home/vincent/Desktop/coutin.jpg");
        F32 scalefactor =400./m_init.sizeJ();
        pop::Vec2F32 border_extra=pop::Vec2F32(0.01f,0.1f);
        Mat2UI8 m = GeometricalTransformation::scale(m_init,Vec2F32(scalefactor,scalefactor));
        std::cout<<m.getDomain()<<std::endl;

        Mat2UI8 tophat_init =   Processing::closing(m,2)-m;
        tophat_init.display("top hat",false);
        Mat2UI8 elt(3,3);
        elt(1,0)=1;
        elt(1,1)=1;
        elt(1,2)=1;

        Mat2UI8 tophat = Processing::closingStructuralElement(tophat_init,elt  ,8);
        tophat = Processing::openingStructuralElement(tophat,elt  ,2  );
        elt = elt.transpose();
        tophat = Processing::openingStructuralElement(tophat,elt  ,2 );
        tophat.display("filter",false);

        int value;
        Mat2UI8 binary = Processing::thresholdOtsuMethod(tophat,value);
        Mat2UI8 binary2 =  Processing::threshold(tophat,20);
        binary = pop::minimum(binary,binary2);
        binary.display("binary",false);



        Mat2UI32 imglabel =ProcessingAdvanced::clusterToLabel(binary, binary.getIteratorENeighborhood(1,0),binary.getIteratorEOrder(1));
        Visualization::labelToRandomRGB(imglabel).display("label",false);
        pop::Vec<pop::Vec2I32> v_xmin;
        pop::Vec<pop::Vec2I32> v_xmax;
        pop::Vec<Mat2UI8> v_img = Analysis::labelToMatrices(imglabel,v_xmin,v_xmax);

        OCRNeuralNetwork ocr;
        ocr.setDictionnary("/home/vincent/DEV2/DEV/LAPI-API/neuralnetwork.xml");

        for(unsigned int index_label =0;index_label<v_img.size();index_label++){
            Vec2I32 domain = v_img[index_label].getDomain();
            if(domain(1)>0.5*binary.sizeJ()){

                Vec2I32 xmin =  v_xmin[index_label];
                xmin(1)+=10;
                Vec2I32 xmax =  v_xmax[index_label];
                xmax(1)-=10;
                int sum_0=0;
                int sum_1=0;
                int bary_i_0=0;
                int bary_i_1=0;
                int index_j_0 = xmin(1)+10;
                int index_j_1 = xmax(1)-10;
                unsigned int index_i_0_min=10000;
                unsigned int index_i_0_max=0;
                unsigned int index_i_1_min=10000;
                unsigned int index_i_1_max=0;
                for(unsigned int i=xmin(0);i<xmax(0);i++){
                    if(binary(i,index_j_0)>0){
                        index_i_0_min=std::min(i,index_i_0_min);
                        index_i_0_max=std::max(i,index_i_0_max);
                        sum_0++;
                        bary_i_0+=i;
                    }
                    if(binary(i,index_j_1)>0){
                        index_i_1_min=std::min(i,index_i_1_min);
                        index_i_1_max=std::max(i,index_i_1_max);
                        sum_1++;
                        bary_i_1+=i;
                    }
                }
                bary_i_0/=sum_0;
                bary_i_1/=sum_1;
                double delta_y = index_j_1-index_j_0;
                double delta_x = bary_i_1-bary_i_0;
                double rot = atan2(delta_x,delta_y);
                xmin =  v_xmin[index_label];
                xmin(1)=std::max(0,xmin(1)-10);
                xmin(0)=std::max(0,xmin(0)-3);
                xmax =  v_xmax[index_label];
                xmax(1)=std::min(tophat_init.getDomain()(1)-1,xmax(1)+25);
                xmax(0)=std::min(tophat_init.getDomain()(0)-1,xmax(0)+3);
                std::cout<<rot<<std::endl;
                Mat2UI8 m_lign = GeometricalTransformation::rotate(m_init(Vec2F32(xmin)/scalefactor,Vec2F32(xmax)/scalefactor),-rot,MATN_BOUNDARY_CONDITION_MIRROR);
                //GeometricalTransformation::scale(,Vec2F32(3,3)).display("rot",true,false);

                xmin(0)=std::abs(delta_x/2)*1./scalefactor;

                xmin(1)=0;
                xmax(0)=m_lign.getDomain()(0)-1-std::abs(delta_x/2)*1./scalefactor;
                xmax(1)=m_lign.getDomain()(1)-1;
                std::cout<<xmin<<std::endl;
                std::cout<<xmax<<std::endl;
                std::cout<<m_lign.getDomain()<<std::endl;

                //                F32 ratio = m_init.sizeJ()/1000;
                //                std::cout<<ratio<<std::endl;
                m_lign = m_lign(xmin,xmax);
                m_lign = Processing::median(m_lign,2);
                Mat2UI8 tophat_letter = Processing::closing(m_lign,7)-m_lign;

                int value = 0;
                m_lign.display("init",false,false);
                Processing::thresholdOtsuMethod(tophat_letter,value);
                tophat_letter = Processing::threshold(tophat_letter,value-5);
                tophat_letter.display("rot",false,false);

                GeometricalTransformation::scale(tophat_letter,Vec2F32(3,3)).display("rot",false,false);

                MatN<1,UI16> m1(VecN<1,int>(tophat_letter.sizeJ())),m2(VecN<1,int>(tophat_letter.sizeJ()));
                Mat2UI16 m(tophat_letter.sizeJ(),2);
                int maxi=0,mini=NumericLimits<int>::maximumRange();
                for(unsigned int j=0;j<tophat_letter.sizeJ();j++){
                    int sum=0;
                    for(unsigned int i=0;i<tophat_letter.sizeI();i++){
                        sum+=tophat_letter(i,j);
                    }
                    m(j,0)=j;
                    m(j,1)=sum;
                    m1(j)=sum;
                    maxi=std::max(maxi,sum);
                    mini=std::min(mini,sum);
                }
                for(unsigned int i=0;i<m1.getDomain()(0);i++){
                    m1(i)= (m1(i)-mini)*100/(maxi-mini);

                }

                m2= Processing::smoothGaussian(m1,2);
                m2= Processing::dynamic(m2,5);
                m2 = Processing::minimaRegional(m2,1);


                MatN<1,UI8> thhinning = Analysis::thinningAtConstantTopology(MatN<1,UI8>(m2));
                int min_previous=-1;
                for(unsigned int j=0;j<m2.getDomain()(0);j++){
                    if(thhinning(j)>0&&m1(j)<10){
                        if(min_previous==-1)
                            min_previous =j;
                        else{
                            int value;

                            Mat2UI8 m_letter = Processing::clusterMax(Processing::thresholdOtsuMethod(tophat_letter(Vec2I32(0,min_previous),Vec2I32(tophat_letter.getDomain()(0)-1,j)),value));//.display();
                            ;
                            std::cout<<ocr.parseMatrix(m_letter)<<" ";
                            min_previous =j;
                        }
                        for(unsigned int i=0;i<m_lign.sizeI();i++){
                            m_lign(i,j)=255;
                        }

                    }
                }
                std::cout<<std::endl;
                GeometricalTransformation::scale(m_lign,Vec2F32(3,3)).display("rot",true,false);

                //           m.display();
                //            return 0;




            }
        }

    }

    return 0;


    //    if(display)temp =Draw::mergeTwoMatrixVertical(temp,tophat);
    //    Mat2UI8 elt(3,3);
    //    elt(1,0)=1;
    //    elt(1,1)=1;
    //    elt(1,2)=1;

    //    tophat = Processing::closingStructuralElement(tophat,elt  ,std::ceil(pixel_width_letter));
    //    if(display)temp =Draw::mergeTwoMatrixVertical(temp,tophat);
    //    tophat = Processing::openingStructuralElement(tophat,elt  ,std::ceil(pixel_width_letter)/4  );
    //    elt = elt.transpose();
    //    tophat = Processing::openingStructuralElement(tophat,elt  ,std::ceil(pixel_width_letter)/factor_opening_vertical );
    //    if(display)temp =Draw::mergeTwoMatrixVertical(temp,tophat);
    //    int value;
    //    Mat2UI8 binary = Processing::thresholdOtsuMethod(tophat,value);
    //    Mat2UI8 binary2 =  Processing::threshold(tophat,10);



    neuralnetwortest2();
    return 1;
}
