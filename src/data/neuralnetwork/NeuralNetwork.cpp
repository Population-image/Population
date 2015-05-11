#include "data/neuralnetwork/NeuralNetwork.h"
#include "data/distribution/DistributionAnalytic.h"
#include "data/mat/MatN.h"
#include "data/mat/MatNInOut.h"
#include "data/mat/MatNDisplay.h"
#include "PopulationConfig.h"
namespace pop {
#define SIGMOID(x) (1.7159f*tanh(0.6666667f*x))
#define DSIGMOID(S) (0.6666667f/1.7159f*(1.7159f+(S))*(1.7159f-(S)))  // derivative of the sigmoid as a function of the sigmoid's output
///////////////////////////////////////////////////////////////////////
//
//  NeuralNetwork class definition

NeuralNetworkFeedForward::NeuralNetworkFeedForward()
    :_is_input_layer(false)
{
}
NeuralNetworkFeedForward::NeuralNetworkFeedForward(const NeuralNetworkFeedForward& n){
    *this = n;
}
NeuralNetworkFeedForward & NeuralNetworkFeedForward::operator=(const NeuralNetworkFeedForward& n){
    this->_label2string = n._label2string;
    for(unsigned int i=0;i<n._layers.size();i++){
        NNLayer * layer = n._layers[i];
        if(i==0)
        {
            if(layer->_type==NNLayer::INPUTMATRIX){
                NNLayerMatrix* inputlayer = dynamic_cast<NNLayerMatrix*>(layer);
                this->addInputLayerMatrix(inputlayer->_neurons_matrix[0].getDomain()(0),inputlayer->_neurons_matrix[0].getDomain()(1),inputlayer->_method,inputlayer->_normalization_value);
            }else{
                this->addInputLayer(static_cast<unsigned int>(layer->_neurons.size()));
            }
        }else{
            if(layer->_type==NNLayer::MATRIXCONVOLUTIONNAL){
                NNLayerMatrixConvolutionalPlusSubScaling* inputlayer = dynamic_cast<NNLayerMatrixConvolutionalPlusSubScaling*>(layer);

                this->addLayerConvolutionalPlusSubScaling(inputlayer->_neurons_matrix.getDomain(),inputlayer->_size_convolutution,inputlayer->_sub_scaling,-1);
                NNLayer* my_layer = *(this->_layers.rbegin());
                for(unsigned int index_w=0;index_w<inputlayer->_weights.size();index_w++){
                    NNWeight * weight = layer->_weights[index_w];
                    my_layer->_weights[index_w]->_Wn = weight->_Wn;
                }
            }else if(layer->_type==NNLayer::MATRIXCONVOLUTIONNAL2){
                NNLayerMatrixConvolutional* inputlayer = dynamic_cast<NNLayerMatrixConvolutional*>(layer);

                this->addLayerConvolutional(inputlayer->_neurons_matrix.getDomain(),inputlayer->_size_convolutution,-1);
                NNLayer* my_layer = *(this->_layers.rbegin());
                for(unsigned int index_w=0;index_w<inputlayer->_weights.size();index_w++){
                    NNWeight * weight = layer->_weights[index_w];
                    my_layer->_weights[index_w]->_Wn = weight->_Wn;
                }
            }else if(layer->_type==NNLayer::MATRIXSUBSALE){
                NNLayerMatrixSubScale* inputlayer = dynamic_cast<NNLayerMatrixSubScale*>(layer);

                this->addLayerSubScaling(inputlayer->_sub_scaling,-1);
                NNLayer* my_layer = *(this->_layers.rbegin());
                for(unsigned int index_w=0;index_w<inputlayer->_weights.size();index_w++){
                    NNWeight * weight = layer->_weights[index_w];
                    my_layer->_weights[index_w]->_Wn = weight->_Wn;
                }

            }else if(layer->_type==NNLayer::MATRIXMAXPOOLING){
                NNLayerMatrixMaxPooling* inputlayer = dynamic_cast<NNLayerMatrixMaxPooling*>(layer);
                this->addLayerMaxPooling(inputlayer->_sub_scaling);
            }
            else if(layer->_type==NNLayer::FULLYCONNECTED)
            {
                this->addLayerFullyConnected(static_cast<unsigned int>(layer->_neurons.size()),-1);
                NNLayer* my_layer = *(this->_layers.rbegin());
                for(unsigned int index_w=0;index_w<layer->_weights.size();index_w++){
                    NNWeight * weight = layer->_weights[index_w];
                    my_layer->_weights[index_w]->_Wn = weight->_Wn;
                }
            }
        }
    }
    return *this;
}


void NeuralNetworkFeedForward::propagateFront(const pop::VecF32& in , pop::VecF32 &out)
{
    // first layer is imput layer: directly set outputs of all of its neurons
    // to the input vector
    NNLayer & layerfirst = *(_layers[0]);
    POP_DbgAssert( in.size() == layerfirst._neurons.size() );
#if defined(HAVE_OPENMP)
#pragma omp parallel for
#endif
    for(int i_neuron=0;i_neuron<static_cast<int>(layerfirst._neurons.size());i_neuron++){
        NNNeuron *  neuron = (layerfirst._neurons[i_neuron]);
        neuron->_Xn =  in(i_neuron);
    }

    // propagate layer by layer
    for(unsigned int i_layer=1;i_layer<_layers.size();i_layer++){
        NNLayer & layer = *(_layers[i_layer]);
        layer.propagateFront();
    }

    // load up output vector with results
    NNLayer & layerout = *(_layers[_layers.size()-1]);
    out.resize(layerout._neurons.size());
#if defined(HAVE_OPENMP)
#pragma omp parallel for
#endif
    for( int i_neuron=0;i_neuron<static_cast<int>(layerout._neurons.size());i_neuron++){
        NNNeuron *  neuron = (layerout._neurons[i_neuron]);
        out(i_neuron) = neuron->_Xn;
    }
}
void NeuralNetworkFeedForward::propagateBackFirstDerivate(const pop::VecF32& desired_output)
{
    NNLayer & layerlast = *(_layers[_layers.size()-1]);
    POP_DbgAssert( desired_output.size() == layerlast._neurons.size() );
#if defined(HAVE_OPENMP)
#pragma omp parallel for
#endif
    for( int i_neuron=0;i_neuron<static_cast<int>(layerlast._neurons.size());i_neuron++){
        NNNeuron *  neuron = (layerlast._neurons[i_neuron]);
        neuron->_dErr_dXn = neuron->_Xn -  desired_output(i_neuron);
    }
    for(int i_layer=static_cast<int>(_layers.size())-1;i_layer>0;i_layer--){
        NNLayer * layer = (_layers[i_layer]);
        layer->propagateBackFirstDerivate();
    }
}
void NeuralNetworkFeedForward::propagateBackSecondDerivate()
{
    NNLayer & layerlast = *(_layers[_layers.size()-1]);
#if defined(HAVE_OPENMP)
#pragma omp parallel for
#endif
    for( int i_neuron=0;i_neuron<static_cast<int>(layerlast._neurons.size());i_neuron++){
        NNNeuron *  neuron = (layerlast._neurons[i_neuron]);
        neuron->_d2Err_dXn2 = 1;
    }

    for(unsigned int i_layer=static_cast<unsigned int>(_layers.size()-1);i_layer>0;i_layer--){
        NNLayer & layer = *(_layers[i_layer]);
        layer.propagateBackSecondDerivate();
    }
}

void NeuralNetworkFeedForward::learningFirstDerivate()
{
    for(unsigned int i_layer=static_cast<unsigned int>(_layers.size()-1);i_layer>0;i_layer--){
        NNLayer & layer = *(_layers[i_layer]);
        layer.learningFirstDerivate();
    }
}

void NeuralNetworkFeedForward::addInputLayer(unsigned int nbr_neuron){
    POP_DbgAssertMessage(_is_input_layer==false,"Add an input neuron before to add weighed layer");
    _is_input_layer = true;
    _layers.push_back( new NNLayer(nbr_neuron) );
    NNLayer & layer = *(_layers[_layers.size()-1]);
    layer._type = NNLayer::INPUT;
}

void NeuralNetworkFeedForward::addInputLayerMatrix(unsigned int height,unsigned int width,NNLayerMatrix::CenteringMethod method,NNLayerMatrix::NormalizationValue normalization){
    POP_DbgAssertMessage(_is_input_layer==false,"Add an input neuron before to add weighed layer");
    _is_input_layer = true;
    _layers.push_back( new NNLayerMatrix(1,height,width,method,normalization) );
    NNLayer & layer = *(_layers[_layers.size()-1]);
    layer._type = NNLayer::INPUTMATRIX;
}

void NeuralNetworkFeedForward::addLayerFullyConnected(unsigned int nbr_neuron,F32 standart_deviation_weight){

    NNLayer & layerprevious = *(_layers[_layers.size()-1]);
    _layers.push_back( new NNLayer(nbr_neuron) );
    NNLayer & layer = *(_layers[_layers.size()-1]);
    layer._type = NNLayer::FULLYCONNECTED;
    // This layer is a fully-connected layer
    // with nbr_neuron units.  Since it is fully-connected,
    // each of the nbr_neuron neurons in the layer
    // is connected to all nbr_neuron_previous_layer in
    // the previous layer.
    // So, there are nbr_neuron neurons and nbr_neuron*(nbr_neuron_previous_layer+1) weights
    //    pop::DistributionUniformInt d(0,1);


    //normalize tbe number inverse square root of the connection feeding into the nodes)
    pop::DistributionNormal d(0,standart_deviation_weight/std::sqrt(layerprevious._neurons.size()*1.f));
    unsigned int nbr_neuron_previous_layer = static_cast<unsigned int>(layerprevious._neurons.size());
    unsigned int nbr_weight = nbr_neuron*(nbr_neuron_previous_layer+1);
    for (unsigned int  i_weight=0; i_weight<nbr_weight; ++i_weight )
    {
        F32 initweight =0;
        if(standart_deviation_weight>0)
            initweight =  d.randomVariable();
        layer._weights.push_back(new NNWeight(initweight ) );
    }

    // Interconnections with previous layer: fully-connected

    unsigned int i_weight = 0;  // weights are not shared in this layer
    for (unsigned int  i_neuron=0; i_neuron<nbr_neuron; ++i_neuron )
    {
        NNNeuron * n = (layer._neurons[ i_neuron ]) ;
        n->addConnectionBiais(layer._weights[i_weight] );  // bias weight
        i_weight++;
        for (unsigned int  i_neuron_previous_layer=0; i_neuron_previous_layer<nbr_neuron_previous_layer; ++i_neuron_previous_layer )
        {
            NNNeuron * nprevious = layerprevious._neurons[ i_neuron_previous_layer ] ;
            n->addConnection(layer._weights[i_weight], nprevious );
            i_weight++;
        }
    }


}
void NeuralNetworkFeedForward::addLayerConvolutional( unsigned int nbr_map, unsigned int kernelsize,F32 standart_deviation_weight){
    if(NNLayerMatrix * layerprevious =dynamic_cast<NNLayerMatrix *>(_layers[_layers.size()-1])){
        unsigned int height_previous = layerprevious->_neurons_matrix(0).getDomain()(0);
        unsigned int width_previous  = layerprevious->_neurons_matrix(0).getDomain()(1);


        unsigned int height = (height_previous-(kernelsize-1));
        unsigned int width  = (width_previous-(kernelsize-1));
        NNLayerMatrixConvolutional * layer = new NNLayerMatrixConvolutional(nbr_map,height,width);
        layer->_size_convolutution = kernelsize;

        layer->_type = NNLayer::MATRIXCONVOLUTIONNAL2;
        _layers.push_back(layer);

        // This layer is a convolutional connected layer
        // with nbr_neuron units.
        // (kernelsize*kernelsize+1)* nbr_map_previous_layer*nbr_map_layer
        unsigned int nbr_map_previous = static_cast<unsigned int>(layerprevious->_neurons_matrix.size());


        //normalize tbe number inverse square root of the connection feeding into the nodes)
        pop::DistributionNormal d(0,standart_deviation_weight/kernelsize);

        unsigned int nbr_weight = nbr_map*nbr_map_previous*(kernelsize*kernelsize+1);
        for (unsigned int  i_weight=0; i_weight<nbr_weight; ++i_weight )
        {
            F32 initweight =0;
            if(standart_deviation_weight>0)
                initweight =  d.randomVariable();
            layer->_weights.push_back(new NNWeight(initweight ) );
        }



        for(unsigned int i_width=0;i_width<width;i_width++)
        {
            for(unsigned int i_height=0;i_height<height;i_height++)
            {
                for(unsigned int i_pattern_previous = 0;i_pattern_previous<nbr_map_previous;i_pattern_previous++)
                {
                    for(unsigned int i_pattern = 0;i_pattern<nbr_map;i_pattern++)
                    {

                        NNNeuron * n = layer->_neurons_matrix(i_pattern)(i_height,i_width) ;
                        unsigned int i_weight_begin = i_pattern*(kernelsize*kernelsize+1)+i_pattern_previous*nbr_map*(kernelsize*kernelsize+1);
                        n->addConnectionBiais(layer->_weights[i_weight_begin]);
                        for(unsigned int i_heigh_kernel=0;i_heigh_kernel<kernelsize;i_heigh_kernel++)
                        {
                            for(unsigned int i_width_kernel=0;i_width_kernel<kernelsize;i_width_kernel++)
                            {

                                int i_width_previous =  i_width + i_width_kernel ;
                                int i_height_previous =  i_height+ i_heigh_kernel;

                                NNNeuron * nprevious = layerprevious->_neurons_matrix(i_pattern_previous)(i_height_previous,i_width_previous) ;
                                unsigned int i_weight = 1+i_width_kernel + i_heigh_kernel*kernelsize + i_weight_begin;
                                n->addConnection(layer->_weights[i_weight], nprevious );
                            }

                        }
                    }

                }

            }
        }
    }
}
void NeuralNetworkFeedForward::addLayerIntegral(unsigned int nbr_integral,F32 standart_deviation_weight){
    if(NNLayerMatrix * layerprevious =dynamic_cast<NNLayerMatrix *>(_layers[_layers.size()-1])){
        unsigned int height_previous = layerprevious->_neurons_matrix(0).getDomain()(0);
        unsigned int width_previous  = layerprevious->_neurons_matrix(0).getDomain()(1);
        unsigned int nbr_map_previous = static_cast<unsigned int>(layerprevious->_neurons_matrix.size());



        NNLayer * layer = new NNLayer(nbr_integral*nbr_map_previous);



        _layers.push_back(layer);

        //normalize tbe number inverse square root of the connection feeding into the nodes)
        pop::DistributionNormal d(0,standart_deviation_weight/std::sqrt(height_previous*width_previous*1.f));

        unsigned int nbr_weight = nbr_integral*(height_previous*width_previous+1);
        for (unsigned int  i_weight=0; i_weight<nbr_weight; ++i_weight )
        {
            F32 initweight =0;
            if(standart_deviation_weight>0)
                initweight =  d.randomVariable();
            layer->_weights.push_back(new NNWeight(initweight ) );
        }

        for(unsigned int i_integral = 0;i_integral<nbr_integral;i_integral++)
        {
            for(unsigned int i_pattern_previous = 0;i_pattern_previous<nbr_map_previous;i_pattern_previous++)
            {
                unsigned int i_weight_begin = i_integral*(height_previous*width_previous+1);
                NNNeuron * n = (layer->_neurons[i_pattern_previous+i_integral*nbr_map_previous ]) ;
                n->addConnectionBiais(layer->_weights[i_weight_begin]);
                for(unsigned int i_height=0;i_height<height_previous;i_height++)
                {
                    for(unsigned int i_width=0;i_width<width_previous;i_width++)
                    {
                        NNNeuron * nprevious = (layerprevious->_neurons_matrix(i_pattern_previous)(i_height,i_width)) ;

                        unsigned int i_weight = 1+i_width + i_height*width_previous + i_weight_begin;
                        n->addConnection(layer->_weights[i_weight], nprevious );
                    }

                }
            }

        }

    }
}

void NeuralNetworkFeedForward::addLayerSubScaling(unsigned int sub_scale_factor,F32 standart_deviation_weight){
    if(NNLayerMatrix * layerprevious =dynamic_cast<NNLayerMatrix *>(_layers[_layers.size()-1])){
        unsigned int height_previous = layerprevious->_neurons_matrix(0).getDomain()(0);
        unsigned int width_previous  = layerprevious->_neurons_matrix(0).getDomain()(1);
        unsigned int nbr_map_previous = static_cast<unsigned int>(layerprevious->_neurons_matrix.size());

        unsigned int height = (height_previous/sub_scale_factor);
        unsigned int width  = (width_previous/sub_scale_factor);

        NNLayerMatrixSubScale * layer = new NNLayerMatrixSubScale(nbr_map_previous,height,width);
        layer->_sub_scaling = sub_scale_factor;

        layer->_type = NNLayer::MATRIXSUBSALE;
        _layers.push_back(layer);




        //normalize tbe number inverse square root of the connection feeding into the nodes)
        pop::DistributionNormal d(0,standart_deviation_weight/sub_scale_factor);

        unsigned int nbr_weight = nbr_map_previous*height*width*(sub_scale_factor*sub_scale_factor+1);
        for (unsigned int  i_weight=0; i_weight<nbr_weight; ++i_weight )
        {
            F32 initweight =0;
            if(standart_deviation_weight>0)
                initweight =  d.randomVariable();
            layer->_weights.push_back(new NNWeight(initweight ) );
        }




        for(unsigned int i_pattern_previous = 0;i_pattern_previous<nbr_map_previous;i_pattern_previous++)
        {
            for(unsigned int i_height=0;i_height<height;i_height++)
            {
                for(unsigned int i_width=0;i_width<width;i_width++)
                {

                    NNNeuron * n = (layer->_neurons_matrix(i_pattern_previous)(i_height,i_width)) ;
                    unsigned int i_weight_begin =i_pattern_previous*width*height*(sub_scale_factor*sub_scale_factor+1) +  (i_width + i_height*width)*(sub_scale_factor*sub_scale_factor+1) ;
                    n->addConnectionBiais(layer->_weights[i_weight_begin]);
                    for(unsigned int i_heigh_kernel=0;i_heigh_kernel<sub_scale_factor;i_heigh_kernel++)
                    {
                        for(unsigned int i_width_kernel=0;i_width_kernel<sub_scale_factor;i_width_kernel++)
                        {
                            int i_width_previous =  i_width*sub_scale_factor + i_width_kernel ;
                            int i_height_previous = i_height*sub_scale_factor+ i_heigh_kernel;

                            NNNeuron * nprevious = (layerprevious->_neurons_matrix(i_pattern_previous)(i_height_previous,i_width_previous)) ;
                            unsigned int i_weight =i_weight_begin+     1+i_width_kernel + i_heigh_kernel*sub_scale_factor ;
                            n->addConnection(layer->_weights[i_weight], nprevious );
                        }

                    }
                }

            }

        }
    }
}
void NeuralNetworkFeedForward::addLayerMaxPooling(unsigned int sub_scale_factor){
    if(NNLayerMatrix * layerprevious =dynamic_cast<NNLayerMatrix *>(_layers[_layers.size()-1])){
        unsigned int height_previous = layerprevious->_neurons_matrix(0).getDomain()(0);
        unsigned int width_previous  = layerprevious->_neurons_matrix(0).getDomain()(1);
        unsigned int nbr_map_previous = static_cast<unsigned int>(layerprevious->_neurons_matrix.size());

        unsigned int height = (height_previous/sub_scale_factor);
        unsigned int width  = (width_previous/sub_scale_factor);

        NNLayerMatrixMaxPooling * layer = new NNLayerMatrixMaxPooling(nbr_map_previous,height,width);
        layer->_sub_scaling = sub_scale_factor;

        layer->_type = NNLayer::MATRIXMAXPOOLING;
        _layers.push_back(layer);




        for(unsigned int i_pattern_previous = 0;i_pattern_previous<nbr_map_previous;i_pattern_previous++)
        {
            for(unsigned int i_height=0;i_height<height;i_height++)
            {
                for(unsigned int i_width=0;i_width<width;i_width++)
                {

                    NNNeuronMaxPool * n = dynamic_cast<NNNeuronMaxPool*>((layer->_neurons_matrix(i_pattern_previous)(i_height,i_width))) ;

                    for(unsigned int i_heigh_kernel=0;i_heigh_kernel<sub_scale_factor;i_heigh_kernel++)
                    {
                        for(unsigned int i_width_kernel=0;i_width_kernel<sub_scale_factor;i_width_kernel++)
                        {
                            int i_width_previous =  i_width*sub_scale_factor + i_width_kernel ;
                            int i_height_previous = i_height*sub_scale_factor+ i_heigh_kernel;
                            NNNeuron * nprevious = (layerprevious->_neurons_matrix(i_pattern_previous)(i_height_previous,i_width_previous)) ;
                            n->addConnection( nprevious );
                        }

                    }
                }

            }

        }
    }
}

void NeuralNetworkFeedForward::addLayerConvolutionalPlusSubScaling( unsigned int nbr_map, unsigned int kernelsize,unsigned int sub_scale_sampling,F32 standart_deviation_weight){
    if(NNLayerMatrix * layerprevious =dynamic_cast<NNLayerMatrix *>(_layers[_layers.size()-1])){


        unsigned int step_previous = (kernelsize-1)/2;
        unsigned int height_previous = layerprevious->_neurons_matrix(0).getDomain()(0);
        unsigned int width_previous  = layerprevious->_neurons_matrix(0).getDomain()(1);

        if ((height_previous-(kernelsize-sub_scale_sampling)) % sub_scale_sampling!=0) {
            std::cerr<<"The heigh of the input matrix must be pair "<<std::endl;
        }

        unsigned int height = (height_previous-(kernelsize-sub_scale_sampling))/sub_scale_sampling;
        unsigned int width  = (width_previous-(kernelsize-sub_scale_sampling))/sub_scale_sampling;

        NNLayerMatrixConvolutionalPlusSubScaling * layer = new NNLayerMatrixConvolutionalPlusSubScaling(nbr_map,height,width);
        layer->_size_convolutution = kernelsize;
        layer->_sub_scaling = sub_scale_sampling;
        layer->_type = NNLayer::MATRIXCONVOLUTIONNAL;
        _layers.push_back(layer);

        // This layer is a convolutional connected layer
        // with nbr_neuron units.
        // (kernelsize*kernelsize+1)* nbr_map_previous_layer*nbr_map_layer
        unsigned int nbr_map_previous = static_cast<unsigned int>(layerprevious->_neurons_matrix.size());


        //normalize tbe number inverse square root of the connection feeding into the nodes)
        pop::DistributionNormal d(0,standart_deviation_weight/kernelsize);

        unsigned int nbr_weight = nbr_map*nbr_map_previous*(kernelsize*kernelsize+1);
        for (unsigned int  i_weight=0; i_weight<nbr_weight; ++i_weight )
        {
            F32 initweight =0;
            if(standart_deviation_weight>0)
                initweight =  d.randomVariable();
            layer->_weights.push_back(new NNWeight(initweight ) );
        }



        for(unsigned int i_width=0;i_width<width;i_width++)
        {
            for(unsigned int i_height=0;i_height<height;i_height++)
            {

                for(unsigned int i_pattern = 0;i_pattern<nbr_map;i_pattern++)
                {

                    NNNeuron * n = (layer->_neurons_matrix(i_pattern)(i_height,i_width)) ;
                    for(unsigned int i_pattern_previous = 0;i_pattern_previous<nbr_map_previous;i_pattern_previous++)
                    {
                        unsigned int i_weight = i_pattern*(kernelsize*kernelsize+1)+i_pattern_previous*nbr_map*(kernelsize*kernelsize+1);
                        n->addConnectionBiais(layer->_weights[i_weight]);
                        for(unsigned int i_width_kernel=0;i_width_kernel<kernelsize;i_width_kernel++)
                        {
                            for(unsigned int i_heigh_kernel=0;i_heigh_kernel<kernelsize;i_heigh_kernel++)
                            {

                                int i_width_previous =  i_width*2 + i_width_kernel ;
                                int i_height_previous =  i_height*2 + i_heigh_kernel ;

                                NNNeuron * nprevious = (layerprevious->_neurons_matrix(i_pattern_previous)(i_height_previous,i_width_previous)) ;
                                unsigned int i_weight_index = 1+i_width_kernel + i_heigh_kernel*kernelsize + i_pattern*(kernelsize*kernelsize+1)+i_pattern_previous*nbr_map*(kernelsize*kernelsize+1);
                                n->addConnection(layer->_weights[i_weight_index], nprevious );
                            }

                        }
                    }

                }

            }
        }
    }
}
NeuralNetworkFeedForward::~NeuralNetworkFeedForward()
{
    init();
}

void NeuralNetworkFeedForward::init(){
    for(unsigned int i=0;i<_layers.size();i++){
        delete _layers[i];
    }
    _layers.clear();
}
void NeuralNetworkFeedForward::setLearningRate(F32 eta)
{
    for(unsigned int i_layer=0;i_layer<_layers.size();i_layer++)
    {
        NNLayer & layer = *(_layers[i_layer]);
        layer.setLearningRate(eta);
    }
}

void NeuralNetworkFeedForward::save(const char * file)const
{
    XMLDocument doc;

    XMLNode node1 = doc.addChild("label2String");
    node1.addAttribute("id",BasicUtility::Any2String(_label2string));
    XMLNode node = doc.addChild("layers");
    for(unsigned int i=0;i<this->_layers.size();i++){
        NNLayer * layer = this->_layers[i];
        if(i==0)
        {
            if(layer->_type==NNLayer::INPUTMATRIX){
                NNLayerMatrix* inputlayer = dynamic_cast<NNLayerMatrix*>(layer);
                XMLNode nodechild = node.addChild("layer");
                nodechild.addAttribute("type","NNLayer::INPUTMATRIX");
                nodechild.addAttribute("size",BasicUtility::Any2String(inputlayer->_neurons_matrix[0].getDomain()));
                nodechild.addAttribute("method",BasicUtility::Any2String((unsigned int)inputlayer->_method));
                nodechild.addAttribute("normalization",BasicUtility::Any2String((unsigned int)inputlayer->_normalization_value));
            }else{

                XMLNode nodechild = node.addChild("layer");
                nodechild.addAttribute("type","NNLayer::INPUT");
                nodechild.addAttribute("size",BasicUtility::Any2String(layer->_neurons.size()));
            }
        }else{
            if(layer->_type==NNLayer::MATRIXCONVOLUTIONNAL){
                NNLayerMatrixConvolutionalPlusSubScaling* inputlayer = dynamic_cast<NNLayerMatrixConvolutionalPlusSubScaling*>(layer);
                XMLNode nodechild = node.addChild("layer");
                nodechild.addAttribute("type","NNLayer::MATRIXCONVOLUTIONNAL");
                nodechild.addAttribute("nbrpattern",BasicUtility::Any2String(inputlayer->_neurons_matrix.getDomain()));
                nodechild.addAttribute("sizekernel",BasicUtility::Any2String(inputlayer->_size_convolutution));
                nodechild.addAttribute("subsampling",BasicUtility::Any2String(inputlayer->_sub_scaling));

                std::string weight_str;
                for(unsigned int index_w=0;index_w<inputlayer->_weights.size();index_w++){
                    NNWeight * weight = inputlayer->_weights[index_w];
                    weight_str+=BasicUtility::Any2String(weight->_Wn)+";";
                }
                nodechild.addAttribute("weight",weight_str);
            }else if(layer->_type==NNLayer::MATRIXCONVOLUTIONNAL2){
                NNLayerMatrixConvolutional* inputlayer = dynamic_cast<NNLayerMatrixConvolutional*>(layer);
                XMLNode nodechild = node.addChild("layer");
                nodechild.addAttribute("type","NNLayer::MATRIXCONVOLUTIONNAL2");
                nodechild.addAttribute("nbrpattern",BasicUtility::Any2String(inputlayer->_neurons_matrix.getDomain()));
                nodechild.addAttribute("sizekernel",BasicUtility::Any2String(inputlayer->_size_convolutution));

                std::string weight_str;
                for(unsigned int index_w=0;index_w<inputlayer->_weights.size();index_w++){
                    NNWeight * weight = inputlayer->_weights[index_w];
                    weight_str+=BasicUtility::Any2String(weight->_Wn)+";";
                }
                nodechild.addAttribute("weight",weight_str);
            }else if(layer->_type==NNLayer::MATRIXSUBSALE){
                NNLayerMatrixSubScale* inputlayer = dynamic_cast<NNLayerMatrixSubScale*>(layer);
                XMLNode nodechild = node.addChild("layer");
                nodechild.addAttribute("type","NNLayer::MATRIXSUBSALE");
                nodechild.addAttribute("nbrpattern",BasicUtility::Any2String(inputlayer->_neurons_matrix.getDomain()));
                nodechild.addAttribute("subsampling",BasicUtility::Any2String(inputlayer->_sub_scaling));
                std::string weight_str;
                for(unsigned int index_w=0;index_w<inputlayer->_weights.size();index_w++){
                    NNWeight * weight = inputlayer->_weights[index_w];
                    weight_str+=BasicUtility::Any2String(weight->_Wn)+";";
                }
                nodechild.addAttribute("weight",weight_str);
            }else if(layer->_type==NNLayer::MATRIXMAXPOOLING){
                NNLayerMatrixMaxPooling* inputlayer = dynamic_cast<NNLayerMatrixMaxPooling*>(layer);
                XMLNode nodechild = node.addChild("layer");
                nodechild.addAttribute("type","NNLayer::MATRIXMAXPOOLING");
                nodechild.addAttribute("nbrpattern",BasicUtility::Any2String(inputlayer->_neurons_matrix.getDomain()));
                nodechild.addAttribute("subsampling",BasicUtility::Any2String(inputlayer->_sub_scaling));
            }
            else if(layer->_type==NNLayer::FULLYCONNECTED)
            {
                XMLNode nodechild = node.addChild("layer");
                nodechild.addAttribute("type","NNLayer::FULLYCONNECTED");
                nodechild.addAttribute("size",BasicUtility::Any2String(layer->_neurons.size()));

                std::string weight_str;
                for(unsigned int index_w=0;index_w<layer->_weights.size();index_w++){
                    NNWeight * weight = layer->_weights[index_w];
                    weight_str+=BasicUtility::Any2String(weight->_Wn)+";";
                }
                nodechild.addAttribute("weight",weight_str);
            }
        }
    }
    doc.save(file);

}

void NeuralNetworkFeedForward::load(const char * file)
{
    XMLDocument doc;
    doc.load(file);
    load(doc);
}
void NeuralNetworkFeedForward::loadByteArray(const char *  file)
{
    XMLDocument doc;
    doc.loadFromByteArray(file);
    load(doc);

}

void NeuralNetworkFeedForward::load(XMLDocument &doc)
{
    this->init();
    XMLNode node1 = doc.getChild("label2String");
    std::string type1 = node1.getAttribute("id");
    BasicUtility::String2Any(type1,_label2string);
    XMLNode node = doc.getChild("layers");
    int i=0;
    for (XMLNode tool = node.firstChild(); tool; tool = tool.nextSibling(),++i)
    {
        std::string type = tool.getAttribute("type");
        if(i==0)
        {
            if(type=="NNLayer::INPUTMATRIX"){
                Vec2I32 domain;
                BasicUtility::String2Any(tool.getAttribute("size"),domain);

                NNLayerMatrix::CenteringMethod method_enum=NNLayerMatrix::BoundingBox;
                if(tool.hasAttribute("method")){
                    int method;
                    //                    std::cout<<tool.getAttribute("method")<<std::endl;
                    BasicUtility::String2Any(tool.getAttribute("method"),method);
                    method_enum = static_cast<NNLayerMatrix::CenteringMethod>(method) ;
                }

                NNLayerMatrix::NormalizationValue method_norm_enum=NNLayerMatrix::MinusOneToOne;
                if(tool.hasAttribute("normalization")){
                    int method_norm;
                    //                    std::cout<<tool.getAttribute("normalization")<<std::endl;
                    BasicUtility::String2Any(tool.getAttribute("normalization"),method_norm);
                    method_norm_enum= static_cast<NNLayerMatrix::NormalizationValue>(method_norm) ;
                }

                this->addInputLayerMatrix(domain(0),domain(1),method_enum,method_norm_enum);

            }else{
                int domain;
                BasicUtility::String2Any(tool.getAttribute("size"),domain);
                this->addInputLayer(domain);
            }
        }
        else
        {
            if(type=="NNLayer::MATRIXCONVOLUTIONNAL"){

                std::string str = tool.getAttribute("nbrpattern");
                int nbr_map;
                BasicUtility::String2Any(str,nbr_map);

                str = tool.getAttribute("sizekernel");
                int sizekernel;
                BasicUtility::String2Any(str,sizekernel);

                str = tool.getAttribute("subsampling");
                int subsampling;
                BasicUtility::String2Any(str,subsampling);

                this->addLayerConvolutionalPlusSubScaling(nbr_map,sizekernel,subsampling,-1);

                str = tool.getAttribute("weight");
                NNLayer* layer = *(this->_layers.rbegin());
                std::istringstream stream(str);
                for(unsigned int index_weight=0;index_weight<layer->_weights.size();index_weight++){
                    F32 weight ;
                    str = pop::BasicUtility::getline( stream, ";" );
                    pop::BasicUtility::String2Any(str,weight );
                    layer->_weights[index_weight]->_Wn = weight;
                }
            }else if(type=="NNLayer::MATRIXCONVOLUTIONNAL2"){

                std::string str = tool.getAttribute("nbrpattern");
                int nbr_map;
                BasicUtility::String2Any(str,nbr_map);

                str = tool.getAttribute("sizekernel");
                int sizekernel;
                BasicUtility::String2Any(str,sizekernel);

                this->addLayerConvolutional(nbr_map,sizekernel,-1);

                str = tool.getAttribute("weight");
                NNLayer* layer = *(this->_layers.rbegin());
                std::istringstream stream(str);
                for(unsigned int index_weight=0;index_weight<layer->_weights.size();index_weight++){
                    F32 weight ;
                    str = pop::BasicUtility::getline( stream, ";" );
                    pop::BasicUtility::String2Any(str,weight );
                    layer->_weights[index_weight]->_Wn = weight;
                }
            }else if(type=="NNLayer::MATRIXSUBSALE"){

                std::string str = tool.getAttribute("nbrpattern");
                int nbr_map;
                BasicUtility::String2Any(str,nbr_map);


                str = tool.getAttribute("subsampling");
                int subsampling;
                BasicUtility::String2Any(str,subsampling);

                this->addLayerSubScaling(subsampling,-1);

                str = tool.getAttribute("weight");
                NNLayer* layer = *(this->_layers.rbegin());
                std::istringstream stream(str);
                for(unsigned int index_weight=0;index_weight<layer->_weights.size();index_weight++){
                    F32 weight ;
                    str = pop::BasicUtility::getline( stream, ";" );
                    pop::BasicUtility::String2Any(str,weight );
                    layer->_weights[index_weight]->_Wn = weight;
                }
            }else if(type=="NNLayer::MATRIXMAXPOOLING"){

                std::string str = tool.getAttribute("nbrpattern");
                int nbr_map;
                BasicUtility::String2Any(str,nbr_map);
                str = tool.getAttribute("subsampling");
                int subsampling;
                BasicUtility::String2Any(str,subsampling);
                this->addLayerMaxPooling(subsampling);
            }
            else if(type=="NNLayer::FULLYCONNECTED")
            {
                std::string str = tool.getAttribute("size");
                int size;
                BasicUtility::String2Any(str,size);
                this->addLayerFullyConnected(size,-1);

                str = tool.getAttribute("weight");
                NNLayer* layer = *(this->_layers.rbegin());
                std::istringstream stream(str);
                for(unsigned int index_weight=0;index_weight<layer->_weights.size();index_weight++){
                    F32 weight ;
                    str = pop::BasicUtility::getline( stream, ";" );
                    pop::BasicUtility::String2Any(str,weight );
                    layer->_weights[index_weight]->_Wn = weight;
                }
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////
//
//  NNLayerMatrix class definition


NNLayerMatrix::NNLayerMatrix(unsigned int nbr_map,unsigned int height,unsigned int width ,NNLayerMatrix::CenteringMethod method,NormalizationValue normalization)
    :NNLayer(nbr_map*height*width)
{
    _method = method;
    _normalization_value = normalization;
    _neurons_matrix.resize(nbr_map,MatN<2,NNNeuron*>(Vec2I32(height,width)));
    for (unsigned int i_pattern=0; i_pattern<nbr_map; ++i_pattern )
    {

        for (unsigned int i_width=0; i_width<width; ++i_width )
        {
            for (unsigned int i_height=0; i_height<height; ++i_height )
            {
                _neurons_matrix(i_pattern)(i_height,i_width)=this->_neurons[i_width + i_height*width+i_pattern*width*height ];
            }
        }
    }
}
NNLayerMatrixConvolutionalPlusSubScaling::NNLayerMatrixConvolutionalPlusSubScaling(unsigned int nbr_map,unsigned int height,unsigned int width )
    :NNLayerMatrix(nbr_map,height, width )
{
}

NNLayerMatrixConvolutional::NNLayerMatrixConvolutional(unsigned int nbr_map,unsigned int height,unsigned int width )
    :NNLayerMatrix(nbr_map,height, width )
{
}
NNLayerMatrixSubScale::NNLayerMatrixSubScale(unsigned int nbr_map,unsigned int height,unsigned int width )
    :NNLayerMatrix(nbr_map,height, width )
{
}



NNLayerMatrixMaxPooling::NNLayerMatrixMaxPooling(unsigned int nbr_map,unsigned int height,unsigned int width )
    :NNLayerMatrix(0,0,0)
{
    for (unsigned int i_index=0; i_index<nbr_map*height*width; ++i_index ){
        this->_neurons.push_back( new NNNeuronMaxPool() );
    }
    this->_neurons_matrix.resize(nbr_map,MatN<2,NNNeuron*>(Vec2I32(height,width)));
    for (unsigned int i_pattern=0; i_pattern<nbr_map; ++i_pattern )
    {

        for (unsigned int i_width=0; i_width<width; ++i_width )
        {
            for (unsigned int i_height=0; i_height<height; ++i_height )
            {
                this->_neurons_matrix(i_pattern)(i_height,i_width)=this->_neurons[i_width + i_height*width+i_pattern*width*height ];
            }
        }
    }
}




void NNLayerMatrix::setLearningRate(F32 eta){
    if(this->_type==MATRIXCONVOLUTIONNAL){
        eta = eta/std::sqrt(_neurons_matrix(0).getDomain()(0)*_neurons_matrix(0).getDomain()(1)*1.f);
    }
    NNLayer::setLearningRate(eta);
}

///////////////////////////////////////////////////////////////////////
//
//  NNLayer class definition


NNLayer::NNLayer(unsigned int nbr_neuron )
{
    for (unsigned int i_index=0; i_index<nbr_neuron; ++i_index ){
        _neurons.push_back( new NNNeuron() );
    }
}
NNLayer::~NNLayer(){
    for (unsigned int i_index=0; i_index<_neurons.size(); ++i_index ){
        delete _neurons[i_index];
    }
    for (unsigned int i_index=0; i_index<_weights.size(); ++i_index ){
        delete _weights[i_index];
    }
}



void NNLayer::setLearningRate(F32 eta)
{
#if defined(HAVE_OPENMP)
#pragma omp parallel for
#endif
    for( int i=0;i<static_cast<int>(_weights.size());i++){
        _weights[i]->_eta = eta;
    }
}
void NNLayer::propagateFront()
{
#if defined(HAVE_OPENMP)
#pragma omp parallel for
#endif
    for( int i_neuron =0;i_neuron<static_cast<int>( _neurons.size()); i_neuron++ )
    {
        NNNeuron * n =(_neurons[i_neuron]);
        n->propagateFront();
    }
}
void NNLayer::initPropagateBackFirstDerivate(){
#if defined(HAVE_OPENMP)
#pragma omp parallel for
#endif
    for( int i_neuron =0;i_neuron< static_cast<int>(_neurons.size()); i_neuron++ )
    {
        NNNeuron * n =(_neurons[i_neuron]);
        n->initPropagateBackFirstDerivate();
    }
}
void NNLayer::propagateBackFirstDerivate()
{
#if defined(HAVE_OPENMP)
#pragma omp parallel for
#endif
    for( int i_neuron =0;i_neuron< static_cast<int>(_neurons.size()); i_neuron++ )
    {
        NNNeuron * n =(_neurons[i_neuron]);
        n->initPropagateBackFirstDerivate();
    }
#if defined(HAVE_OPENMP)
#pragma omp parallel for
#endif
    for( int i_neuron =0;i_neuron<static_cast<int>( _neurons.size()); i_neuron++ )
    {
        NNNeuron * n =(_neurons[i_neuron]);
        n->propagateBackFirstDerivate();
    }
}
void NNLayer::initPropagateBackSecondDerivate(){
#if defined(HAVE_OPENMP)
#pragma omp parallel for
#endif
    for( int i_neuron =0;i_neuron< static_cast<int>(_neurons.size()); i_neuron++ )
    {
        NNNeuron * n =(_neurons[i_neuron]);
        n->initPropagateBackSecondDerivate();
    }
}

void NNLayer::propagateBackSecondDerivate()
{
#if defined(HAVE_OPENMP)
#pragma omp parallel for
#endif
    for( int i_neuron =0;i_neuron< static_cast<int>(_neurons.size()); i_neuron++ )
    {
        NNNeuron * n =(_neurons[i_neuron]);
        n->initPropagateBackSecondDerivate();
    }
#if defined(HAVE_OPENMP)
#pragma omp parallel for
#endif
    for( int i_neuron =0;i_neuron<static_cast<int>( _neurons.size()); i_neuron++ )
    {
        NNNeuron * n =(_neurons[i_neuron]);
        n->propagateBackSecondDerivate();
    }
}


void NNLayer::learningFirstDerivate()
{
#if defined(HAVE_OPENMP)
#pragma omp parallel for
#endif
    for( int i_weight =0;i_weight< static_cast<int>(_weights.size()); i_weight++ )
    {
        NNWeight * w =(_weights[i_weight]);
        w->learningFirstDerivate();
    }
}

///////////////////////////////////////////////////////////////////////
//
//  NNWeight


NNWeight::NNWeight(  F32 Wn, F32 eta) :
    _Wn( Wn ),_eta(eta)
{
}
void NNWeight::learningFirstDerivate()
{
    _Wn = _Wn-_eta*_dE_dWn;
}

///////////////////////////////////////////////////////////////////////
//
//  NNNeuron


NNNeuron::NNNeuron(ActivationFunction f_act) :
    _f_act(f_act), _Xn( 0.0 )
{
}
NNNeuron::~NNNeuron(){
}

void NNNeuron::addConnection(NNWeight* weight , NNNeuron* neuron)
{
    _connections.push_back( NNConnection( weight, neuron ) );
}

void NNNeuron::addConnectionBiais(NNWeight* weight)
{
    _connections.push_back( NNConnection( weight ) );
}
void NNNeuron::propagateFront(){

    _Yn=0;
    for ( unsigned int i_connection =0;i_connection< _connections.size(); i_connection++  )
    {
        NNConnection& c = (_connections[i_connection]);
        F32 weight = c._weight->_Wn;
        F32 neuron_out_previous = (c.isBiais() ? 1 : c._neuron->_Xn);
        _Yn += weight*neuron_out_previous;
    }
    switch ( _f_act) {
    case SIGMOID_FUNCTION:
        _Xn = SIGMOID( _Yn );
        break;
    case IDENTITY_FUNCTION:
        _Xn = _Yn;
        break;
    default:
        _Xn = SIGMOID( _Yn );
        POP_DbgAssertMessage(false,"error not known function");
        break;
    }
}
void NNNeuron::initPropagateBackFirstDerivate(){
    _dErr_dYn=0;
    for ( unsigned int i_connection =0;i_connection< _connections.size(); i_connection++  )
    {
        NNConnection& c = (_connections[i_connection]);
        c._weight->_dE_dWn =0;
        if(!c.isBiais())
        {
            c._neuron->_dErr_dXn=0;
        }
    }
}
void NNNeuron::initPropagateBackSecondDerivate(){
    _d2Err_dYn2=0;
    for ( unsigned int i_connection =0;i_connection< _connections.size(); i_connection++  )
    {
        NNConnection& c = (_connections[i_connection]);
        c._weight->_d2E_dWn2 =0;
        if(!c.isBiais())
        {
            c._neuron->_d2Err_dXn2=0;
        }
    }
}

void NNNeuron::propagateBackFirstDerivate(){

    F32 fprime_Y;
    switch ( _f_act) {
    case SIGMOID_FUNCTION:
        fprime_Y = DSIGMOID( _Xn );
        break;
    case IDENTITY_FUNCTION:
        fprime_Y = 1;
        break;
    default:
        fprime_Y =  DSIGMOID( _Xn );
        POP_DbgAssertMessage(false,"error not known function");
        break;
    }

    _dErr_dYn = fprime_Y * _dErr_dXn;

    for ( unsigned int i_connection =0;i_connection< _connections.size(); i_connection++  )
    {
        NNConnection& c = (_connections[i_connection]);
        F32 Xnm1 = (c.isBiais() ? 1 : c._neuron->_Xn);
        c._weight->_dE_dWn += Xnm1 * _dErr_dYn;
        if(!c.isBiais())
        {
            c._neuron->_dErr_dXn += c._weight->_Wn * _dErr_dYn;
        }
    }
}
void NNNeuron::propagateBackSecondDerivate(){

    F32 fprime_Y;
    switch ( _f_act) {
    case SIGMOID_FUNCTION:
        fprime_Y = DSIGMOID( _Xn );
        break;
    case IDENTITY_FUNCTION:
        fprime_Y = 1;
        break;
    default:
        fprime_Y =  DSIGMOID( _Xn );
        POP_DbgAssertMessage(false,"error not known function");
        break;
    }

    _d2Err_dYn2 = fprime_Y*fprime_Y*_d2Err_dXn2;
    for ( unsigned int i_connection =0;i_connection< _connections.size(); i_connection++  )
    {
        NNConnection& c = (_connections[i_connection]);
        F32 Xnm1;
        Xnm1 = (c.isBiais() ? 1 : c._neuron->_Xn);
        c._weight->_d2E_dWn2 += Xnm1*Xnm1*_d2Err_dYn2;
        if(!c.isBiais())
        {
            c._neuron->_d2Err_dXn2+= c._weight->_Wn*c._weight->_Wn*_d2Err_dYn2;
        }
    }
}
NNConnection::NNConnection()
    :_weight(NULL),_neuron(NULL)
{

}


NNConnection::NNConnection(NNWeight* weight , NNNeuron* neuron)
    :_weight(weight),_neuron(neuron)
{

}

void TrainingNeuralNetwork::neuralNetworkForRecognitionForHandwrittenDigits(NeuralNetworkFeedForward &n,std::string train_datapath,  std::string train_labelpath,std::string test_datapath,  std::string test_labelpath,int lecun_or_simard,F32 elastic_distortion)
{

    Vec<Vec<Mat2UI8> > number_training =  loadMNIST(train_datapath,train_labelpath);
    Vec<Vec<Mat2UI8> > number_test =  loadMNIST(test_datapath,test_labelpath);

    //        number_training.resize(2);
    //        number_test.resize(2);

    if(lecun_or_simard==0){
        std::cout<<"LECUN"<<std::endl;
        //LeCun network lenet5
        Vec2I32 domain(28,28);
        n.addInputLayerMatrix(domain(0),domain(1));
        n.addLayerConvolutional(10,5,1);
        n.addLayerMaxPooling(2);
        n.addLayerConvolutional(50,5,1);
        n.addLayerMaxPooling(2);
        n.addLayerFullyConnected(100,1);
    }else if(lecun_or_simard==1){
        std::cout<<"SIMARD"<<std::endl;
        //Simard network
        Vec2I32 domain(29,29);
        n.addInputLayerMatrix(domain(0),domain(1));
        n.addLayerConvolutionalPlusSubScaling(6,5,2,1);
        n.addLayerConvolutionalPlusSubScaling(50,5,2,1);
        n.addLayerFullyConnected(100,1);
    }else{
        std::cout<<"TARIEL"<<std::endl;
        Vec2I32 domain(29,29);
        n.addInputLayerMatrix(domain(0),domain(1));
        n.addLayerConvolutionalPlusSubScaling(6,5,2,1);
        n.addLayerIntegral(500,1);
        n.addLayerFullyConnected(100,1);
    }
    n.addLayerFullyConnected(static_cast<unsigned int>(number_training.size()),1);

    Vec<std::string> label_digit;
    for(int i=0;i<10;i++)
        label_digit.push_back(BasicUtility::Any2String(i));
    n.label2String() = label_digit;


    F32 sigma_min =0.15f;
    F32 sigma_max =0.25f;
    F32 alpha =0.13f;

    F32 standard_deviation_angle = PI/4;
    F32 standard_deviation_shear_j = PI/6;
    DistributionUniformReal dAngle(-standard_deviation_angle,standard_deviation_angle);
    DistributionUniformReal dShear(-standard_deviation_shear_j,standard_deviation_shear_j);

    DistributionUniformReal d_deviation_length(sigma_min,sigma_max);
    DistributionUniformReal d_correlation_lenght(0,alpha);

    DistributionUniformReal d(0,0.5);
    DistributionSign sign;



    Vec<VecF32> vtraining_in;
    Vec<VecF32> vtraining_out;

    for(unsigned int i=0;i<number_training.size();i++){
        for(unsigned int j=0;j<number_training(i).size();j++){
            Mat2UI8 binary = number_training(i)(j);
            int size_i = binary.sizeI();
            Draw::addBorder(binary,10, UI8(0));

            //            disp.display(binary);
            // We apply elastic deformation and affine transformation (rotation, shear)

            Mat2UI8 binary_scale =  binary;//GeometricalTransformation::scale(binary,Vec3F32(2,2));
            for(unsigned int k=0;k<elastic_distortion;k++){
                F32 deviation_length_random = d_deviation_length.randomVariable();
                F32 correlation_lenght_random =d_correlation_lenght.randomVariable();
                //                std::cout<<"deviation_length "<<deviation_length_random<<std::endl;
                //                std::cout<<"correlation_lenght_random "<<correlation_lenght_random<<std::endl;
                Mat2UI8 m= GeometricalTransformation::elasticDeformation(binary_scale,deviation_length_random*size_i,correlation_lenght_random*size_i);
                F32 angle = dAngle.randomVariable();
                F32 shear = dShear.randomVariable();

                F32 alphax=1,alphay=1;
                if(sign.randomVariable()>0)
                    alphax= 1+d.randomVariable();
                else
                    alphay= 1/(1+d.randomVariable());
                Vec2F32 v(alphax,alphay);
                //                std::cout<<"scale "<<v<<std::endl;
                //                std::cout<<"angle "<<angle<<std::endl;
                //                std::cout<<"shear "<<shear<<std::endl;
                Mat2x33F32 maffine  = GeometricalTransformation::translation2DHomogeneousCoordinate(m.getDomain()/2);//go back to the buttom left corner (origin)
                maffine *=  GeometricalTransformation::scale2DHomogeneousCoordinate(v);
                maffine *=  GeometricalTransformation::shear2DHomogeneousCoordinate(shear,0);
                maffine *=  GeometricalTransformation::rotation2DHomogeneousCoordinate(angle);//rotate
                maffine *=  GeometricalTransformation::translation2DHomogeneousCoordinate(-m.getDomain()/2);
                m = GeometricalTransformation::transformHomogeneous2D(maffine, m);
                //                m.display();
                VecF32 vin = n.inputMatrixToInputNeuron(m);
                vtraining_in.push_back(vin);
                VecF32 v_out(static_cast<int>(number_training.size()),-1);
                v_out(i)=1;
                vtraining_out.push_back(v_out);
            }
            VecF32 vin = n.inputMatrixToInputNeuron(binary);
            vtraining_in.push_back(vin);
            VecF32 v_out(static_cast<int>(number_training.size()),-1);
            v_out(i)=1;
            vtraining_out.push_back(v_out);

        }
    }

    Vec<VecF32> vtest_in;
    Vec<VecF32> vtest_out;
    for(unsigned int i=0;i<number_test.size();i++){
        for(unsigned int j=0;j<number_test(i).size();j++){
            Mat2UI8 binary = number_test(i)(j);
            VecF32 vin = n.inputMatrixToInputNeuron(binary);
            vtest_in.push_back(vin);
            VecF32 v_out(static_cast<int>(number_test.size()),-1);
            v_out(i)=1;
            vtest_out.push_back(v_out);
        }
    }

    trainingFirstDerivative(n,vtraining_in,vtraining_out,vtest_in,vtest_out,0.01f,50);
}

//static std::string getCurrentTime() {
//    time_t rawtime;
//    time(&rawtime);
//    struct tm * timeinfo = localtime(&rawtime);
//    return std::string(asctime(timeinfo));
//}

void TrainingNeuralNetwork::trainingFirstDerivative(NeuralNetworkFeedForward&n, const Vec<VecF32>& trainingins, const Vec<VecF32>& trainingouts, F32 eta, unsigned int nbr_epoch)
{
    n.setLearningRate(eta);
    std::vector<int> v_global_rand(trainingins.size());
    for(unsigned int i=0;i<v_global_rand.size();i++)
        v_global_rand[i]=i;
    std::cout<<"iter_epoch\t error_train"<<std::endl;

    for(unsigned int i=0;i<nbr_epoch;i++){
        std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,Distribution::irand());
        int error=0;
        for(unsigned int j=0;j<v_global_rand.size();j++){
            VecF32 vout;
            n.propagateFront(trainingins(v_global_rand[j]),vout);
            n.propagateBackFirstDerivate(trainingouts(v_global_rand[j]));
            n.learningFirstDerivate();
            int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
            int label2 = std::distance(trainingouts(v_global_rand[j]).begin(),std::max_element(trainingouts(v_global_rand[j]).begin(),trainingouts(v_global_rand[j]).end()));
            if(label1!=label2)
                error++;
        }

        std::cout<<i<<"\t"<<error*1.0/v_global_rand.size()<<std::endl;
    }
}

void TrainingNeuralNetwork::trainingFirstDerivative(NeuralNetworkFeedForward&n,const Vec<VecF32>& trainingins,const Vec<VecF32>& trainingouts,const Vec<VecF32>& testins,const Vec<VecF32>& testouts,F32 eta,unsigned int nbr_epoch)
{
    n.setLearningRate(eta);
    std::vector<int> v_global_rand(trainingins.size());
    for(unsigned int i=0;i<v_global_rand.size();i++)
        v_global_rand[i]=i;
    std::cout<<"iter_epoch\t error_train\t error_test\t learning rate"<<std::endl;

    for(unsigned int i=0;i<nbr_epoch;i++){
        std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,Distribution::irand());
        int error_training=0,error_test=0;
        for(unsigned int j=0;j<v_global_rand.size();j++){
            VecF32 vout;
            n.propagateFront(trainingins(v_global_rand[j]),vout);
            n.propagateBackFirstDerivate(trainingouts(v_global_rand[j]));
            n.learningFirstDerivate();
            int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
            int label2 = std::distance(trainingouts(v_global_rand[j]).begin(),std::max_element(trainingouts(v_global_rand[j]).begin(),trainingouts(v_global_rand[j]).end()));
            if(label1!=label2)
                error_training++;
        }
        //        std::cout<<testins.size()<<std::endl;

        for(unsigned int j=0;j<testins.size();j++){
            VecF32 vout;
            n.propagateFront(testins(j),vout);
            int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
            int label2 = std::distance(testouts(j).begin(),std::max_element(testouts(j).begin(),testouts(j).end()));
            if(label1!=label2)
                error_test++;
        }
        n.save(("neuralnetwork"+BasicUtility::IntFixedDigit2String(i,2)+".xml").c_str() );

        std::cout<<i<<"\t"<<error_training*1./trainingins.size()<<"\t"<<error_test*1.0/testins.size() <<"\t"<<eta<<std::endl;
        eta *=0.9f;
        n.setLearningRate(eta);
    }
}


//void NeuralNetworkFeedForwardMerge::propagateFront(const pop::VecF32& in , pop::VecF32 &out){
//    pop::VecF32 out1;
//    n1->propagateFront(in,out1);
//    pop::VecF32 out2;
//    n2->propagateFront(in,out2);
//    out1.insert( out1.end(), out2.begin(), out2.end() );
//    n.propagateFront(out1,out);
//}

//void NeuralNetworkFeedForwardMerge::propagateBackFirstDerivate(const pop::VecF32& desired_output){
//    n.propagateBackFirstDerivate(desired_output);
//    NNLayer & layerfirst = *(n._layers[0]);

//    int size1 = n1->_layers[n1->_layers.size()-1]->_neurons.size();
//    int size2 = n2->_layers[n2->_layers.size()-1]->_neurons.size();
//    pop::VecF32 desired_output1(size1);
//    pop::VecF32 desired_output2(size2);

//    for(unsigned int i =0;i<layerfirst._neurons.size();i++){
//        NNNeuron *  neuron = layerfirst._neurons[i];
//        if(i<size1)
//            desired_output1(i)=neuron->_Xn;
//        if(i>=size1)
//            desired_output2(i-size1)=neuron->_Xn;
//    }
//    n1->propagateBackFirstDerivate(desired_output1);
//    n2->propagateBackFirstDerivate(desired_output2);
//}

//void NeuralNetworkFeedForwardMerge::learningFirstDerivate(){
//    n.learningFirstDerivate();
//    n1->learningFirstDerivate();
//    n2->learningFirstDerivate();
//}

//void NeuralNetworkFeedForwardMerge::setLearningRate(F32 eta){
//    n.setLearningRate(eta);
//    n1->setLearningRate(eta);
//    n2->setLearningRate(eta);
//}

//void NeuralNetworkFeedForwardMerge::addLayerFullyConnected(int nbr_output_neuron,NeuralNetworkFeedForward * nn1,NeuralNetworkFeedForward* nn2){
//    n1= nn1;
//    n2= nn2;
//    n.addInputLayer(n1->_layers[n1->_layers.size()-1]->_neurons.size()+n2->_layers[n2->_layers.size()-1]->_neurons.size());
//    n.addLayerFullyConnected(nbr_output_neuron);
//}
void NNNeuronMaxPool::addConnection( NNNeuron* neuron){
    this->_connections.push_back( NNConnection( NULL, neuron ) );
}

void NNNeuronMaxPool::propagateFront(){
    this->_Yn=-(std::numeric_limits<F32>::max)();
    for ( unsigned int i_connection =0;i_connection< _connections.size(); i_connection++  )
    {

        NNConnection& c = (_connections[i_connection]);
        this->_Yn = (std::max)(_Yn,c._neuron->_Xn);
    }
    this->_Xn = this->_Yn;
}

void NNNeuronMaxPool::propagateBackFirstDerivate(){

    _dErr_dYn = _dErr_dXn;
    F32 max_value = -(std::numeric_limits<F32>::max)();
    unsigned int i_index_max=0;
    for ( unsigned int i_connection =0;i_connection< _connections.size(); i_connection++  )
    {
        NNConnection& c = (_connections[i_connection]);
        NNNeuron * neuron = c._neuron;
        c._neuron->_dErr_dXn = _dErr_dXn;
        if(neuron->_Xn>max_value){
            i_index_max = i_connection;
            max_value = neuron->_Xn;
        }
    }

    NNConnection& c = (_connections[i_index_max]);
    c._neuron->_dErr_dXn = 0;
}

void NNNeuronMaxPool::initPropagateBackFirstDerivate(){
    _dErr_dYn=0;
    for ( unsigned int i_connection =0;i_connection< _connections.size(); i_connection++  )
    {
        NNConnection& c = (_connections[i_connection]);
        c._neuron->_dErr_dXn=0;
    }
}
int TrainingNeuralNetwork::_reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
Vec<Vec<Mat2UI8> > TrainingNeuralNetwork::loadMNIST( std::string datapath,  std::string labelpath){
    Vec<Vec<Mat2UI8> > dataset(10);
    std::ifstream datas(datapath.c_str(),std::ios::binary);
    std::ifstream labels(labelpath.c_str(),std::ios::binary);

    if (!datas.is_open() || !labels.is_open()){
        std::cerr<<"binary files could not be loaded" << std::endl;
        return dataset;
    }

    int magic_number=0; int number_of_images=0;int r; int c;
    int n_rows=0; int n_cols=0; unsigned char temp=0;

    // parse data header
    datas.read((char*)&magic_number,sizeof(magic_number));
    magic_number=_reverseInt(magic_number);
    datas.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images=_reverseInt(number_of_images);
    datas.read((char*)&n_rows,sizeof(n_rows));
    n_rows=_reverseInt(n_rows);
    datas.read((char*)&n_cols,sizeof(n_cols));
    n_cols=_reverseInt(n_cols);

    // parse label header - ignore
    int dummy;
    labels.read((char*)&dummy,sizeof(dummy));
    labels.read((char*)&dummy,sizeof(dummy));

    for(int i=0;i<number_of_images;++i){
        pop::Mat2UI8 img(n_rows,n_cols);

        for(r=0;r<n_rows;++r){
            for(c=0;c<n_cols;++c){
                datas.read((char*)&temp,sizeof(temp));
                img(r,c) = temp;
            }
        }
        labels.read((char*)&temp,sizeof(temp));
        dataset[(int)temp].push_back(img);
    }
    return dataset;
}

Vec<pop::Mat2UI8> TrainingNeuralNetwork::geometricalTransformationDataBaseMatrix( Vec<pop::Mat2UI8>  number_training,
                                                                                  unsigned int number,
                                                                                  F32 sigma_elastic_distortion_min,
                                                                                  F32 sigma_elastic_distortion_max,
                                                                                  F32 alpha_elastic_distortion_min,
                                                                                  F32 alpha_elastic_distortion_max,
                                                                                  F32 beta_angle_degree_rotation,
                                                                                  F32 beta_angle_degree_shear,
                                                                                  F32 gamma_x_scale,
                                                                                  F32 gamma_y_scale){

    DistributionUniformReal dAngle(-beta_angle_degree_rotation*pop::PI/180,beta_angle_degree_rotation*pop::PI/180);
    DistributionUniformReal dShear(-beta_angle_degree_shear*pop::PI/180,beta_angle_degree_shear*pop::PI/180);

    DistributionUniformReal d_deviation_length(sigma_elastic_distortion_min,sigma_elastic_distortion_max);
    DistributionUniformReal d_correlation_lenght(alpha_elastic_distortion_min,alpha_elastic_distortion_max);

    DistributionUniformReal d_scale_x(1-gamma_x_scale/100,1+gamma_x_scale/100);
    DistributionUniformReal d_scale_y(1-gamma_y_scale/100,1+gamma_y_scale/100);


    Vec<pop::Mat2UI8> v_out_i;
    for(unsigned int j=0;j<number_training.size();j++){

        Mat2UI8 binary = number_training(j);
        v_out_i.push_back(binary);
        Draw::addBorder(binary,2, UI8(0));


        Mat2UI8 binary_scale =  binary;
        for(unsigned int k=0;k<number;k++){
            F32 deviation_length_random = d_deviation_length.randomVariable();
            F32 correlation_lenght_random =d_correlation_lenght.randomVariable();
            Mat2UI8 m= GeometricalTransformation::elasticDeformation(binary_scale,deviation_length_random,correlation_lenght_random);
            F32 angle = dAngle.randomVariable();
            F32 shear = dShear.randomVariable();

            F32 alphax=d_scale_x.randomVariable();
            F32 alphay=d_scale_y.randomVariable();

            Vec2F32 v(alphax,alphay);
            //                std::cout<<"scale "<<v<<std::endl;
            //                std::cout<<"angle "<<angle<<std::endl;
            //                std::cout<<"shear "<<shear<<std::endl;
            Mat2x33F32 maffine  = GeometricalTransformation::translation2DHomogeneousCoordinate(m.getDomain()/2);//go back to the buttom left corner (origin)
            maffine *=  GeometricalTransformation::scale2DHomogeneousCoordinate(v);
            maffine *=  GeometricalTransformation::shear2DHomogeneousCoordinate(shear,0);
            maffine *=  GeometricalTransformation::rotation2DHomogeneousCoordinate(angle);//rotate
            maffine *=  GeometricalTransformation::translation2DHomogeneousCoordinate(-m.getDomain()/2);
            m = GeometricalTransformation::transformHomogeneous2D(maffine, m);
            //                F32 sum2=0;
            //                ForEachDomain2D(x,m){
            //                    sum2+=m(x);
            //                }
            //                std::cout<<sum2/sum<<std::endl;
            //             m.display();
            v_out_i.push_back(m);
        }
    }
    return v_out_i;
}
void TrainingNeuralNetwork::convertMatrixToInputValueNeuron(Vec<VecF32> &v_neuron_in, Vec<VecF32> &v_neuron_out,const Vec<Vec<pop::Mat2UI8> >& number_training,Vec2I32 domain ,NNLayerMatrix::CenteringMethod method,NNLayerMatrix::NormalizationValue normalization_value){



    for(unsigned int i=0;i<number_training.size();i++){
        for(unsigned int j=0;j<number_training(i).size();j++){
            Mat2UI8 binary = number_training(i)(j);

            VecF32 vin = NNLayerMatrix::inputMatrixToInputNeuron(binary,domain,method,normalization_value);
            v_neuron_in.push_back(vin);
            VecF32 v_out(static_cast<int>(number_training.size()),-1);
            v_out(i)=1;
            v_neuron_out.push_back(v_out);

        }
    }
}

void NeuralLayer::setLearnableParameter(F32 mu){
    _mu = mu;
}
NeuralLayerLinear::NeuralLayerLinear(unsigned int nbr_neurons)
    :_Y(nbr_neurons),_X(nbr_neurons)
{
}
VecF32& NeuralLayerLinear::X(){return _X;}
const VecF32& NeuralLayerLinear::X()const{return _X;}
VecF32& NeuralLayerLinear::d_E_X(){return _d_E_X;}
void NeuralLayerLinear::setTrainable(bool istrainable){
    if(istrainable==true){
        this->_d_E_Y = this->_Y;
        this->_d_E_X = this->_X;
    }else{
        this->_d_E_Y.clear();
        this->_d_E_X.clear();
    }
}
NeuralLayerMatrix::NeuralLayerMatrix(unsigned int sizei,unsigned int sizej,unsigned int nbr_map)
    :NeuralLayerLinear(sizei* sizej*nbr_map)
{
    for(unsigned int i=0;i<nbr_map;i++){
        MatNReference<2,F32> m(Vec2I32(sizei, sizej),this->_Y.data()+sizei*sizej*i);
        _Y_reference.push_back(MatNReference<2,F32>(Vec2I32(sizei, sizej),this->_Y.data()+sizei*sizej*i));
        _X_reference.push_back(MatNReference<2,F32>(Vec2I32(sizei, sizej),this->_X.data()+sizei*sizej*i));

    }
}
const Vec<MatNReference<2,F32> > & NeuralLayerMatrix::X_map()const{return _X_reference;}
Vec<MatNReference<2,F32> >& NeuralLayerMatrix::X_map(){return _X_reference;}



const Vec<MatNReference<2,F32> > & NeuralLayerMatrix::d_E_X_map()const{return _d_E_X_reference;}
Vec<MatNReference<2,F32> >& NeuralLayerMatrix::d_E_X_map(){return _d_E_X_reference;}


void NeuralLayerMatrix::setTrainable(bool istrainable){
    NeuralLayerLinear::setTrainable(istrainable);
    if(istrainable==true){
        for(unsigned int i=0;i<_X_reference.size();i++){
            _d_E_Y_reference.push_back(MatNReference<2,F32>(_X_reference(0).getDomain(),_d_E_Y.data()+_X_reference(0).getDomain().multCoordinate()*i));
            _d_E_X_reference.push_back(MatNReference<2,F32>(_X_reference(0).getDomain(),_d_E_X.data()+_X_reference(0).getDomain().multCoordinate()*i));
        }
    }else{
        this->_d_E_Y_reference.clear();
        this->_d_E_X_reference.clear();
    }
}
NeuralLayerLinearFullyConnected::NeuralLayerLinearFullyConnected(unsigned int nbr_neurons_previous,unsigned int nbr_neurons)
    :NeuralLayerLinear(nbr_neurons),_W(nbr_neurons,nbr_neurons_previous+1),_X_biais(nbr_neurons_previous+1,1)
{
    //normalize tbe number inverse square root of the connection feeding into the nodes)
    DistributionNormal n(0,1.f/std::sqrt(nbr_neurons_previous+1.f));
    for(unsigned int i=0;i<_W.size();i++){
        _W(i)=n.randomVariable();
    }
}
void NeuralLayerLinearFullyConnected::setTrainable(bool istrainable){
    NeuralLayerLinear::setTrainable(istrainable);
    if(istrainable==true){
        this->_d_E_W = this->_W;
    }else{
        this->_d_E_W.clear();
    }
}

void NeuralLayerLinearFullyConnected::forwardCPU(const NeuralLayer& layer_previous){
    std::copy(layer_previous.X().begin(),layer_previous.X().end(),this->_X_biais.begin());
    this->_Y = this->_W * this->_X_biais;
    for(unsigned int i=0;i<_Y.size();i++){
        this->_X(i) = NeuronSigmoid::activation(this->_Y(i));
    }

}
void NeuralLayerLinearFullyConnected::backwardCPU(NeuralLayer& layer_previous){

    VecF32& d_E_X_previous= layer_previous.d_E_X();
    for(unsigned int i=0;i<this->_Y.size();i++){
        this->_d_E_Y(i) = this->_d_E_X(i)*NeuronSigmoid::derivedActivation(this->_X(i));
    }
    for(unsigned int i=0;i<this->_W.sizeI();i++){
        for(unsigned int j=0;j<this->_W.sizeJ();j++){
            this->_d_E_W(i,j)=this->_X_biais(j)*this->_d_E_Y(i);
        }
    }
    for(unsigned int j=0;j<d_E_X_previous.size();j++){
        d_E_X_previous(j)=0;
        for(unsigned int i=0;i<this->_W.sizeI();i++){
            d_E_X_previous(j)+=this->_d_E_Y(i)*this->_W(i,j);
        }
    }
}
void NeuralLayerLinearFullyConnected::learn(){
    for(unsigned int i=0;i<this->_W.sizeI();i++){
        for(unsigned int j=0;j<this->_W.sizeJ();j++){
            this->_W(i,j)= this->_W(i,j) -  this->_mu*this->_d_E_W(i,j);
        }
    }
}
NeuralLayer * NeuralLayerLinearFullyConnected::clone(){
    return new   NeuralLayerLinearFullyConnected(*this);
}

NeuralLayerMatrixConvolutionSubScaling::NeuralLayerMatrixConvolutionSubScaling(unsigned int nbr_map,unsigned int sub_scaling_factor,unsigned int radius_kernel,unsigned int sizei_map_previous,unsigned int sizej_map_previous,unsigned int nbr_map_previous)
    :NeuralLayerMatrix(std::floor (  (sizei_map_previous-1-2*radius_kernel)/(1.*sub_scaling_factor))+1,std::floor (  (sizej_map_previous-1-2*radius_kernel)/(1.*sub_scaling_factor))+1,nbr_map),
      _W_kernels(nbr_map*nbr_map_previous,Mat2F32(radius_kernel*2+1,radius_kernel*2+1)),
      _W_biais(nbr_map*nbr_map_previous),
      _sub_resolution_factor (sub_scaling_factor),
      _radius_kernel (radius_kernel)
{
    //normalize tbe number inverse square root of the connection feeding into the nodes)
    DistributionNormal n(0,1.f/((radius_kernel*2+1)*std::sqrt(nbr_map_previous*1.)));
    for(unsigned int i = 0;i<_W_kernels.size();i++){
        for(unsigned int j = 0;j<_W_kernels(i).size();j++){
            _W_kernels(i)(j)=n.randomVariable();
        }
        _W_biais(i)=n.randomVariable();
    }
}
void NeuralLayerMatrixConvolutionSubScaling::setTrainable(bool istrainable){
    NeuralLayerMatrix::setTrainable(istrainable);
    if(istrainable==true){
        _d_E_W_kernels = _W_kernels;
        _d_E_W_biais   = _W_biais;
    }else{
        _d_E_W_kernels.clear();
        _d_E_W_biais.clear();
    }
}
void NeuralLayerMatrixConvolutionSubScaling::forwardCPU(const NeuralLayer& layer_previous){

    if(const NeuralLayerMatrix * neural_matrix = dynamic_cast<const NeuralLayerMatrix *>(&layer_previous)){

#if defined(HAVE_OPENMP)
    #pragma omp parallel for
#endif
        for( int index_map=0;index_map<static_cast<int>(this->_Y_reference.size());index_map++){
            MatNReference<2,F32> &map_out =  this->_Y_reference[index_map];
            unsigned int index_start_kernel = index_map*neural_matrix->X_map().size();
            for(unsigned int i_map_next=0,i_map_previous=_radius_kernel;i_map_next<map_out.sizeI();i_map_next++,i_map_previous+=_sub_resolution_factor){
                for(unsigned int j_map_next=0,j_map_previous=_radius_kernel;j_map_next<map_out.sizeJ();j_map_next++,j_map_previous+=_sub_resolution_factor){
                    F32 sum=0;
                    //convolution
                    for(unsigned int index_map_previous=0;index_map_previous<neural_matrix->X_map().size();index_map_previous++){
                        sum+=_W_biais[ index_map_previous + index_start_kernel];
                        for(unsigned int i=0,index_kernel_ij=0,index_map = (i_map_previous-_radius_kernel)*neural_matrix->X_map()(0).sizeJ()+(j_map_previous-_radius_kernel);i<_W_kernels(0).sizeI();i++,index_map+=(neural_matrix->X_map()(0).sizeJ()-_W_kernels(0).sizeJ())){
                            for(unsigned int j=0;j<_W_kernels(0).sizeJ();j++,index_map++,index_kernel_ij++){
                                sum+=_W_kernels(index_map_previous + index_start_kernel)(index_kernel_ij)*neural_matrix->X_map()(index_map_previous)(index_map);
                            }
                        }
                    }
                    map_out(i_map_next,j_map_next)=sum;
                }
            }
        }
    }
    for(unsigned int i=0;i<_Y.size();i++){
        this->_X(i) = NeuronSigmoid::activation(this->_Y(i));
    }

}
void NeuralLayerMatrixConvolutionSubScaling::backwardCPU(NeuralLayer& layer_previous){
    for(unsigned int i=0;i<this->_Y.size();i++){
        this->_d_E_Y(i) = this->_d_E_X(i)*NeuronSigmoid::derivedActivation(this->_X(i));
    }

    //reset
    for(unsigned int i=0;i<this->_d_E_W_kernels.size();i++){
        for(unsigned int j=0;j<this->_d_E_W_kernels(i).size();j++){
            this->_d_E_W_kernels(i)(j)=0;
        }
    }
    for(unsigned int i=0;i<this->_d_E_W_biais.size();i++){
        this->_d_E_W_biais(i)=0;
    }


    if( NeuralLayerMatrix * neural_matrix = dynamic_cast< NeuralLayerMatrix *>(&layer_previous)){
        for(unsigned int i=0;i<neural_matrix->d_E_X_map().size();i++){
            for(unsigned int j=0;j<neural_matrix->d_E_X_map()(i).size();j++){
                neural_matrix->d_E_X_map()(i)(j)=0;
            }
        }
        //            std::cout<<"yep"<<std::endl;
        for(unsigned int index_map=0;index_map<this->_d_E_Y_reference.size();index_map++){
            MatNReference<2,F32> &map_error_out =  this->_d_E_Y_reference[index_map];
            //                std::cout<<map_error_out<<std::endl;
            //                std::cout<<this->_d_E_X<<std::endl;
            unsigned int index_start_kernel = index_map*neural_matrix->X_map().size();
            for(unsigned int i_map_next=0,i_map_previous=_radius_kernel;i_map_next<map_error_out.sizeI();i_map_next++,i_map_previous+=_sub_resolution_factor){
                for(unsigned int j_map_next=0,j_map_previous=_radius_kernel;j_map_next<map_error_out.sizeJ();j_map_next++,j_map_previous+=_sub_resolution_factor){
                    F32 d_error_y_value = map_error_out(i_map_next,j_map_next);

                    //convolution
                    for(unsigned int index_map_previous=0;index_map_previous<neural_matrix->X_map().size();index_map_previous++){
                        this->_d_E_W_biais(index_map_previous + index_start_kernel)+=d_error_y_value;
                        for(unsigned int i=0,index_kernel_ij=0,index_map = (i_map_previous-_radius_kernel)*neural_matrix->X_map()(0).sizeJ()+(j_map_previous-_radius_kernel);i<_W_kernels(0).sizeI();i++,index_map+=(neural_matrix->X_map()(0).sizeJ()-_W_kernels(0).sizeJ())){
                            for(unsigned int j=0;j<_W_kernels(0).sizeJ();j++,index_map++,index_kernel_ij++){

                                this->_d_E_W_kernels(index_map_previous + index_start_kernel)(index_kernel_ij)+=neural_matrix->X_map()(index_map_previous)(index_map)*d_error_y_value;
                                neural_matrix->d_E_X_map()(index_map_previous)(index_map)+=  this->_W_kernels(index_map_previous + index_start_kernel)(index_kernel_ij)  *d_error_y_value;
                            }
                        }
                    }
                }
            }
        }

    }
}
void NeuralLayerMatrixConvolutionSubScaling::learn(){
    for(unsigned int i=0;i<this->_d_E_W_kernels.size();i++){
        for(unsigned int j=0;j<this->_d_E_W_kernels(i).size();j++){
            this->_W_kernels(i)(j)=this->_W_kernels(i)(j)-_mu*this->_d_E_W_kernels(i)(j);
        }
    }
    for(unsigned int i=0;i<this->_d_E_W_biais.size();i++){
        this->_W_biais(i)=this->_W_biais(i)-_mu*this->_d_E_W_biais(i);
    }
}
NeuralLayer * NeuralLayerMatrixConvolutionSubScaling::clone(){
    NeuralLayerMatrixConvolutionSubScaling * layer = new NeuralLayerMatrixConvolutionSubScaling(*this);
    layer->_Y_reference.clear();
    layer->_X_reference.clear();
    for(unsigned int i=0;i<this->X_map().size();i++){
        layer->_Y_reference.push_back(MatNReference<2,F32>(this->X_map()(0).getDomain(),layer->_Y.data()+this->X_map()(0).getDomain().multCoordinate()*i));
        layer->_X_reference.push_back(MatNReference<2,F32>(this->X_map()(0).getDomain(),layer->_X.data()+this->X_map()(0).getDomain().multCoordinate()*i));
    }
    return layer;
}
NeuralLayerLinearInput::NeuralLayerLinearInput(unsigned int nbr_neurons)
    :NeuralLayerLinear(nbr_neurons){}
void NeuralLayerLinearInput::forwardCPU(const NeuralLayer& ) {}
void NeuralLayerLinearInput::backwardCPU(NeuralLayer& ) {}
void NeuralLayerLinearInput::learn(){}
void NeuralLayerLinearInput::setTrainable(bool istrainable){NeuralLayerLinear::setTrainable(istrainable);}
NeuralLayer * NeuralLayerLinearInput::clone(){
    return new NeuralLayerLinearInput(*this);
}
NeuralLayerMatrixInput::NeuralLayerMatrixInput(unsigned int sizei,unsigned int sizej,unsigned int nbr_map)
    :NeuralLayerMatrix(sizei,  sizej,  nbr_map){}
void NeuralLayerMatrixInput::forwardCPU(const NeuralLayer& ) {}
void NeuralLayerMatrixInput::backwardCPU(NeuralLayer& ) {}
void NeuralLayerMatrixInput::learn(){}
void NeuralLayerMatrixInput::setTrainable(bool istrainable){NeuralLayerMatrix::setTrainable(istrainable);}
NeuralLayer * NeuralLayerMatrixInput::clone(){
    NeuralLayerMatrixInput * layer = new NeuralLayerMatrixInput(this->X_map()(0).sizeI(),this->X_map()(0).sizeJ(),this->X_map().size());
    return layer;
}

NeuralNet::NeuralNet(NNLayerMatrix::CenteringMethod method,NNLayerMatrix::NormalizationValue normalization)
    :_method(method),_normalization_value(normalization)
{}

NeuralNet::NeuralNet(const NeuralNet & neural){

    this->_label2string = neural._label2string;
    this->_method = neural._method;
    this->_normalization_value = neural._normalization_value;

    this->clear();
    for(unsigned int i=0;i<neural._v_layer.size();i++){
        this->_v_layer.push_back(neural._v_layer(i)->clone());
    }
}

NeuralNet & NeuralNet::operator =(const NeuralNet & neural){
    this->_label2string = neural._label2string;
    this->_method = neural._method;
    this->_normalization_value = neural._normalization_value;
    this->clear();
    for(unsigned int i=0;i<neural._v_layer.size();i++){
        this->_v_layer.push_back(neural._v_layer(i)->clone());
    }
    return *this;
}

NeuralNet::~NeuralNet(){
    clear();
}

void NeuralNet::addLayerLinearInput(unsigned int nbr_neurons){
    this->_v_layer.push_back(new NeuralLayerLinearInput(nbr_neurons));
}
void NeuralNet::addLayerMatrixInput(unsigned int size_i,unsigned int size_j,unsigned int nbr_map){
    this->_v_layer.push_back(new NeuralLayerMatrixInput(size_i,size_j,nbr_map));
}
void NeuralNet::addLayerLinearFullyConnected(unsigned int nbr_neurons){
    if(_v_layer.size()==0){
        this->_v_layer.push_back(new NeuralLayerLinearFullyConnected(0,nbr_neurons));
    }else{
        this->_v_layer.push_back(new NeuralLayerLinearFullyConnected((*(_v_layer.rbegin()))-> X().size(),nbr_neurons));
    }
}
void NeuralNet::addLayerMatrixConvolutionSubScaling(unsigned int nbr_map,unsigned int sub_scaling_factor,unsigned int radius_kernel){
    if(NeuralLayerMatrix * neural_matrix = dynamic_cast<NeuralLayerMatrix *>(*(_v_layer.rbegin()))){
        this->_v_layer.push_back(new NeuralLayerMatrixConvolutionSubScaling( nbr_map, sub_scaling_factor,  radius_kernel,neural_matrix->X_map()(0).sizeI(),neural_matrix->X_map()(0).sizeJ(),neural_matrix->X_map().size()));
    }
}

void NeuralNet::setLearnableParameter(F32 mu){
    for(unsigned int i=0;i<_v_layer.size();i++){
        _v_layer(i)->setLearnableParameter(mu);
    }
}

void NeuralNet::setTrainable(bool istrainable){
    for(unsigned int i=0;i<_v_layer.size();i++){
        _v_layer(i)->setTrainable(istrainable);
    }
}
void NeuralNet::learn(){
    for(unsigned int i=0;i<_v_layer.size();i++){
        _v_layer(i)->learn();
    }
}
void NeuralNet::forwardCPU(const VecF32& X_in, VecF32& X_out){
    std::copy(X_in.begin(),X_in.end(), (*(_v_layer.begin()))->X().begin());
    for(unsigned int i=1;i<_v_layer.size();i++){
        _v_layer(i)->forwardCPU(*_v_layer(i-1));
        //            std::cout<<_v_layer(i)->X()<<std::endl;
    }
    if(X_out.size()!=(*(_v_layer.rbegin()))->X().size()){
        X_out.resize((*(_v_layer.rbegin()))->X().size());
    }
    std::copy((*(_v_layer.rbegin()))->X().begin(),(*(_v_layer.rbegin()))->X().end(),X_out.begin());
}

void NeuralNet::backwardCPU(const VecF32& X_expected){

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
void NeuralNet::clear(){
    for(unsigned int i=0;i<_v_layer.size();i++){
        delete _v_layer[i];
    }
    _v_layer.clear();
}
void NeuralNet::load(const char * file)
{
    XMLDocument doc;
    doc.load(file);
    load(doc);
}
void NeuralNet::loadByteArray(const char *  file)
{
    XMLDocument doc;
    doc.loadFromByteArray(file);
    load(doc);

}

void NeuralNet::load(XMLDocument &doc)
{
    this->clear();
    XMLNode node1 = doc.getChild("label2String");
    std::string type1 = node1.getAttribute("id");
    BasicUtility::String2Any(type1,_label2string);
    XMLNode node = doc.getChild("layers");
    int i=0;
    for (XMLNode tool = node.firstChild(); tool; tool = tool.nextSibling(),++i){
        std::string type = tool.getAttribute("type");
        if(type=="NNLayer::INPUTMATRIX"){
            Vec2I32 domain;
            int nbr_map;
            BasicUtility::String2Any(tool.getAttribute("size"),domain);
            BasicUtility::String2Any(tool.getAttribute("nbr_map"),nbr_map);
            if(tool.hasAttribute("method")){
                int method;
                BasicUtility::String2Any(tool.getAttribute("method"),method);
                _method = static_cast<NNLayerMatrix::CenteringMethod>(method) ;
            }
            if(tool.hasAttribute("normalization")){
                int method_norm;
                BasicUtility::String2Any(tool.getAttribute("normalization"),method_norm);
                _normalization_value= static_cast<NNLayerMatrix::NormalizationValue>(method_norm) ;
            }
            this->addLayerMatrixInput(domain(0),domain(1),nbr_map);
        }else if(type=="NNLayer::INPUTLINEAR"){
            int domain;
            BasicUtility::String2Any(tool.getAttribute("size"),domain);
            this->addLayerLinearInput(domain);
        }
        else if(type=="NNLayer::MATRIXCONVOLUTIONNAL"){

            std::string str = tool.getAttribute("nbr_map");
            int nbr_map;
            BasicUtility::String2Any(str,nbr_map);

            str = tool.getAttribute("sizekernel");
            int sizekernel;
            BasicUtility::String2Any(str,sizekernel);

            str = tool.getAttribute("subsampling");
            int subsampling;
            BasicUtility::String2Any(str,subsampling);

            this->addLayerMatrixConvolutionSubScaling(nbr_map,subsampling,sizekernel);

            std::string str_biais = tool.getAttribute("weight_biais");
            std::string str_kernel = tool.getAttribute("weight_kernel");
            if(NeuralLayerMatrixConvolutionSubScaling * neural_matrix = dynamic_cast<NeuralLayerMatrixConvolutionSubScaling *>(*(_v_layer.rbegin()))){
                std::istringstream stream_biais(str_biais);
                for(unsigned int index_weight=0;index_weight<neural_matrix->_W_biais.size();index_weight++){
                    F32 weight ;
                    str = pop::BasicUtility::getline( stream_biais, ";" );
                    pop::BasicUtility::String2Any(str,weight );
                    neural_matrix->_W_biais[index_weight]=weight;
                }
                std::istringstream stream_kernel(str_kernel);
                for(unsigned int index_weight=0;index_weight<neural_matrix->_W_kernels.size();index_weight++){
                    for(unsigned int index_weight_j=0;index_weight_j<neural_matrix->_W_kernels(index_weight).size();index_weight_j++){
                        F32 weight ;
                        str = pop::BasicUtility::getline( stream_kernel, ";" );
                        pop::BasicUtility::String2Any(str,weight );
                        neural_matrix->_W_kernels(index_weight)(index_weight_j)=weight;
                    }
                }
            }
        }else if(type=="NNLayer::FULLYCONNECTED"){
            std::string str = tool.getAttribute("size");
            int size;
            BasicUtility::String2Any(str,size);
            this->addLayerLinearFullyConnected(size);
            str = tool.getAttribute("weight");
            if(NeuralLayerLinearFullyConnected * neural_linear = dynamic_cast<NeuralLayerLinearFullyConnected *>(*(_v_layer.rbegin()))){
                std::istringstream stream(str);
                for(unsigned int index_weight=0;index_weight<neural_linear->_W.size();index_weight++){
                    F32 weight ;
                    str = pop::BasicUtility::getline( stream, ";" );
                    pop::BasicUtility::String2Any(str,weight );
                    neural_linear->_W[index_weight] = weight;
                }
            }
        }
    }
}
void NeuralNet::save(const char * file)const
{
    XMLDocument doc;
    XMLNode node1 = doc.addChild("label2String");
    node1.addAttribute("id",BasicUtility::Any2String(_label2string));
    XMLNode node = doc.addChild("layers");
    for(unsigned int i=0;i<this->_v_layer.size();i++){
        NeuralLayer * layer = this->_v_layer[i];
        if(const NeuralLayerMatrixInput *layer_matrix = dynamic_cast<const NeuralLayerMatrixInput *>(layer)){
            XMLNode nodechild = node.addChild("layer");
            nodechild.addAttribute("type","NNLayer::INPUTMATRIX");
            nodechild.addAttribute("size",BasicUtility::Any2String(layer_matrix->X_map()(0).getDomain()));
            nodechild.addAttribute("nbr_map",BasicUtility::Any2String(layer_matrix->X_map().size()));
            nodechild.addAttribute("method",BasicUtility::Any2String(_method));
            nodechild.addAttribute("normalization",BasicUtility::Any2String(_normalization_value));
        }
        else if(const NeuralLayerLinearInput *layer_linear = dynamic_cast<const NeuralLayerLinearInput *>(layer)){

            XMLNode nodechild = node.addChild("layer");
            nodechild.addAttribute("type","NNLayer::INPUTLINEAR");
            nodechild.addAttribute("size",BasicUtility::Any2String(layer_linear->X().size()));
            nodechild.addAttribute("method",BasicUtility::Any2String((_method)));
            nodechild.addAttribute("normalization",BasicUtility::Any2String(_normalization_value));
        }else if(const NeuralLayerMatrixConvolutionSubScaling *layer_conv = dynamic_cast<const NeuralLayerMatrixConvolutionSubScaling *>(layer)){
            XMLNode nodechild = node.addChild("layer");
            nodechild.addAttribute("type","NNLayer::MATRIXCONVOLUTIONNAL");
            nodechild.addAttribute("nbr_map",BasicUtility::Any2String(layer_conv->X_map().size()));
            nodechild.addAttribute("sizekernel",BasicUtility::Any2String(layer_conv->_radius_kernel));
            nodechild.addAttribute("subsampling",BasicUtility::Any2String(layer_conv->_sub_resolution_factor));

            std::string weight_str;
            for(unsigned int index_w=0;index_w<layer_conv->_W_biais.size();index_w++){
                weight_str+=BasicUtility::Any2String(layer_conv->_W_biais[index_w])+";";
            }
            nodechild.addAttribute("weight_biais",weight_str);
            weight_str.clear();
            for(unsigned int index_w=0;index_w<layer_conv->_W_kernels.size();index_w++){
                for(unsigned int index_weight_j=0;index_weight_j<layer_conv->_W_kernels(index_w).size();index_weight_j++){
                    weight_str+=BasicUtility::Any2String(layer_conv->_W_kernels(index_w)(index_weight_j))+";";
                }
            }
            nodechild.addAttribute("weight_kernel",weight_str);

        }
        else if(const NeuralLayerLinearFullyConnected *layer_fully= dynamic_cast<const NeuralLayerLinearFullyConnected *>(layer))
        {
            XMLNode nodechild = node.addChild("layer");
            nodechild.addAttribute("type","NNLayer::FULLYCONNECTED");
            nodechild.addAttribute("size",BasicUtility::Any2String(layer_fully->X().size()));

            std::string weight_str;
            for(unsigned int index_w=0;index_w<layer_fully->_W.size();index_w++){
                weight_str+=BasicUtility::Any2String(layer_fully->_W[index_w])+";";
            }
            nodechild.addAttribute("weight",weight_str);
        }
    }
    doc.save(file);
}
const Vec<std::string>& NeuralNet::label2String()const{
    return _label2string;
}
Vec<std::string>& NeuralNet::label2String(){
    return _label2string;
}
const Vec<NeuralLayer*>& NeuralNet::layers()const{
    return _v_layer;
}
Vec<NeuralLayer*>& NeuralNet::layers(){
    return _v_layer;
}
VecF32 NeuralNet::inputMatrixToInputNeuron(const MatN<2,UI8>  & matrix){
    if(NeuralLayerMatrix* layer_matrix = dynamic_cast<NeuralLayerMatrix *>(this->_v_layer(0))){
        Vec2I32 domain = layer_matrix->X_map()(0).getDomain();
        return NNLayerMatrix::inputMatrixToInputNeuron(matrix,domain,this->_method,this->_normalization_value);
    }else{
        std::cerr<<"No matrixlayer  for neural network"<<std::endl;
        return VecF32();
    }
}
}
