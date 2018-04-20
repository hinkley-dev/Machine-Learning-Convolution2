#include "layersinusoid.h"

LayerSinusoid::LayerSinusoid(size_t inputs) :
 Layer(inputs)
{
  m_inputs = inputs;
  weightCount = 0;
}

LayerSinusoid::~LayerSinusoid()
{

}

void LayerSinusoid::activate(const Vec& weights,const Vec& x)
{
  in.copy(x);
  for(size_t i = 0; i < activation.size(); i++)
	{
    activation[i] = sin(x[i]);
	}
  activation[activation.size()-1] = x[activation.size()-1];


}

void LayerSinusoid::backprop(const Vec& weights, Vec& prevBlame)
{

  for(size_t i = 0; i < prevBlame.size(); ++i)
  {
    prevBlame[i] = blame[i]*cos(in[i]);
  }


}



size_t LayerSinusoid::getInputCount()
{
  return m_inputs;
}

void LayerSinusoid::update_gradient(const Vec& x, Vec& gradient)
{

}
