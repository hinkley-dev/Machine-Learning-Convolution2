#include "layertanh.h"

LayerTanh::LayerTanh(size_t inputs) :
 Layer(inputs)
{
  m_inputs = inputs;
  weightCount = 0;
}

LayerTanh::~LayerTanh()
{

}

void LayerTanh::activate(const Vec& weights,const Vec& x)
{
  for(size_t i = 0; i < activation.size(); i++)
	{
		if(x[i] >= 700.0)
			activation[i] = 1.0;
		else if(x[i] < -700.0)
			activation[i] = -1.0;
		else activation[i] = tanh(x[i]);
	}


}

void LayerTanh::backprop(const Vec& weights, Vec& prevBlame)
{

  for(size_t i = 0; i < activation.size(); ++i)
  {
    prevBlame[i] = blame[i] * (1.0 - (activation[i] * activation[i]));
  }


}



size_t LayerTanh::getInputCount()
{
  return m_inputs;
}

void LayerTanh::update_gradient(const Vec& x, Vec& gradient)
{

}
