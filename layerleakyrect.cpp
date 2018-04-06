#include "layerleakyrect.h"

LayerLeakyRectifier::LayerLeakyRectifier(size_t inputs) :
 Layer(inputs)
{
  m_inputs = inputs;
  weightCount = 0;
}

LayerLeakyRectifier::~LayerLeakyRectifier()
{

}

void LayerLeakyRectifier::activate(const Vec& weights,const Vec& x)
{
  for(size_t i = 0; i < activation.size(); i++)
	{
		if(x[i] < 0.0)
			activation[i] = 0.01*x[i];
		else activation[i] = x[i];
	}

}

void LayerLeakyRectifier::backprop(const Vec& weights, Vec& prevBlame)
{

  for(size_t i = 0; i < activation.size(); ++i)
  {
    if(activation[i] < 0)
      prevBlame[i] = blame[i] * 0.01;
    else
      prevBlame[i] = blame[i];
  }


}



size_t LayerLeakyRectifier::getInputCount()
{
  return m_inputs;
}

void LayerLeakyRectifier::update_gradient(const Vec& x, Vec& gradient)
{

}
