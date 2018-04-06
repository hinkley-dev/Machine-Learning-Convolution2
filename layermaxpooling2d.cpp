#include "layermaxpooling2d.h"

LayerMaxPooling2D::LayerMaxPooling2D(size_t filterDim1, size_t filterDim2, size_t filterCount) :
 Layer(filterDim1*filterDim2*filterCount, filterCount*(filterDim1/2)*(filterDim2/2))
{
  if(filterDim1 != filterDim2) throw Ex("Layer Max pooling requires square dims");

  m_inputs = filterDim1*filterDim2*filterCount;
  m_inputDim = filterDim1;

  m_layerCount = filterCount;
  weightCount = 0;

  maxIndicies.resize(activation.size());

}

LayerMaxPooling2D::~LayerMaxPooling2D()
{

}

void LayerMaxPooling2D::activate(const Vec& weights,const Vec& x)
{
  size_t firstItr = 0;
  size_t secondItr = m_inputDim;
  size_t activationIndex = 0;
  for(size_t i = 0; i < m_layerCount; ++i)
  {
    for(size_t j = 0; j < activation.size() / m_layerCount; ++j)
    {
      maxIndicies[activationIndex] = maxIndexOfPool(x,firstItr, firstItr+1, secondItr, secondItr+1);
      activation[activationIndex] = x[maxIndicies[activationIndex]];
      firstItr++;
      secondItr++;
       if((firstItr + 1) % m_inputDim == 0)
       {
         firstItr = secondItr + 1;
         secondItr += m_inputDim + 1;
       }
       else
       {
         firstItr++;
         secondItr++;
       }
       activationIndex++;
    }
  }


}



void LayerMaxPooling2D::backprop(const Vec& weights, Vec& prevBlame)
{
  for(size_t i = 0; i < blame.size(); ++i)
    prevBlame[maxIndicies[i]] = blame[i];
}



size_t LayerMaxPooling2D::getInputCount()
{
  return m_inputs;
}

void LayerMaxPooling2D::update_gradient(const Vec& x, Vec& gradient)
{

}

size_t LayerMaxPooling2D::maxIndexOfPool(const Vec& x, size_t a, size_t b, size_t c, size_t d)
{
  size_t indexOfMax = a;
  if(x[b] > x[indexOfMax])
    indexOfMax = b;
  if(x[c] > x[indexOfMax])
    indexOfMax = c;
  if(x[d] > x[indexOfMax])
    indexOfMax = d;

  return indexOfMax;
}
