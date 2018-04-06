#include "layer.h"

Layer::Layer(size_t inputs, size_t outputs) :
  activation(outputs), blame(outputs)
{
  activation.fill(0.0);
  blame.fill(0.0);
}

Layer::Layer(size_t inputs) :
  activation(inputs), blame(inputs)
{
  activation.fill(0.0);
  blame.fill(0.0);
}


Layer::~Layer()
{

}

size_t Layer::countTensorElements(const Vec& dims)
{
	size_t n = 1;
	for(size_t i = 0; i < dims.size(); ++i)
		n *= dims[i];
	return n;
}

const Vec& Layer::getActivation()
{
  return activation;
}

void Layer::setBlame(const Vec& _blame)
{
  blame.copy(_blame);
}

size_t Layer::getWeightCount()
{
  return weightCount;
}
