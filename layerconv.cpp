#include "layerconv.h"

LayerConv::LayerConv(const Vec& inputDims,
                     const Vec& filterDims,
                     const Vec& outputDims)
: Layer(Layer::countTensorElements(inputDims), Layer::countTensorElements(outputDims))
{
  m_inputs = Layer::countTensorElements(inputDims);
  m_inputDims.copy(inputDims);
  m_filterDims.copy(filterDims);
  m_outputDims.copy(outputDims);
  weightCount = 1;
  for(size_t i = 0; i < m_filterDims.size() -1; ++i)
  {
    weightCount *= (m_filterDims)[i];
  }
  weightCount++;
  weightCount *= (m_filterDims)[filterDims.size() -1];

}

LayerConv::~LayerConv()
{

}

size_t LayerConv::getFilterSize()
{
  return Layer::countTensorElements(m_filterDims);
}


void LayerConv::activate(const Vec& weights,const Vec& x)
{
  if(x.size() != m_inputs)
  {
     cout << "input size: " << x.size() << ",  m_inputs set val: " << m_inputs << endl;
     throw Ex("input values not matching for conv activate...");
  }
  Tensor input(*(Vec*)&x, m_inputDims);


  size_t weightsPerFilter = 1;
  for(size_t i = 0; i < m_filterDims.size() -1; ++i)
  {
    weightsPerFilter *= m_filterDims[i];
  }

  size_t weightIndex = 0;
  size_t filterCount = m_filterDims[m_filterDims.size()-1];
  VecWrapper v_filterDims(m_filterDims, 0, m_filterDims.size()-1);
  for(size_t filters = 0; filters < filterCount; ++filters)
  {
    double bias = weights[weightIndex];

    weightIndex++;
    VecWrapper v_filter(*(Vec*)&weights, weightIndex, weightsPerFilter);
    weightIndex += weightsPerFilter;

    Tensor filter(v_filter, v_filterDims);
    Vec v_out(activation.size() / filterCount);
    v_out.fill(0.0);
    VecWrapper v_outputDims(m_outputDims, 0, m_outputDims.size()-1);
    Tensor out(v_out, v_outputDims);

    Tensor::convolve(input, filter, out);
    out += bias;

    activation.put(filters * v_out.size(), out);
  }



}

void LayerConv::backprop(const Vec& weights, Vec& prevBlame)
{

  size_t weightsPerFilter = 1;
  for(size_t i = 0; i < m_filterDims.size() -1; ++i)
  {
    weightsPerFilter *= m_filterDims[i];
  }

  size_t weightIndex = 0;
  size_t blameIndex = 0;
  size_t filterCount = m_filterDims[m_filterDims.size()-1];
  VecWrapper v_filterDims(m_filterDims, 0, m_filterDims.size()-1);
  Vec temp(prevBlame.size());
  temp.fill(0.0);

  Tensor t_prevBlame(temp, m_inputDims);
  t_prevBlame.fill(0.0);

  for(size_t filters = 0; filters < filterCount; ++filters)
  {


    weightIndex++;
    VecWrapper v_filter(*(Vec*)&weights, weightIndex, weightsPerFilter);
    weightIndex += weightsPerFilter;

    Tensor filter(v_filter, v_filterDims);

    VecWrapper v_outputDims(m_outputDims, 0, m_outputDims.size()-1);
    VecWrapper v_blame(blame, blameIndex, blame.size() / filterCount);
    blameIndex += blame.size() / filterCount;
    Tensor t_blame(v_blame, v_outputDims);


    Tensor::convolve(t_blame, filter, t_prevBlame, true,1);
  }
  prevBlame.copy(t_prevBlame);

}



size_t LayerConv::getInputCount()
{
  return m_inputs;
}

void LayerConv::update_gradient(const Vec& x, Vec& gradient)
{

  Tensor input(*(Vec*)&x, m_inputDims);


  size_t gradientsPerFilter = 1;
  for(size_t i = 0; i < m_filterDims.size() -1; ++i)
  {
    gradientsPerFilter *= m_filterDims[i];
  }

  size_t blameIndex = 0;
  size_t gradientIndex = 0;
  size_t filterCount = m_filterDims[m_filterDims.size()-1];
  VecWrapper v_filterDims(m_filterDims, 0, m_filterDims.size()-1);
  for(size_t filters = 0; filters < filterCount; ++filters)
  {
    size_t gradientPlacer = gradientIndex;
    VecWrapper v_outputDims(m_outputDims, 0, m_outputDims.size()-1);
    VecWrapper v_blame(blame, blameIndex, blame.size() / filterCount);
    blameIndex += blame.size() / filterCount;
    Tensor t_blame(v_blame, v_outputDims);


    gradient[gradientIndex] = v_blame.sum();

    gradientIndex++;
    VecWrapper v_gradient(*(Vec*)&gradient, gradientIndex, gradientsPerFilter);
    gradientIndex += gradientsPerFilter;
    Tensor t_gradient(v_gradient, v_filterDims);







    Tensor::convolve(input, t_blame, t_gradient);
    gradient.put(gradientPlacer + 1, t_gradient);
  }


}
