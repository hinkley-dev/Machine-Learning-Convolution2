#include "layer.h"
#include "layerlinear.h"
#include "matrix.h"
#include "vec.h"

#include <cmath>

LayerLinear::LayerLinear(size_t inputs, size_t outputs) :
 Layer(inputs, outputs)
{
  m_inputs = inputs;
  m_outputs = outputs;
  weightCount = m_outputs * m_inputs + m_outputs;
}

LayerLinear::~LayerLinear()
{

}

size_t LayerLinear::getInputCount()
{
  return m_inputs;
}

void LayerLinear::activate(const Vec& weights,const Vec& x)
{
  //b will be first "outputs" values in weights
  VecWrapper b(*(Vec*)&weights, 0, m_outputs);
  size_t matrixIndex = m_outputs;

  for(size_t i = 0; i < m_outputs; ++i)
  {
    VecWrapper weightsRow(*(Vec*)&weights, matrixIndex, m_inputs);
    // cout << "weights row: " << endl;
    // weightsRow.print();
    // cout << endl;
    // cout << "b[i]: " << b[i] << endl;
    activation[i] = x.dotProduct(weightsRow) + b[i];
    matrixIndex +=weightsRow.size();
  }
}


//the first blame will be y - y_hat


void LayerLinear::backprop(const Vec& weights, Vec& prevBlame)
{

  //making M
  Matrix M(m_outputs, m_inputs);
  size_t matrixIndex = m_outputs -1;

  for(size_t i = 0; i < m_outputs; ++i)
  {
    for(size_t j = 0; j < m_inputs; ++j)
    {
      matrixIndex++;
      M[i][j] = weights[matrixIndex];
    }
  }

  //calculate prevBlame
  Matrix Blame = Matrix(blame);

  Matrix* M_transposed = M.transpose();
  for(size_t i = 0; i < prevBlame.size(); ++i)
  {
    double val = blame.dotProduct(M_transposed->row(i));
    prevBlame[i] +=  val;
  }

  delete M_transposed;


}


void LayerLinear::update_gradient(const Vec& x, Vec& gradient)
{


  //updating b values
  for(size_t i = 0; i < m_outputs; ++i)
  {
    gradient[i] += blame[i];
  }


 size_t matrixIndex = m_outputs;

 //updating matrix values
 for(size_t i = 0; i < m_outputs; ++i)
 {
   for(size_t j = 0; j < m_inputs; ++j)
   {
     gradient[matrixIndex]  += (blame[i] * x[j]);
     matrixIndex++;
   }
 }



}




















//hello
