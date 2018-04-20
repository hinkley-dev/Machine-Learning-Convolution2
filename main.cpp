// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <exception>
#include <string>
#include <memory>
#include "error.h"
#include "string.h"
#include "rand.h"
#include "matrix.h"
#include "supervised.h"
#include "baseline.h"
#include "layer.h"
#include "layerlinear.h"
#include "layerconv.h"
#include "layerleakyrect.h"
#include "layermaxpooling2d.h"
#include "layersinusoid.h"
#include "neuralnet.h"
#include <algorithm>
#include "imputer.h"
#include "nomcat.h"
#include "normalizer.h"
#include "svg.h"
#include <fstream>
#include <time.h>
#include "matrix.h"

using std::cout;
using std::cerr;
using std::string;
using std::auto_ptr;



void testLearner(SupervisedLearner& learner)
{


}


size_t convertToDecimal(const Vec& oneHot)
{
	return oneHot.indexOfMax();
}


void make_features_and_labels(const Matrix& data, Matrix& feats, Matrix& labs)
{

}

void make_training_and_testing(Matrix& trainFeats, Matrix& trainLabs, Matrix& testFeats, Matrix& testLabs, Matrix& data)
{
	size_t trainingDataCount = 256;
	size_t testingDataCount = 100;

	trainFeats.setSize(trainingDataCount, 1);
	trainLabs.setSize(trainingDataCount, 1);
	testFeats.setSize(testingDataCount, 1);
	testLabs.setSize(testingDataCount, 1);

	for(size_t i = 0; i < trainingDataCount; ++i)
	{
		trainFeats[i][0] = (i / (double)trainingDataCount);
		trainLabs[i][0] = data[i][0];

	}
	for(size_t i = 0; i < testingDataCount; ++i)
	{
		testFeats[i][0] = ((i + trainingDataCount) / (double)trainingDataCount);
		testLabs[i][0] = data[trainingDataCount +i][0];
	}



}

void preprocessData(Matrix& m)
{


}

void shuffleTrainData(Matrix& trainFeats, Matrix& trainLabs, Rand random)
{
	if(trainLabs.rows() != trainFeats.rows()) throw Ex("training data not intiialized correctly");
	for(size_t i = 0; i < trainFeats.rows(); ++i)
	{
		size_t r1 = random.next(trainFeats.rows());
		size_t r2 = random.next(trainFeats.rows());
		trainFeats.swapRows(r1,r2);
		trainLabs.swapRows(r1,r2);
	}
}

void testTimeseries(Rand random)
{



	string fn = "data/";
	Matrix timeseries_data;
	timeseries_data.loadARFF(fn + "timeseries.arff");
	Matrix trainFeats, trainLabs, testFeats, testLabs;

	double x_max  =1.5;
	double y_max = 15.0;
	GSVG svg(1024, 768,0.0, 0.0,x_max, y_max, 200);

	svg.horizMarks(x_max*10);
	svg.vertMarks(y_max);

	make_training_and_testing(trainFeats, trainLabs, testFeats, testLabs, timeseries_data);
	//shuffle testingDataCount

for(size_t reg = 0; reg < 3; ++reg)
{
			NeuralNet nn(random);
			nn.add(new LayerLinear(1,101));
			nn.add(new LayerSinusoid(101));
			nn.add(new LayerLinear(101,1));
			nn.init_weights();
			nn.setReg(reg);

			double lr = 0.000005;
			//lr 0.000005 best for noReg with average error = 20.2709


			for(size_t j = 0; j < 3; ++j)
			{
				shuffleTrainData(trainFeats, trainLabs, random);
				for(size_t i = 0; i < trainFeats.rows(); ++i)
				{
					Vec prediction = nn.predict(trainFeats.row(i));
					nn.backprop(trainLabs.row(i));
					nn.update_gradient(trainFeats.row(i));
					nn.refine_weights(lr);
					//if(i%10==0)lr*=0.9;
				}
			}
			shuffleTrainData(trainFeats, trainLabs, random);

			double averageError = 0.0;
			for(size_t i = 0; i < trainFeats.rows(); ++i)
			{
				if(reg == 0)
					svg.dot(trainFeats.row(i)[0], trainLabs.row(i)[0],0.5, 0x00ff00); //green for official
				Vec prediction = nn.predict(trainFeats.row(i));
				double er = (trainLabs.row(i)[0] - prediction[0]) / trainLabs.row(i)[0];
				averageError += abs(er);
			}

			 unsigned int predictionColor = 0x8B008B;
			 if(reg == 0) predictionColor = 0x8B008B; //purple
			 if(reg == 1) predictionColor = 0x0000ff; //blue
			 if(reg == 2) predictionColor = 0xffa500; //orange

			for(size_t i = 0; i < testFeats.rows(); ++i)
			{
				if(reg ==0)
					svg.dot(testFeats.row(i)[0], testLabs.row(i)[0],0.5, 0x00ff00); //green for official
				Vec prediction = nn.predict(testFeats.row(i));
				svg.dot(testFeats.row(i)[0],prediction[0],0.5, predictionColor); // for predictions it was not trained on
				double er = (testLabs.row(i)[0] - prediction[0]) / testLabs.row(i)[0];
				averageError+= abs(er);
			}
			averageError = averageError*100 / (testFeats.rows() + trainFeats.rows());
			cout << "average error: " << averageError << endl;
			cout << "weight val: " << nn.get_weights().sum() << endl;
}
svg.line(1.0,0.0,1.0,y_max,1.5, 0xff0000 ); //red line where it starts to predict
svg.text(1.05, y_max*0.9, "Testing");
svg.text(0.8, y_max*0.9, "Training");

//labels
svg.line(0.2,y_max*0.95,0.3,y_max*0.95,2.0, 0x00ff00 ); //green
svg.line(0.2,y_max*0.85,0.3,y_max*0.85,2.0, 0x8B008B ); //purple line where it starts to predict
svg.line(0.2,y_max*0.75,0.3,y_max*0.75,2.0, 0x0000ff ); //blue line where it starts to predict
svg.line(0.2,y_max*0.65,0.3,y_max*0.65,2.0, 0xffa500 ); //orange line where it starts to predict the future

svg.text(0.35, y_max*0.95, "Given Data");
svg.text(0.35, y_max*0.85, "No Regularization");
svg.text(0.35, y_max*0.75, "L1 Regularization");
svg.text(0.35, y_max*0.65, "L2 Regularization");


double y_label_pos_x_axis = svg.horizLabelPos();

double x_label_pos_y_axis = svg.vertLabelPos();
svg.text(x_max / 2, y_label_pos_x_axis*1.3, "Timeseries data index / 256");
svg.text(x_label_pos_y_axis*2, y_max / 2, "Unemplyment rate");

//
std::ofstream s;
s.exceptions(std::ios::badbit);
s.open("chart.svg", std::ios::binary);
svg.print(s);
  //




}

void debugSpew(Rand random)
{

}



int main(int argc, char *argv[])
{
	Rand random(123);
	enableFloatingPointExceptions();
	int ret = 1;


	try
	{

		ret = 0;
	}
	catch(const std::exception& e)
	{
		cerr << "An error occurred: " << e.what() << "\n";
	}
	try
	{
		testTimeseries(random);





		ret = 0;
	}
	catch(const std::exception& e)
	{
		cerr << "An error occurred: " << e.what() << "\n";
	}
	cout.flush();
	cerr.flush();
	return ret;
}
