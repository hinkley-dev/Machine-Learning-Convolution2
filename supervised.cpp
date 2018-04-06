#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "supervised.h"
#include "error.h"
#include "string.h"
#include "rand.h"
#include "vec.h"
#include <math.h>

using std::vector;
using namespace std;

// virtual
void SupervisedLearner::filter_data(const Matrix& feat_in, const Matrix& lab_in, Matrix& feat_out, Matrix& lab_out)
{
	feat_out.copy(feat_in);
	lab_out.copy(lab_in);
}

size_t SupervisedLearner::countMisclassifications(const Matrix& features, const Matrix& labels)
{
	if(features.rows() != labels.rows())
		throw Ex("Mismatching number of rows");
	size_t mis = 0;
	for(size_t i = 0; i < features.rows(); i++)
	{
		const Vec& pred = predict(features[i]);
		const Vec& lab = labels[i];
		for(size_t j = 0; j < lab.size(); j++)
		{
			if(pred[j] != lab[j])
			{
				mis++;
			}
		}
	}
	return mis;
}

// virtual
void SupervisedLearner::trainIncremental(const Vec& feat, const Vec& lab)
{
	throw Ex("Sorry, this learner does not support incremental training");
}

float SupervisedLearner::sum_squared_error(Matrix& features, Matrix& labels)
{
	if(features.rows() != labels.rows())
		throw Ex("Mismatching number of rows");

	float sse = 0;
	for(size_t i = 0; i < features.rows(); i++)
	{

		const Vec& pred = predict(features[i]);

		const Vec& lab = labels[i];

		for(size_t j = 0; j < lab.size(); j++)
		{
			sse+= ((pred[j] - lab[j])*(pred[j] - lab[j]));
		}
	}
	return sse;
}



//folds means you take 1/folds of the data for testing and the rest for training
void SupervisedLearner::crossValidation(Matrix& feat, Matrix& lab,size_t repetitions, size_t folds)
{


	size_t testRowCount = lab.rows()/folds;
	size_t trainRowCount = lab.rows() - testRowCount;




	float sse = 0;
	float mse = 0;
	for(size_t i = 0; i < repetitions; ++i)
	{

		for(size_t j = 0; j < folds; ++j)
		{

			//start building matricies
			size_t startRow = j*testRowCount;


			Matrix testLabs(testRowCount, lab.cols());
			testLabs.copyBlock(0,0, lab, startRow, 0, testRowCount, lab.cols());

			Matrix testFeats(testRowCount, feat.cols());
			testFeats.copyBlock(0,0, feat, startRow, 0, testRowCount, feat.cols());


			Matrix trainLabs(trainRowCount, lab.cols());
			trainLabs.copyBlock(0, 0, lab, 0,0,startRow,lab.cols());

			//check to make sure there needs to be a second matrix
			if((startRow + testRowCount) != feat.rows())
			{
				trainLabs.copyBlock(startRow, 0, lab, startRow + testRowCount, 0,trainRowCount - startRow, lab.cols());
			}

			Matrix trainFeats(trainRowCount, feat.cols());
			trainFeats.copyBlock(0, 0, feat, 0,0,startRow,feat.cols());

			//check to make sure there needs to be a second matrix
			if((startRow + testRowCount) != feat.rows())
			{
				trainFeats.copyBlock(startRow, 0, feat, startRow + testRowCount, 0,trainRowCount - startRow, feat.cols());
			}
			//end of building Matricies

			train(trainFeats, trainLabs);

			sse += sum_squared_error(testFeats, testLabs);


		}
		mse += sse /(feat.rows());
		sse = 0;

		Rand r(123);
		srand(time(NULL));
		for(size_t k = 0; k < feat.rows(); ++k)
		{
			int firstRow =rand() % feat.rows();
			int secondRow = rand() % feat.rows();
			feat.swapRows(firstRow, secondRow);
			lab.swapRows(firstRow, secondRow);
		}
	}

	float rmse = sqrt(mse / repetitions);
	std::cout << rmse << std::endl;

}
