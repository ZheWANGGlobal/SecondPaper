#include "hCRF.h"
#include "gtest/gtest.h"
/**
This tests all the optimiser, to check for memory leak and other.  Should be run
using Valgrind or similar tool. It use the ToolboxLDCRF to do is work
**/

class OptimizerTest: public ::testing::Test{
protected:
	virtual void SetUp(){
		dataTrain.load("data/train_data.hcr", "data/train_start_end.hcr");
		dataTest.load("data/test_data.hcr", "data/test_start_end.hcr");
	}
	DataSet dataTrain, dataTest;
};

namespace{
	TEST_F(OptimizerTest, LBFGS)
	{
		ToolboxLDCRF toolbox = ToolboxLDCRF(3, OPTIMIZER_LBFGS);
		toolbox.setRegularizationL2(1.0);
		toolbox.train(dataTrain);
		toolbox.test(dataTest);
	}

	TEST_F(OptimizerTest, OWLQN)
	{
		ToolboxLDCRF toolbox = ToolboxLDCRF(3, OPTIMIZER_OWLQN);
		toolbox.setRegularizationL2(1.0);
		toolbox.train(dataTrain);
		toolbox.test(dataTest);
	}
	TEST_F(OptimizerTest, CG)
	{
		ToolboxLDCRF toolbox = ToolboxLDCRF(3, OPTIMIZER_CG);
		toolbox.train(dataTrain);
		toolbox.test(dataTest);
	}
	TEST_F(OptimizerTest, ASA)
	{
		ToolboxLDCRF toolbox = ToolboxLDCRF(3, OPTIMIZER_ASA);
		toolbox.setRegularizationL2(1.0);
		toolbox.train(dataTrain);
		toolbox.test(dataTest);
	}
	TEST_F(OptimizerTest, BFGS)
	{
		ToolboxLDCRF toolbox = ToolboxLDCRF(3, OPTIMIZER_BFGS);
		toolbox.setRegularizationL2(1.0);
		toolbox.train(dataTrain);
		toolbox.test(dataTest);
	}
}

