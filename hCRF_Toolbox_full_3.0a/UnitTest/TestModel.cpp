#include "hCRF.h"
#include "gtest/gtest.h"

namespace{
	TEST(Model, InitModel)
	{
		// Test that the default parameter are set correctly
		Model a = Model();
		EXPECT_EQ(a.getNumberOfSequenceLabels(), 0);
		EXPECT_EQ(a.getNumberOfStateLabels(), 0);
		EXPECT_EQ(a.getNumberOfStates(), 0);
	}
}
