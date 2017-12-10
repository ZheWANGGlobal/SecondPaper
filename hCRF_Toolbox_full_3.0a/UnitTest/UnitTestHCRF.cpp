// UnitTestHCRF.cpp : Defines the entry point for the console application.
//

#include <math.h>
#include "hCRF.h"
#include "gtest/gtest.h"

namespace {

   TEST(TestMatrix, MatrixInit)
   {
	  dMatrix a (5,10);
	  EXPECT_EQ(a.getWidth(), 5);
	  EXPECT_EQ(a.getHeight(), 10);
   }

   TEST(TestMatrix, l1Norm)
   {
	  dMatrix a(1,3);
	  a(0, 0) = 1;
	  a(1, 0) = -2;
	  a(2, 0) = 3;
	  EXPECT_EQ(a.l1Norm(), 6);
	  EXPECT_EQ(a.l2Norm(), sqrt(double(14)));
	  EXPECT_EQ(a.l2Norm(false), 14);
	  EXPECT_EQ(a.l2Norm(true), sqrt(double(14)));
   }

   TEST(TestMatrix, MatrixEquality)
   {
	  dMatrix a (10,10,0);
	  dMatrix b(10, 10, 0);
	  EXPECT_EQ(a, b);
	  b(5,5) = 1;
	  EXPECT_NE(a, b);
	  dMatrix c(9,10, 0);
	  EXPECT_NE(a, c);
	  c = dMatrix(10,9,0);
	  EXPECT_NE(a, c);
   }

   TEST(TestMatrix, InPlaceAddition)
   {
	  dMatrix a(10, 10, 0);
	  a.getValue(0,0) += 1;
	  EXPECT_EQ(a.getValue(0,0), 1);
	  a(0,0) += 1;
	  EXPECT_EQ(a(0,0), 2);
	  iVector b(10, COLVECTOR, 2);
	  EXPECT_EQ(b[0], 2);
	  b[1] += 2;
	  EXPECT_EQ(b[1], 4);
   }

   TEST(TestMatrix, MatrixAddition)
   {
	  dMatrix a(2, 2, 0);
	  dMatrix b(2, 2, 0);
	  double  temp[4]  = {1, 2, 3, 4};
	  a.set(temp, 2, 2);
	  EXPECT_EQ(a + b, a);
	  double temp2[4] = {4, 2, 7, 6};
	  b.set(temp2, 2, 2);
	  EXPECT_EQ(a+b, b+a) << "Addition fail to  commute";
	  dMatrix c(2,2,0);
	  c = a+b;
	  EXPECT_EQ(c, a+b) << "c= a+b not working";
   }


}  // namespace

int main(int argc, char **argv) {
   dMatrix a(1,3);
	a(0,0) = 1;
	a(1,0) = -2;
	a(2,0) = 3;
	EXPECT_EQ(a.l1Norm(), 6);
	EXPECT_EQ(a.l2Norm(), sqrt(double(14)));
	EXPECT_EQ(a.l2Norm(false), 14);
	EXPECT_EQ(a.l2Norm(true), sqrt(double(14)));
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
