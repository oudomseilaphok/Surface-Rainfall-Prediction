// stdafx.cpp : source file that includes just the standard includes
// RainFallsEstimationProject.pch will be the pre-compiled header
// stdafx.obj will contain the pre-compiled type information

#include "stdafx.h"
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>                                                           
#include <numeric>         

// TODO: reference any additional headers you need in STDAFX.H
// and not in this file

// Seila :: Finding Most Frequent Element in Vector Array
int UtilityComponent::mostFrequentElement(std::vector<int> arr) {
	// Sort the array 
	std::sort(arr.begin(), arr.end());
	int n = arr.size();

	// find the max frequency using linear traversal 
	int max_count = 1, res = arr[0], curr_count = 1;
	for (int i = 1; i < n; i++) {
		if (arr[i] == arr[i - 1])
			curr_count++;
		else {
			if (curr_count > max_count) {
				max_count = curr_count;
				res = arr[i - 1];
			}
			curr_count = 1;
		}
	}

	// If last element is most frequent 
	if (curr_count > max_count)
	{
		max_count = curr_count;
		res = arr[n - 1];
	}

	return res;
}

// This function is used to check the data is string or float
// In our Experiment it is used to clean some raw data that is empty or is not desirable
// Seila: check to filter out false data of AWS
bool UtilityComponent::isFloat(std::string myString) {
	std::istringstream iss(myString);
	float f;
	iss >> std::noskipws >> f; // noskipws considers leading whitespace invalid
						  // Check the entire string was consumed and if either failbit or badbit is set
	return iss.eof() && !iss.fail();
}

// Seila : extract RGB -> Hex color code from the RWS Image Pixel
std::string UtilityComponent::pixelToHex(cv::Mat image, int x, int y)
{
	std::string result;

	int R = image.at<cv::Vec3b>(cv::Point(x, y))[0];
	int G = image.at<cv::Vec3b>(cv::Point(x, y))[1];
	int B = image.at<cv::Vec3b>(cv::Point(x, y))[2];

	char r[255];
	sprintf_s(r, "%.2X", R);
	result.append(r);

	char g[255];
	sprintf_s(g, "%.2X", G);
	result.append(g);

	char b[255];
	sprintf_s(b, "%.2X", B);
	result.append(b);

	return result;
}

// correlation function
double sum(std::vector<double> a)
{
	double s = 0;
	for (int i = 0; i < a.size(); i++)
	{
		s += a[i];
	}
	return s;
}

double mean(std::vector<double> a)
{
	return sum(a) / a.size();
}

double sqsum(std::vector<double> a)
{
	double s = 0;
	for (int i = 0; i < a.size(); i++)
	{
		s += pow(a[i], 2);
	}
	return s;
}

double stdev(std::vector<double> nums)
{
	double N = nums.size();
	return pow(sqsum(nums) / N - pow(sum(nums) / N, 2), 0.5);
}

std::vector<double> operator-(std::vector<double> a, double b)
{
	std::vector<double> retvect;
	for (int i = 0; i < a.size(); i++)
	{
		retvect.push_back(a[i] - b);
	}
	return retvect;
}


std::vector<double> operator*(std::vector<double> a, std::vector<double> b)
{
	std::vector<double> retvect;
	for (int i = 0; i < a.size(); i++)
	{
		retvect.push_back(a[i] * b[i]);
	}
	return retvect;
}

//Seila :: This functions get Pearson coefficient correlation between two series of data
double UtilityComponent::pearsoncoeff(std::vector<double> X, std::vector<double> Y)
{
	return sum((X - mean(X))*(Y - mean(Y))) / (X.size()*stdev(X)* stdev(Y));
}

//Seila :: This functions get Pearson coefficient correlation between two series of data
std::vector<double> UtilityComponent::timeSeriesIncrement(std::vector<double> input)
{
	std::vector<double> output;
	int n = input.size();
	for (int i = 0; i < input.size(); i++) {
		for (int j = 0; j < n; j++) {
			output.push_back(input[i]);
			//std::cout << input[i] << std::endl;
		}
		n = n - 1;
		//std::cout << "n = " << n << std::endl;
	}
	return output;
}

std::map<int, double>  UtilityComponent::getScenarioPercentage(double windSpeed, std::string caseScenario) {
	std::map<int, double> output = {
						{ 4, 0.2 },
						{ 5, 0.1 },
						{ 7, 0.5 },
						{ 8, 0.2 }
	};
	
	return output;
}
// Seila :: this functions extract the position X,Y from Longitude and Latitude of Radar Image 
// Radar Image Type : "RDR_CMI_201701010000", 526 * 576 pixels
std::vector<double> UtilityComponent::extractAWSLocation(double targetLong, double targetLat) {
	std::vector<double> XY;
	static double  PI, DEGRAD, RADDEG;
	static double  re, olon, olat, sn, sf, ro;
	double         slat1, slat2, alon, alat, xn, yn, ra, theta;
	//double			targetLong = 128.37762; // AWS 560
	//double			targetLat = 37.56197; // AWS 560

	PI = asin(1.0) * 2.0;
	DEGRAD = PI / 180.0;
	RADDEG = 180.0 / PI;

	re = 6370.19584 / 2.0;
	slat1 = 30 * DEGRAD;
	slat2 = 60 * DEGRAD;
	olon = 126.3096 * DEGRAD;
	olat = 34.4274 * DEGRAD;

	sn = tan(PI*0.25 + slat2 * 0.5) / tan(PI*0.25 + slat1 * 0.5);
	sn = log(cos(slat1) / cos(slat2)) / log(sn);
	sf = tan(PI*0.25 + slat1 * 0.5);
	sf = pow(sf, sn)*cos(slat1) / sn;
	ro = tan(PI*0.25 + olat * 0.5);
	ro = re * sf / pow(ro, sn);

	ra = tan(PI*0.25 + (targetLat)*DEGRAD*0.5);
	ra = re * sf / pow(ra, sn);
	theta = (targetLong)*DEGRAD - olon;
	if (theta > PI) theta -= 2.0*PI;
	if (theta < -PI) theta += 2.0*PI;
	theta *= sn;
	XY.push_back((float)(ra*sin(theta)) + 229);
	XY.push_back(576 - ((float)(ro - ra * cos(theta)) + 204));
	//std::cout << "X = " << (float)(ra*sin(theta)) + 229 << std::endl;
	//std::cout << "Y = " << 576 - ((float)(ro - ra * cos(theta)) + 204) << std::endl;
	return XY;
}
// end

double UtilityComponent::averageAngle(std::vector<double> angleList)
{
	double y_part = 0, x_part = 0;
	int i;
	int size = angleList.size();
	for (i = 0; i < size; i++)
	{
		x_part += cos(angleList[i] * M_PI / 180);
		y_part += sin(angleList[i] * M_PI / 180);
	}

	return 360 + (atan2(y_part / size, x_part / size) * 180 / M_PI);
	//return std::accumulate(angleList.begin(), angleList.end(), 0.0) / angleList.size();
}


