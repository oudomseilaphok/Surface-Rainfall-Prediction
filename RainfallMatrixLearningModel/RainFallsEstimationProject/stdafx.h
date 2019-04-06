// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"
#include "awscompiler.h"
#include "radarcompiler.h"
#include <stdio.h>
#include <tchar.h>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <map>

class UtilityComponent {
public:
	// Seila: check to filter out false data of AWS
	bool isFloat(std::string data);

	// Seila : extract hex color code from the RWS Image Pixel
	std::string pixelToHex(cv::Mat image, int x, int y);

	double pearsoncoeff(std::vector<double> X, std::vector<double> Y);

	std::vector<double> timeSeriesIncrement(std::vector<double> input);

	int mostFrequentElement(std::vector<int> arr);

	std::vector<double> extractAWSLocation(double targetLong, double targetLat);

	std::map<int, double>  getScenarioPercentage(double windSpeed, std::string caseScenario);

	double averageAngle(std::vector<double> anglesList);
};


// TODO: reference additional headers your program requires here
