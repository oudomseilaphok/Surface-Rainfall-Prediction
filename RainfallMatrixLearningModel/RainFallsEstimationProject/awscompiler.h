#pragma once
#ifndef AWSCOMPILER_H
#define AWSCOMPILER_H_
#include "stdafx.h"
#include <string>
#include <vector>

class AWSCompiler {
public:
	// Seila : Extract Rainly Day from the AWS data based on how many minutes of rain per day
	std::vector< std::string > getAWSRainyDay(std::string sourcePath, int timeInterval);
	// Seila : Get AWS Longitude and Latitude from the stored AWS Information
	std::vector< double > getAWSLongLat(int ID);
};
#endif
