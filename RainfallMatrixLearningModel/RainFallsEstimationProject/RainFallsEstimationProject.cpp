// RainEstimationProject.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include "awscompiler.h"
#include "radarcompiler.h"
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cstdio>
#include <windows.h>

using namespace cv;
using namespace std;


string seoulID[5] = { "400", "401", "402", "403", "404" };
double seoulLong[5] = { 127.04671, 127.02601, 127.14498, 127.0967, 126.82953 };
double seoulLat[5] = { 37.5134, 37.48462, 37.55552, 37.51151, 37.5739 };

string gangWonID[5] = { "563", "583", "556", "560", "561" };
double gangWonLong[5] = { 128.6828, 128.1551, 127.98527, 128.56447, 128.15275 };
double gangWonLat[5] = { 37.46356, 37.46463, 38.09799, 37.64794, 37.58219 };

string incheonID[5] = { "511", "502", "570", "501", "513" };
double incheonLong[5] = { 126.69047 , 126.29164 , 126.64233 , 125.69819 , 126.14495 };
double incheonLat[5] = { 37.55498, 37.78944, 37.6232, 37.65891, 37.22721 };

string gyeongidoID[5] = { "531", "505", "546", "540", "495" };
double gyeongidoLong[5] = { 127.55219, 127.34543, 127.2592, 126.892, 127.16254167 };
double gyeongidoLat[5] = { 37.89807, 37.82439, 37.43529, 37.6373, 36.98062 };

// Radar Source Directory Path
string radarSourcePath = "C:/Users/BlueX/Desktop/RWS/all_in_one/";
AWSCompiler awsCompiler;
UtilityComponent utilityComponent;
RadarCompiler radarCompiler;

// TD = Training Station, DS = Different Station
std::vector<string> datesTD;
std::vector<string> _availableExpDateTimeTD;
std::vector<string> _10MinDatetimesTD;
std::vector<string> _allOneMinDatetimesTD;
std::vector<double> _allOneMinrainfallsTD;
std::vector<double> _allOneTemperatureTD;
std::vector<double> _allOneWindSpeedTD;
std::vector<double> _allOneWindDirTD;
std::vector<double> _allOneHumidityTD;

// Seila : Extract Raining Scenario Date for Training Data where there is ground rainfall received
std::vector<string> showRainyScenario(string sourcePath, std::vector<string> dates) {
	std::vector<string> outputExperimentDateTimeTD;
	ifstream file(sourcePath);
	if (!file.is_open()) std::cout << "ERROR: File Open" << '\n';
	int shuffle = 0;
	string stationNumber;
	string dateTime;
	string windDirection;
	string windSpeed;
	string temperature;
	string rainFall;
	string rainExistence;
	string pressure;
	string localPressure;
	string humid;
	string sunshine;
	string sunshineDuration;

	string tempDate = "";
	int tempDateIndex = 0;

	double totalRainfalls = 0.0;
	double totalWindSpeed = 0.0;
	double totalWindDirection = 0.0;
	double totalTemperature = 0.0;
	//std::cout <<"Gathering Data..." << std::endl;

	while (file.good()) {
		getline(file, stationNumber, ',');
		getline(file, dateTime, ',');
		getline(file, temperature, ',');
		getline(file, rainFall, ',');
		getline(file, rainExistence, ',');
		getline(file, windDirection, ',');
		getline(file, windSpeed, ',');
		getline(file, pressure, ',');
		getline(file, localPressure, ',');
		getline(file, humid, ',');
		getline(file, sunshine, ',');
		getline(file, sunshineDuration, '\n');

		_allOneMinDatetimesTD.push_back(dateTime);
		_allOneMinrainfallsTD.push_back(strtof((rainFall).c_str(), 0));
		_allOneWindSpeedTD.push_back(strtof((windSpeed).c_str(), 0));
		_allOneWindDirTD.push_back(strtof((windDirection).c_str(), 0));
		// Convert to 10 minutes / case Data because radar data has 10 minute / case
		if (dateTime.substr(0, 4) == "2017") { //2017-01-02 18:18
			if (dateTime.substr(15, 1) == "0") {

				_10MinDatetimesTD.push_back(dateTime);
				if ((std::find(dates.begin(), dates.end(), dateTime.substr(0, 10)) != dates.end()) && _10MinDatetimesTD.size() > 1) {
					int _10MinPosition = std::find(_10MinDatetimesTD.begin(), _10MinDatetimesTD.end(), dateTime) - _10MinDatetimesTD.begin();
					outputExperimentDateTimeTD.push_back(_10MinDatetimesTD[_10MinPosition - 1]);
					
				}
				totalRainfalls = 0.0;
			}
		}
		else {
			totalRainfalls = 0.0;
		}
	}
	file.close();
	return outputExperimentDateTimeTD;
}



// Seila : Extract Raining Scenario Date for Testing on Same Station that Model generated from
double get10MinutesRainfall(string dateTime) {
	double rainIntensity = 0;
	int position = std::find(_allOneMinDatetimesTD.begin(), _allOneMinDatetimesTD.end(), dateTime) - _allOneMinDatetimesTD.begin();
	for (int i = 1; i <= 10; i++) {
		//cout << "ground : " << _allOneMinrainfallsTD[position + i] << endl;
		rainIntensity = rainIntensity + _allOneMinrainfallsTD[position + i];
	}
	rainIntensity = rainIntensity / 10;
	return rainIntensity;
}


double get10MinutesWindSpeed(string dateTime) {
	double outputWindSpeed = 0;
	int position = std::find(_allOneMinDatetimesTD.begin(), _allOneMinDatetimesTD.end(), dateTime) - _allOneMinDatetimesTD.begin();
	for (int i = 1; i <= 10; i++) {
		//cout << "ground : " << _allOneWindSpeedTD[position + i] << endl;
		outputWindSpeed = outputWindSpeed + _allOneWindSpeedTD[position + i];
	}
	outputWindSpeed = outputWindSpeed / 10;
	return outputWindSpeed;
}

double get10MinutesWindDirection(string dateTime) {
	double outputDirection = 0;
	std:vector<double> allWindDirections;
	int position = std::find(_allOneMinDatetimesTD.begin(), _allOneMinDatetimesTD.end(), dateTime) - _allOneMinDatetimesTD.begin();
	while (outputDirection == 0) {
		outputDirection = _allOneWindDirTD[position];
		cout << dateTime << " : " << outputDirection << endl;
		position++;
	}
	return outputDirection;

}

// These two functions are needed if temp

//double get10MinutesTemperature(string dateTime) {
//	double outputTemperature = 0;
//	int position = std::find(_allOneMinDatetimesTD.begin(), _allOneMinDatetimesTD.end(), dateTime) - _allOneMinDatetimesTD.begin();
//	for (int i = 1; i <= 10; i++) {
//		//cout << "ground : " << _allOneTemperatureTD[position + i] << endl;
//		outputTemperature = outputTemperature + _allOneTemperatureTD[position + i];
//	}
//	outputTemperature = outputTemperature / 10;
//	return outputTemperature;
//}
//
//double get10MinutesHumid(string dateTime) {
//	double outputHumid = 0;
//	int position = std::find(_allOneMinDatetimesTD.begin(), _allOneMinDatetimesTD.end(), dateTime) - _allOneMinDatetimesTD.begin();
//	for (int i = 1; i <= 10; i++) {
//		//cout << "ground : " << _allOneTemperatureTD[position + i] << endl;
//		outputHumid = outputHumid + _allOneHumidityTD[position + i];
//	}
//	outputHumid = (outputHumid / 10);
//	return outputHumid;
//}


int main(int argc, char** argv)
{

	int stationIndex = 4;
	string trainDataSource = "aws/SURFACE_AWS_" + gangWonID[stationIndex] + "_MI_2017-07_2017-07_2018.csv";
	int trainXaxis = utilityComponent.extractAWSLocation(gangWonLong[stationIndex], gangWonLat[stationIndex])[0];
	int trainYaxis = utilityComponent.extractAWSLocation(gangWonLong[stationIndex], gangWonLat[stationIndex])[1];

	datesTD = awsCompiler.getAWSRainyDay(trainDataSource, 60);
	
	_availableExpDateTimeTD = showRainyScenario(trainDataSource, datesTD);
	
	fstream file;

	int spatialArea = 7; // 3 = 3*3; 5 = 5*5
	int spatialRadius = (spatialArea - 1) / 2;
	string filename = std::to_string(stationIndex) + "_20170708_0130_2340_3pixels.csv";
	file.open(filename, fstream::out);
	file << "Date,AWS,wind_speed,wind_direction";
	for (int i = 1; i <= (spatialArea * spatialArea); i++) {
		file << ",R" << i;
	}
	file << "\n";

	int startIndex = std::find(_availableExpDateTimeTD.begin(), _availableExpDateTimeTD.end(), "2017-07-08 01:30") - _availableExpDateTimeTD.begin();
	int endIndex = std::find(_availableExpDateTimeTD.begin(), _availableExpDateTimeTD.end(), "2017-07-08 23:40") - _availableExpDateTimeTD.begin();
	
	for (int j = startIndex; j <= endIndex; j++) {

		std:vector<double> tempRadarValue;
		int countRadar = 1;
		double awsValue = get10MinutesRainfall(_availableExpDateTimeTD[j]);
		double windSpeedValue = get10MinutesWindSpeed(_availableExpDateTimeTD[j]);
		double windDirectionValue = get10MinutesWindDirection(_availableExpDateTimeTD[j]);
		//if (awsValue > 0) {

		file << _availableExpDateTimeTD[j] << "," << awsValue << "," << windSpeedValue << "," << windDirectionValue;

			for (int x = (-1 * spatialRadius); x <= spatialRadius; x++) {
				int realX = trainXaxis + x;
				for (int y = (-1 * spatialRadius); y <= spatialRadius; y++) {
					int realY = trainYaxis + y;
					double radarRainTemp = radarCompiler.getRadarRainfallsLevel(_availableExpDateTimeTD[j], realX, realY);
					file << "," << radarRainTemp;
				}
			}

			file << "\n";
	}
	file.close();
	system("pause");

}








