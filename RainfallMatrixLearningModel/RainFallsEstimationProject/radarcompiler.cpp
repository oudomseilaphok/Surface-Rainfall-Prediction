#include "radarcompiler.h"
#include "stdafx.h"
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

// Seila : Define RWS Data Path
std::string radarFilePath = "C:/Users/BlueX/Desktop/RWS/all_in_one/";
cv::Mat image = cv::imread("C:/Users/BlueX/Desktop/RWS/case_image.png", cv::IMREAD_COLOR);

// Seila : convert color Hex Code value of the pixel to rainfall level (mm)
double RadarCompiler::hexColorCodeToRainfall(std::string hexCode) {
	UtilityComponent utilityComponent;
	double rainFall = 0.0;
	if (image.empty())                      // Check for invalid input
	{
		//std::cout << "Could not open or find the Test Case Image image" << std::endl;
	}
	else {
			 if (hexCode == utilityComponent.pixelToHex(image, 571, 97)) { rainFall = 100; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 112)) { rainFall = 90; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 127)) { rainFall = 80; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 142)) { rainFall = 70; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 157)) { rainFall = 60; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 172)) { rainFall = 50; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 187)) { rainFall = 40; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 202)) { rainFall = 35; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 217)) { rainFall = 30; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 232)) { rainFall = 25; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 247)) { rainFall = 20; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 262)) { rainFall = 18; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 277)) { rainFall = 16; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 292)) { rainFall = 14; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 307)) { rainFall = 12; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 322)) { rainFall = 10; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 337)) { rainFall = 9; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 352)) { rainFall = 8; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 367)) { rainFall = 7; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 382)) { rainFall = 6; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 397)) { rainFall = 5; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 412)) { rainFall = 4; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 427)) { rainFall = 3; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 442)) { rainFall = 2; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 457)) { rainFall = 1.5; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 472)) { rainFall = 1; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 487)) { rainFall = 0.8; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 502)) { rainFall = 0.6; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 517)) { rainFall = 0.4; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 532)) { rainFall = 0.2; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 547)) { rainFall = 0.1; }
		else if (hexCode == utilityComponent.pixelToHex(image, 571, 562)) { rainFall = 0; }

	}
	return rainFall;
};


// Seila : extract rainfall level from the targeted date and piont (x, y)
double RadarCompiler::getRadarRainfallsLevel(std::string dateTime, int x, int y) {
	UtilityComponent utilityComponent;
	double rainFall = 0.0;
	std::string FileName = "RDR_CMI_" + dateTime.substr(0, 4) + dateTime.substr(5, 2) + dateTime.substr(8, 2) + dateTime.substr(11, 2) + dateTime.substr(14, 2) + ".png";
	FileName = radarFilePath + FileName;
	cv::Mat image;
	image = cv::imread(FileName.c_str(), cv::IMREAD_COLOR); // Read the file
	if (image.empty())                      // Check for invalid input
	{
		//std::cout << "Could not open or find the image" << std::endl;
	}
	else {
		rainFall = hexColorCodeToRainfall(utilityComponent.pixelToHex(image, x, y));
	}
	return rainFall;
}

// Seila : This function is for extract rainfall moving direction based on previous strong rainfall location
double RadarCompiler::returnRainMoveDirection(std::string dateTime, int onRadarX, int onRadarY) {

	double direction = 2; // default direction is 2, in case the extraction failed
	std::vector <std::string> analyzeTime;
	std::vector <double> xPosition;
	std::vector <double> yPosition;
	std::string T0FileName = "RDR_CMI_" + dateTime.substr(0, 4) + dateTime.substr(5, 2) + dateTime.substr(8, 2) + dateTime.substr(11, 2) + "00" + ".png";
	std::string T1FileName = "RDR_CMI_" + dateTime.substr(0, 4) + dateTime.substr(5, 2) + dateTime.substr(8, 2) + dateTime.substr(11, 2) + "10" + ".png";

	std::string T4FileName = "RDR_CMI_" + dateTime.substr(0, 4) + dateTime.substr(5, 2) + dateTime.substr(8, 2) + dateTime.substr(11, 2) + "40" + ".png";
	std::string T5FileName = "RDR_CMI_" + dateTime.substr(0, 4) + dateTime.substr(5, 2) + dateTime.substr(8, 2) + dateTime.substr(11, 2) + "50" + ".png";

	analyzeTime.push_back(T0FileName);
	analyzeTime.push_back(T1FileName);
	analyzeTime.push_back(T4FileName);
	analyzeTime.push_back(T5FileName);

	for (int i = 0; i < analyzeTime.size(); i++) {
		int highestRainValue = 0;
		int xHighPos = 0;
		int yHighPos = 0;
		UtilityComponent utilityComponent;
		std::string fileName = analyzeTime[i];
		fileName = radarFilePath + fileName;
		cv::Mat image;
		image = cv::imread(fileName.c_str(), cv::IMREAD_COLOR); // Read the file
		// if cannot find radar File
		if (image.empty())                      // Check for invalid input
		{
			//std::cout << "Could not open or find the image" << std::endl;
		}
		else {
			for (int x = (-1 * 16); x <= 16; x++) {
				int realX = onRadarX + x;

				for (int y = (-1 * 16); y <= 16; y++) {
					int realY = onRadarY + y;
					//get total Radar Rainfall // change int to double
					double total25Cells = 0;
					for (int internalX = (-1 * 4); internalX <= 4; internalX++) {
						int realIntX = realX + internalX;
						for (int internalY = (-1 * 4); internalY <= 4; internalY++) {
							int realIntY = realY + internalY;
							//get total Radar Rainfall // change int to double
							total25Cells = total25Cells + hexColorCodeToRainfall(utilityComponent.pixelToHex(image, realIntX, realIntY));

						}
					}
					if (total25Cells > highestRainValue) {
						highestRainValue = total25Cells;
						xHighPos = realX;
						yHighPos = realY;
					}
				}
			}
			xPosition.push_back(xHighPos);
			yPosition.push_back(yHighPos);
		}
	}


	int xChange = xPosition[0] - xPosition[1];
	int yChange = yPosition[0] - yPosition[1];

	// check if new higher rain value come into scope
	if (xChange >= 0) {
		xChange = xPosition[2] - xPosition[3];
		yChange = yPosition[2] - yPosition[3];
	}
	// check if new higher rain value come into scope
	if (xChange >= 0) {
		xChange = xPosition[1] - xPosition[2];
		yChange = yPosition[1] - yPosition[2];
	}

	// case 3
	if (xChange < 0 && yChange > 0) {
		direction = 3;
		std::cout << "Direction : " << direction << std::endl;
	}
	// case 1
	else if (xChange < 0 && yChange < 0) {
		direction = 1;
		std::cout << "Direction : " << direction << std::endl;
	}

	// case 2
	else if (xChange < 0 && yChange == 0) {
		direction = 2;
		std::cout << "Direction : " << direction << std::endl;
	}

	// case 0
	else if (xChange == 0 && yChange < 0) {
		direction = 0;
		std::cout << "Direction : " << direction << std::endl;
	}

	// case 4
	else if (xChange == 0 && yChange > 0) {
		direction = 4;
		std::cout << "Direction : " << direction << std::endl;
	}

	// case 9, if unable to extract use case 2 // special weather condition
	else {
		std::cout << "Unable to extract wind direction" << std::endl;
	}

	return direction;
}


