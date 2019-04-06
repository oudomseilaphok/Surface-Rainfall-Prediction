#include "awscompiler.h"
#include "stdafx.h"
#include <iostream>
#include <iomanip> 

// Seila : Extract Rainly Day from the AWS data based on how many minutes of rain per day
std::vector<std::string> AWSCompiler::getAWSRainyDay(std::string sourcePath, int timeInterval) {
	UtilityComponent utilityComponent;
	std::vector<std::string> resultdates;
	
	std::ifstream file(sourcePath);
	if (!file.is_open()) std::cout << "ERROR: File Open" << '\n';

	std::string stationNumber;
	std::string dateTime;
	std::string windDirection;
	std::string windSpeed;
	std::string temperature;
	std::string rainFall;
	std::string rainExistence;
	std::string pressure;
	std::string localPressure;
	std::string humid;
	std::string sunshine;
	std::string sunshineDuration;

	int intervalCounter = 0;
	std::string currentDate = "";

	//std::cout << std::left << std::setw(5) << "#" << std::setw(5) << "YYYY-MM-DD" << '\n';
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

		// Display all rainy day code		
		if (utilityComponent.isFloat(rainFall) && strtof((rainFall).c_str(), 0) > 0) {
			if (currentDate == "" || std::find(resultdates.begin(), resultdates.end(), dateTime.substr(0, 10)) != resultdates.end() || currentDate != dateTime.substr(0, 10)) {
				intervalCounter = 0;
				currentDate = dateTime.substr(0, 10);
			}
			else {
				intervalCounter++;
				//dateTime.substr(0, 10) == "2017-07-02" || dateTime.substr(0, 10) == "2017-07-03" ||
				if (intervalCounter >= timeInterval && (dateTime.substr(0, 10) == "2017-07-10" || dateTime.substr(0, 10) == "2017-07-02")) {
					resultdates.push_back(dateTime.substr(0, 10));
					//std::cout << std::left << std::setw(5) << resultdates.size() << std::setw(5) << dateTime.substr(0, 10) << '\n';
				}
			}
		}
	}
	//std::cout << "Total date that has more than " << timeInterval << " minutes of rain : " << resultdates.size() << "\n\n";
	file.close();
	return resultdates;
}
