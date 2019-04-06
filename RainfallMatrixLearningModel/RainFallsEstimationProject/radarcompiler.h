#pragma once
#ifndef RADARCOMPILER_H
#define RADARCOMPILER_H_
#include "stdafx.h"
#include <string>

class RadarCompiler {

public:
	// Seila : convert color Hex Code value of the pixel to rainfall level (mm)
	double hexColorCodeToRainfall(std::string hexCode);
	// Seila : extract rainfall level from the targeted date and piont (x, y)
	double getRadarRainfallsLevel(std::string dateTime, int x, int y);
	double returnRainMoveDirection(std::string dateTime, int x, int y);
};
#endif
#pragma once
