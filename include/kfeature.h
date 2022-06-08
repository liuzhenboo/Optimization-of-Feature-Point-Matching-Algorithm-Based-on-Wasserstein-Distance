#pragma once
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "cell.h"
using namespace std;
using namespace cv;
class Cell;
class Kfeature
{
public:
    //data------------------------
    // matching results
    int trainId_;

    // function-----------------------------
    Kfeature(/* args */);
    ~Kfeature();
};
