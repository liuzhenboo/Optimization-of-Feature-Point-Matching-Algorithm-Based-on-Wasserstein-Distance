#pragma once
#include <vector>
#include <opencv2/highgui/highgui.hpp>

#include "kfeature.h"
using namespace std;
using namespace cv;

class Kfeature;

class Cell
{
public:
    // datas
    std::vector<Kfeature *> kfeatures_;
    int idx_, idy_;

    // fuctions
    Cell(/* args */);
    Cell(int i, int j);
    ~Cell();
};
