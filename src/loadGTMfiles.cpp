/**********************************************************************************************************
FILE: loadGTMfiles.cpp

PLATFORM: Windows 7, MS Visual Studio 2010, OpenCV 2.4.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: May 2017

LOCATION: TechGate Vienna, Donau-City-Stra�e 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides functionalities for loading and showing the GTMs.
**********************************************************************************************************/
#include <iostream>
#include "kfeature.h"
#include "cell.h"
#include "loadGTMfiles.h"
#include "io_helper.h"
#include "readGTM.h"

#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

//Show a fraction of the matches
int showMatches(std::vector<cv::DMatch> matches,
                std::vector<cv::KeyPoint> keypL,
                std::vector<cv::KeyPoint> keypR,
                cv::Mat imgs[2],
                size_t keepNMatches = 20);
int Optimize_Matches(std::vector<cv::DMatch> matches,
                     std::vector<cv::KeyPoint> keypL,
                     std::vector<cv::KeyPoint> keypR, cv::Mat imgs[2]);
int Optimize(std::vector<cv::DMatch> &good_matches, std::vector<cv::DMatch> matches,
             std::vector<cv::KeyPoint> keypL,
             std::vector<cv::KeyPoint> keypR,
             cv::Mat imgs[2]);

void evaluate(vector<DMatch> true_match, vector<DMatch> my_match);

/* Reads all images and GTMs and shows the matches of each image pair in the given folder
*
* string img_path    Input  -> Path to the images
* string l_img_pref	 Input  -> Prefix and/or postfix with optional subfolder for the left/first images.
*									   It can include a folder structure that follows after the filepath, a file prefix,
*									   a '*' indicating the position of the number and a postfix. If it is empty,
*									   all files from the folder img_path are used (also if l_img_pref only contains a
*									   folder ending with '/', every file within this folder is used). It is possible to
*									   specify only a prefix with or without '*' at the end. If a prefix is used, all
*									   characters until the first number (excluding) must be provided. For a
*									   postfix, '*' must be placed before the postfix. Valid examples :
*									   folder/pre_*post, *post, pre_*, pre_, folder/*post, folder/pre_*, folder/pre_,
*									   folder/, folder/folder/, folder/folder/pre_*post, ...
*									   For non stereo images (consecutive images), r_img_pref must be empty.
* string r_img_pref	 Input  -> Prefix and/or postfix with optional subfolder for the right/second images.
*									   For non stereo images (consecutive images), r_img_pref must be empty.
*									   For further details see the description of l_img_pref.
* string gtm_path		 Input  -> Path to the ground truth images. It must contain all GTM files that correspond to all
*									   image pairs specified by img_path, l_img_pref, and r_img_pref.
* string gtm_postfix Input  -> Specifies the postfix of GTM files. It must include the intended inlier
*									   ratio (10 * inlier ratio in percent) and keypoint type. E.g. 'inlRat950FAST.gtm'.
* 									 Specifying an additional folder is also possible: e.g. 'folder/*inlRat950FAST.gtm'
*
* Return value:      0:  Everything ok
*							      -1:  Error while reading GTM file
*/
int showGTM(std::string img_path, std::string l_img_pref, std::string r_img_pref,
            std::string gtm_path, std::string gtm_postfix)
{
  int err;
  cv::Mat src[2];
  string fileprefl, fileprefr;
  vector<string> filenamesl, filenamesr, filenamesgtm;

  fileprefl = l_img_pref;
  if (r_img_pref.empty())
    fileprefr = fileprefl;
  else
    fileprefr = r_img_pref;

  //Load corresponding image names
  err = loadImgStereoSequence(img_path, fileprefl, fileprefr, filenamesl, filenamesr);
  if (err || filenamesl.empty() || filenamesr.empty() || (filenamesl.size() != filenamesr.size()))
  {
    cout << "Could not find flow images! Exiting." << endl;
    exit(0);
  }

  //load GTM names
  err = loadGTMSequence(gtm_path, gtm_postfix, filenamesgtm);
  if (err || filenamesgtm.empty() || (filenamesgtm.size() != filenamesl.size()))
  {
    cout << "Could not find GTM files! Exiting." << endl;
    exit(0);
  }

  // read images and GTMs
  err = 0;
  std::vector<bool> leftInlier;
  std::vector<cv::DMatch> matchesGT;
  std::vector<cv::KeyPoint> keypL;
  std::vector<cv::KeyPoint> keypR;
  double inlRatioL, inlRatioR, inlRatioO, positivesGT, negativesGTl, negativesGTr, usedMatchTH;
  for (int k = 0; k < (int)filenamesl.size(); k++)
  {
    cv::Mat flowimg;
    src[0] = cv::imread(filenamesl[k]);
    src[1] = cv::imread(filenamesr[k]);
    if (readGTMatchesDisk(filenamesgtm[k],
                          leftInlier,
                          matchesGT,
                          keypL,
                          keypR,
                          &inlRatioL,
                          &inlRatioR,
                          &inlRatioO,
                          &positivesGT,
                          &negativesGTl,
                          &negativesGTr,
                          &usedMatchTH))
    {
      cout << "Succesfully read GTM file " << filenamesgtm[k] << endl;
      cout << "Inlier ratio in first/left image: " << inlRatioL << endl;
      cout << "Inlier ratio in second/right image: " << inlRatioR << endl;
      cout << "Mean inlier ratio of both images: " << inlRatioO << endl;
      cout << "Number of true positive matches: " << positivesGT << endl;
      cout << "Number of left negatives (having no corresponding right match): " << negativesGTl << endl;
      cout << "Number of right negatives (having no corresponding left match): " << negativesGTr << endl;
      cout << "Threshold used to generate GTM: " << usedMatchTH << endl
           << endl;
      // cout << matchesGT.size() << endl;
      // cout << keypL.size() << endl;
      // cout << keypR.size() << endl;
      // cout << leftInlier.size() << endl;
      Optimize_Matches(matchesGT, keypL, keypR, src);
      //showMatches(matchesGT, keypL, keypR, src, 10);
    }
    else
    {
      cout << "Error while reading GTM file " << filenamesgtm[k] << endl
           << endl;
      err = -1;
    }
  }

  return err;
}

/* Shows a fraction of the matches.
*
* vector<DMatch> matches	Input  -> Matches
* vector<KeyPoint> keypL	Input  -> left/first keypoints
* vector<KeyPoint> keypR	Input  -> right/second keypoints
* Mat imgs[2]				Input  -> Images used for matching [0] = left/first image, [1] = right/second image
* size_t keepNMatches		Input  -> Number of matches to display [Default = 20]
*
* Return value:				 0:		  Everything ok
*/
int showMatches(std::vector<cv::DMatch> matches,
                std::vector<cv::KeyPoint> keypL,
                std::vector<cv::KeyPoint> keypR,
                cv::Mat imgs[2],
                size_t keepNMatches)
{
  //---------------------------------------------------------------------------

  //------------------------------------------------------
  vector<char> matchesMask(matches.size(), false);
  ;
  Mat img_correctMatches;
  float keepXthMatch;
  float oldremainder, newremainder = 0;

  //Reduce number of displayed matches
  keepXthMatch = 1.0f;
  if (matches.size() > keepNMatches)
    keepXthMatch = (float)matches.size() / (float)keepNMatches;
  oldremainder = 0;
  for (unsigned int i = 0; i < matches.size(); i++)
  {
    newremainder = fmod((float)i, keepXthMatch);
    if (oldremainder >= newremainder)
    {
      matchesMask[i] = true;
    }
    oldremainder = newremainder;
  }
  //Draw true positive matches
  drawMatches(imgs[0],
              keypL,
              imgs[1],
              keypR,
              matches,
              img_correctMatches,
              Scalar::all(-1),
              Scalar::all(-1),
              matchesMask,
              cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  imwrite("./gr.png", img_correctMatches);
  //Show result
  cvNamedWindow("Ground Truth Matches");
  imshow("Ground Truth Matches", img_correctMatches);
  waitKey(0);
  cv::destroyWindow("Ground Truth Matches");

  return 0;
}
void evaluate(vector<DMatch> true_match, vector<DMatch> my_match)
{
  std::map<int, int> true_matches;
  for (int i = 0; i < true_match.size(); i++)
  {
    true_matches[true_match[i].trainIdx] = true_match[i].queryIdx;
  }
  double s = 0, s1 = 0;

  for (int j = 0; j < my_match.size(); j++)
  {
    int trainId = my_match[j].trainIdx;
    int queryId = my_match[j].queryIdx;
    if (true_matches.find(trainId) != true_matches.end())
    {
      s1++;
      if (queryId == true_matches[trainId])
      {
        s++;
      }
    }
  }
  std::cout << "s = " << s << ",   "
            << "s1=" << s1 << std::endl;

  std::cout << "result:  " << s / s1 << std::endl;
}

int Optimize(std::vector<cv::DMatch> &good_matches, std::vector<cv::DMatch> matches,
             std::vector<cv::KeyPoint> keypL,
             std::vector<cv::KeyPoint> keypR,
             cv::Mat imgs[2])
{
  // init feature and cell
  int rows = imgs[0].rows;
  int cols = imgs[0].cols;
  // std::cout << rows << "   " << cols << std::endl;
  int row_n = 30, col_m = 40;
  std::vector<std::vector<Cell *>> Map1, Map2;
  for (int i = 0; i < col_m; i++)
  {
    std::vector<Cell *> temp1, temp2;
    for (int j = 0; j < row_n; j++)
    {
      Cell *c1 = new Cell(i, j);
      Cell *c2 = new Cell(i, j);

      temp1.push_back(c1);
      temp2.push_back(c2);
    }
    Map1.push_back(temp1);
    Map2.push_back(temp2);
    //std::cout << temp1.size() << std::endl;
  }

  // for (int it = 0; it < keypL.size(); it++)
  // {
  //   double x = keypL[it].pt.x;
  //   double y = keypL[it].pt.y;
  //   int id_y = (y * row_n) / rows;
  //   int id_x = (x * col_m) / cols;
  //   Kfeature *kfeature = new Kfeature();
  //   kfeature->trainId_ = it;
  //   Map1[id_x][id_y]->kfeatures_.push_back(kfeature);
  // }

  // for (int iu = 0; iu < keypR.size(); iu++)
  // {
  //   double x = keypR[iu].pt.x;
  //   double y = keypR[iu].pt.y;
  //   int id_y = (y * row_n) / rows;
  //   int id_x = (x * col_m) / cols;
  //   Kfeature *kfeature = new Kfeature();
  //   kfeature->trainId_ = iu;
  //   Map2[id_x][id_y]->kfeatures_.push_back(kfeature);
  // }

  for (auto M0 : matches)
  {
    double xl = keypL[M0.trainIdx].pt.x;
    double yl = keypL[M0.trainIdx].pt.y;
    double xr = keypR[M0.queryIdx].pt.x;
    double yr = keypR[M0.queryIdx].pt.y;
    int id1_y = (yl * row_n) / rows;
    int id1_x = (xl * col_m) / cols;
    int id2_y = (yr * row_n) / rows;
    int id2_x = (xr * col_m) / cols;
    Kfeature *kfeaturel = new Kfeature();
    kfeaturel->trainId_ = M0.trainIdx;
    Map1[id1_x][id1_y]->kfeatures_.push_back(kfeaturel);
    Kfeature *kfeaturer = new Kfeature();
    kfeaturer->trainId_ = M0.queryIdx;
    Map2[id2_x][id2_y]->kfeatures_.push_back(kfeaturer);
  }
  // for (int ii = 0; ii < col_m; ii++)
  // {
  //   for (int jj = 0; jj < row_n; jj++)
  //   {
  //     std::cout << Map1[ii][jj]->kfeatures_.size() << ", ";
  //   }
  // }
  // std::cout << std::endl;
  // for (int i = 0; i < col_m; i++)
  // {
  //   for (size_t j = 0; j < row_n; j++)
  //   {
  //     std::cout << Map2[i][j]->kfeatures_.size() << ", ";
  //   }
  // }
  int r = 5;
  for (auto M : matches)
  {
    std::vector<double> map1_distr, map2_distr;
    double xl = keypL[M.trainIdx].pt.x;
    double yl = keypL[M.trainIdx].pt.y;
    double xr = keypR[M.queryIdx].pt.x;
    double yr = keypR[M.queryIdx].pt.y;
    int id1_y = (yl * row_n) / rows;
    int id1_x = (xl * col_m) / cols;
    int id2_y = (yr * row_n) / rows;
    int id2_x = (xr * col_m) / cols;

    for (int xi = 0; xi < 2 * r; xi++)
    {
      for (int yi = 0; yi < 2 * r; yi++)
      {
        if ((id1_x - r + xi) < 0 || (id1_x - r + xi) >= col_m || (id1_y - r + yi) < 0 || (id1_y - r + yi) > row_n)
          map1_distr.push_back(0);
        else
          map1_distr.push_back(Map1[id1_x][id1_y]->kfeatures_.size());

        if ((id2_x - r + xi) < 0 || (id2_x - r + xi) >= col_m || (id2_y - r + yi) < 0 || (id2_y - r + yi) > row_n)
          map2_distr.push_back(0);
        else
          map2_distr.push_back(Map2[id2_x][id2_y]->kfeatures_.size());
      }
    }
    std::vector<double> map1w(r * r, 1.0);

    map1w[r * r / 2 - 1] = 5;
    map1w[r * r / 2 - 2] = 7;
    map1w[r * r / 2] = 7;
    map1w[r * r / 2 + 1] = 7;
    map1w[r * r / 2 + 1] = 5;
    std::vector<double> map2w(r * r, 1.0);
    map2w[r * r / 2 - 1] = 5;
    map2w[r * r / 2 - 2] = 7;
    map2w[r * r / 2] = 7;
    map2w[r * r / 2 + 1] = 7;
    map2w[r * r / 2 + 1] = 5;

    // int sum = abs(Map1[id1_x][id1_y]->kfeatures_.size() - Map2[id2_x][id2_y]->kfeatures_.size());
    // if (sum <= 1)
    //   good_matches.push_back(M);
    double dist = wasserstein(map1_distr, map1w, map2_distr, map2w);
    // std::cout << "dist score : " << dist << std::endl;
    if (dist < 2)
      good_matches.push_back(M);
    //std::cout << map1_distr.size() << std::endl;
  }
}
int Optimize_Matches(std::vector<cv::DMatch> matches,
                     std::vector<cv::KeyPoint> keypL,
                     std::vector<cv::KeyPoint> keypR,
                     cv::Mat imgs[2])
{
  //--------------------------------------------------------
  Ptr<ORB> orb = ORB::create(2000);
  Mat des1, des2;
  orb->compute(imgs[0], keypL, des1);
  orb->compute(imgs[1], keypR, des2);

  // Mat outimg1;
  // drawKeypoints(imgs[0], keypL, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
  // imshow("ORB features", outimg1);
  // waitKey(0);

  vector<DMatch> matches1;
  BFMatcher matcher(NORM_HAMMING);
  //BFMatcher matcher ( NORM_HAMMING );
  matcher.match(des1, des2, matches1);
  std::vector<DMatch> good_matches;

  // 比率测试
  double min_dist = 10000, max_dist = 0;
  for (int i = 0; i < des1.rows; i++)
  {
    double dist = matches1[i].distance;
    if (dist < min_dist)
      min_dist = dist;
    if (dist > max_dist)
      max_dist = dist;
  }
  for (int i = 0; i < des1.rows; i++)
  {
    if (matches1[i].distance <= max(2 * min_dist, 50.0))
    {
      good_matches.push_back(matches1[i]);
    }
  }
  std::vector<DMatch> good_matches1;

  Optimize(good_matches1, good_matches, keypL, keypR, imgs);
  evaluate(matches, good_matches1);

  evaluate(matches, good_matches);

  // std::cout << matches.size() << std::endl
  //           << matches1.size() << std::endl;
  showMatches(good_matches1, keypL, keypR, imgs, 10000);
  //------------------------------------------------------
  return 0;
}
