#include <Eigen/Dense>
#include <Eigen/core>
#include <glog/logging.h>
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <string.h>

using namespace cv;

// start globals
#define WINDOW_WIDTH 64
#define WINDOW_HEIGHT 128
#define CELL_SIZE 64
#define BIN_COUNT 9
#define HISTOGRAM_WIDTH 8
#define HISTOGRAM_HEIGHT 16
#define NORMAL_BLOCK_WIDTH 7
#define NORMAL_BLOCK_HEIGHT 15
#define PI 3.14159265
#define HOG_SIZE 3780
// end globals

class HOG {

public:
  HOG() {}
  ~HOG() {}

  // start structs
  struct gradientData {
    double g_x;
    double g_y;
    double gradient;
    double angle;
  };

  struct histogram {
    double zero = 0;
    double twenty = 0;
    double forty = 0;
    double sixty = 0;
    double eighty = 0;
    double oneHundred = 0;
    double oneHundredTwenty = 0;
    double oneHundredForty = 0;
    double oneHundredSixty = 0;
  };
  // end structs

  typedef struct gradientData gradient_t;
  typedef struct histogram histogram_t;

  gradient_t calculateGradient(const cv::Mat &image, double x, double y);

  void addToBin(histogram_t *cellHistogram, const gradient_t &pixelIter);

  histogram_t
  getHistogram(const std::vector<std::vector<gradient_t>> &gradientMatrix,
               int x, int y);

  std::vector<double> histogramToDoubleArray(histogram_t *block);

  std::vector<double> concatenateVectors(std::vector<double> block1,
                                         std::vector<double> block2,
                                         std::vector<double> block3,
                                         std::vector<double> block4,
                                         int vecSize);

  double vectorSum(histogram_t *block);

  std::vector<double> vectorNormalize(histogram_t block, double norm);

  std::vector<double> getHOGForWindow(Mat img);

  Mat visualize(const Mat &img, std::vector<double> &descriptor,
                const Size &size);
};