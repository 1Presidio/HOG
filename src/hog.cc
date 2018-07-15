#include "hog.h"

using namespace cv;

/*
 * This function calculates the gradient for a pixel using the left and right
 * pixels as the y delta and the top and bottom pixels as the x delta
 * We can then calculate the magnitude and direction of the gradient
 */
HOG::gradient_t calculateGradient(const cv::Mat &image, double x, double y) {
  double x_right = x + 1;
  double x_left = x - 1;
  double y_top = y - 1;
  double y_bottom = y + 1;

  // handling edge cases
  if (x_left < 0) {
    x_left = 0;
  } else {
    x_left = image.at<uchar>(x_left, y);
  }

  if (y_top < 0) {
    y_top = 0;
  } else {
    y_top = image.at<uchar>(x, y_top);
  }

  if (x_right >= image.cols) {
    x_right = 0;
  } else {
    x_right = image.at<uchar>(x_right, y);
  }

  if (y_bottom >= image.rows) {
    y_bottom = 0;
  } else {
    y_bottom = image.at<uchar>(x, y_bottom);
  }

  int g_x;
  int g_y;      // x and y direction gradients
  double angle; // angle of gradient

  g_x = x_right - x_left;
  g_y = y_bottom - y_top;

  if (g_x == 0 && g_y != 0) {
    angle = 90;
  } else if (g_y == 0 && g_x != 0) {
    angle = 0;
  } else if (g_y == 0 && g_x == 0) {
    angle = 0;
  } else {
    angle = (atan(g_y / g_x) * 180) / PI; // convert to degrees

    if (angle < 0) {
      angle *= -1;
      // negative angles are the same as positive according to the paper this
      // a.k.a unsigned gradient
    }
  }

  double gradient = sqrt(pow(g_x, 2) + pow(g_y, 2));
  HOG::gradient_t pixelGradient;
  pixelGradient.g_x = g_x;
  pixelGradient.g_y = g_y;
  pixelGradient.gradient = gradient;
  pixelGradient.angle = angle;
  return pixelGradient;
}

/*
 * This function adds a gradient to the histogram with it's respective
 * preportions
 */
void addToBin(HOG::histogram_t *cellHistogram,
              const HOG::gradient_t &pixelIter) {
  double magnitude = pixelIter.gradient;
  double angle = pixelIter.angle;
  if (angle < 20) {
    double upperBin = magnitude * ((angle) / 20);
    double lowerBin = magnitude - upperBin;
    cellHistogram->twenty += upperBin;
    cellHistogram->zero += lowerBin;
  } else if (angle < 40) {
    double upperBin = magnitude * ((angle - 20) / 20);
    double lowerBin = magnitude - upperBin;
    cellHistogram->forty += upperBin;
    cellHistogram->twenty += lowerBin;
  } else if (angle < 60) {
    double upperBin = magnitude * ((angle - 40) / 20);
    double lowerBin = magnitude - upperBin;
    cellHistogram->sixty += upperBin;
    cellHistogram->forty += lowerBin;
  } else if (angle < 80) {
    double upperBin = magnitude * ((angle - 60) / 20);
    double lowerBin = magnitude - upperBin;
    cellHistogram->eighty += upperBin;
    cellHistogram->sixty += lowerBin;
  } else if (angle < 100) {
    double upperBin = magnitude * ((angle - 80) / 20);
    double lowerBin = magnitude - upperBin;
    cellHistogram->oneHundred += upperBin;
    cellHistogram->eighty += lowerBin;
  } else if (angle < 120) {
    double upperBin = magnitude * ((angle - 100) / 20);
    double lowerBin = magnitude - upperBin;
    cellHistogram->oneHundredTwenty += upperBin;
    cellHistogram->oneHundred += lowerBin;
  } else if (angle < 140) {
    double upperBin = magnitude * ((angle - 120) / 20);
    double lowerBin = magnitude - upperBin;
    cellHistogram->oneHundredForty += upperBin;
    cellHistogram->oneHundredTwenty += lowerBin;
  } else if (angle < 160) {
    double upperBin = magnitude * ((angle - 140) / 20);
    double lowerBin = magnitude - upperBin;
    cellHistogram->oneHundredSixty += upperBin;
    cellHistogram->oneHundredForty += lowerBin;
  } else if (angle < 180) {
    double upperBin = magnitude * ((angle - 160) / 20);
    double lowerBin = magnitude - upperBin;
    cellHistogram->zero += upperBin;
    cellHistogram->oneHundredSixty += lowerBin;
  } else {
    LOG(ERROR) << "BAD VALUE";
    LOG(ERROR) << "angle: " << angle;
  }
  return;
}

/*
 * Gets a histogram for a 8 x 8 square out of the original image
 */
HOG::histogram_t
getHistogram(const std::vector<std::vector<HOG::gradient_t>> &gradientMatrix,
             int x, int y) {
  HOG::gradient_t pixelIter;
  HOG::histogram_t cellHistogram;
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      pixelIter = gradientMatrix[8 * x + i][8 * y + j];
      addToBin(&cellHistogram, pixelIter);
    }
  }
  return cellHistogram;
}

/*
 * histogram type to a double array converter
 */
std::vector<double> histogramToDoubleArray(HOG::histogram_t *block) {
  std::vector<double> res;
  res.resize(BIN_COUNT);

  res[0] = block->zero;
  res[1] = block->twenty;
  res[2] = block->forty;
  res[3] = block->sixty;
  res[4] = block->eighty;
  res[5] = block->oneHundred;
  res[6] = block->oneHundredTwenty;
  res[7] = block->oneHundredForty;
  res[8] = block->oneHundredSixty;
  return res;
}

/*
 * concatenates 4 vectors together for the final descriptor
 */
std::vector<double> concatenateVectors(std::vector<double> block1,
                                       std::vector<double> block2,
                                       std::vector<double> block3,
                                       std::vector<double> block4,
                                       int vecSize) {
  std::vector<double> res;
  int vectorCount = 4;
  res.resize(vectorCount * BIN_COUNT);

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < BIN_COUNT; ++j) {
      if (i == 0) {
        res[j] = block1[j];
      } else if (i == 1) {
        res[BIN_COUNT + j] = block2[j];
      } else if (i == 2) {
        res[i * BIN_COUNT + j] = block3[j];
      } else if (i == 3) {
        res[i * BIN_COUNT + j] = block4[j];
      }
    }
  }
  return res;
}

/*
 * sum of a histogram block
 */
double vectorSum(HOG::histogram_t block) {
  double blockSum = 0;
  blockSum += pow(block.zero, 2);
  blockSum += pow(block.twenty, 2);
  blockSum += pow(block.forty, 2);
  blockSum += pow(block.sixty, 2);
  blockSum += pow(block.eighty, 2);
  blockSum += pow(block.oneHundred, 2);
  blockSum += pow(block.oneHundredTwenty, 2);
  blockSum += pow(block.oneHundredForty, 2);
  blockSum += pow(block.oneHundredSixty, 2);
  return blockSum;
}

/*
 * normalizes a histogram block
 */
std::vector<double> vectorNormalize(HOG::histogram_t block, double norm) {
  std::vector<double> res;
  res.resize(BIN_COUNT);

  res[0] = block.zero / norm;
  res[1] = block.twenty / norm;
  res[2] = block.forty / norm;
  res[3] = block.sixty / norm;
  res[4] = block.eighty / norm;
  res[5] = block.oneHundred / norm;
  res[6] = block.oneHundredTwenty / norm;
  res[7] = block.oneHundredForty / norm;
  res[8] = block.oneHundredSixty / norm;
  return res;
}

/*
 * Get the HOG descriptor for 64 x 128 chunk in an image of a video frame
 */
std::vector<double> getHOGForWindow(Mat img) {
  Mat grayImage; // gray scale the image
  cvtColor(img, grayImage, CV_BGR2GRAY);

  std::vector<std::vector<HOG::gradient_t>> gradientMatrix;
  gradientMatrix.resize(WINDOW_WIDTH);
  for (int i = 0; i < WINDOW_WIDTH; ++i) {
    gradientMatrix[i].resize(WINDOW_HEIGHT);
  }

  // get gradients and angles of each pixel
  for (int i = 0; i < WINDOW_WIDTH; ++i) {
    for (int j = 0; j < WINDOW_HEIGHT; ++j) {
      // here we iterate through a 64x128 chunk of an image in a video to
      // perform gradient calculations on each pixel. we are interested
      // in those with gradients
      gradientMatrix[i][j] = calculateGradient(grayImage, (double)i, (double)j);
    }
  }

  // get histograms of each 8x8 block
  std::vector<std::vector<HOG::histogram_t>> histogramMatrix;
  histogramMatrix.resize(WINDOW_WIDTH);
  for (int i = 0; i < WINDOW_WIDTH; ++i) {
    histogramMatrix[i].resize(WINDOW_HEIGHT);
  }

  HOG::histogram_t gradientHistogram;
  for (int i = 0; i < HISTOGRAM_WIDTH; ++i) {
    for (int j = 0; j < HISTOGRAM_HEIGHT; ++j) {
      histogramMatrix[i][j] = getHistogram(gradientMatrix, i, j);
    }
  }

  // histogram normalization in a 16x16 fashion
  // this helps "reduce noise" and let us keep what we care about
  std::vector<std::vector<std::vector<double>>> normalizedHistogramMatrix;
  normalizedHistogramMatrix.resize(NORMAL_BLOCK_WIDTH);
  for (int i = 0; i < NORMAL_BLOCK_WIDTH; ++i) {
    normalizedHistogramMatrix[i].resize(NORMAL_BLOCK_HEIGHT);
  }

  for (int i = 0; i < NORMAL_BLOCK_WIDTH; ++i) {
    for (int j = 0; j < NORMAL_BLOCK_HEIGHT; ++j) {

      double totalBlockSum = 0;
      HOG::histogram_t upperLeftBlock = histogramMatrix[i][j];
      HOG::histogram_t upperRightBlock = histogramMatrix[i][j + 1];
      HOG::histogram_t bottomLeftBlock = histogramMatrix[i + 1][j];
      HOG::histogram_t bottomRightBlock = histogramMatrix[i + 1][j + 1];

      totalBlockSum += vectorSum(upperLeftBlock);
      totalBlockSum += vectorSum(upperRightBlock);
      totalBlockSum += vectorSum(bottomLeftBlock);
      totalBlockSum += vectorSum(bottomRightBlock);

      double noramalDivisor = pow(totalBlockSum, 0.5);

      std::vector<double> block1 =
          vectorNormalize(upperLeftBlock, noramalDivisor);
      std::vector<double> block2 =
          vectorNormalize(upperRightBlock, noramalDivisor);
      std::vector<double> block3 =
          vectorNormalize(bottomLeftBlock, noramalDivisor);
      std::vector<double> block4 =
          vectorNormalize(bottomRightBlock, noramalDivisor);
      normalizedHistogramMatrix[i][j] =
          concatenateVectors(block1, block2, block3, block4, BIN_COUNT);
    }
  }

  std::ofstream ofs;
  ofs.open("res.txt", std::ios::app);

  std::vector<double> hogDescriptor;
  hogDescriptor.resize(HOG_SIZE);
  int vectorBlockCount = 4;
  int rowLength = vectorBlockCount * BIN_COUNT;
  // final step is to calculate the combined HOG descriptor.
  for (int i = 0; i < NORMAL_BLOCK_WIDTH; ++i) {
    for (int j = 0; j < NORMAL_BLOCK_HEIGHT; ++j) {
      std::vector<double> block = normalizedHistogramMatrix[i][j];
      for (int k = 0; k < 4; k++) {
        for (int l = 0; l < BIN_COUNT; l++) {
          // each row is 36 wide (BIN_COUNT = 9 * 4 vector blocks in the
          // noramlization phase)
          // NORMAL
          int index =
              rowLength * i + (NORMAL_BLOCK_WIDTH * rowLength) * j + k + l;
          ofs << std::setprecision(5) << (block[k * 9 + l]) << " ";
          hogDescriptor[index] = (block[k * 9 + l]);
        }
      }
    }
  }
  ofs << "\n";
  ofs.close();

  return hogDescriptor;
}

/*
 * Visualize the HOG descriptor
 * source:
 * juergenbrauer.org/old_wiki/doku.php?id=public:hog_descriptor_computation_and_visualization
 */
Mat visualize(const Mat &img, std::vector<double> &descriptor,
              const Size &size) {
  const int DIMX = size.width;
  const int DIMY = size.height;
  float zoomFac = 3;
  Mat visu;
  resize(img, visu, Size((int)(img.cols * zoomFac), (int)(img.rows * zoomFac)));

  int cellSize = 8;
  int gradientBinSize = 9;
  float radRangeForOneBin =
      (float)(CV_PI / (float)gradientBinSize); // dividing 180 into 9 bins, how
                                               // large (in rad) is one bin?

  // prepare data structure: 9 orientation / gradient strenghts for each cell
  int cells_in_x_dir = DIMX / cellSize;
  int cells_in_y_dir = DIMY / cellSize;
  float ***gradientStrengths = new float **[cells_in_y_dir];
  int **cellUpdateCounter = new int *[cells_in_y_dir];
  for (int y = 0; y < cells_in_y_dir; y++) {
    gradientStrengths[y] = new float *[cells_in_x_dir];
    cellUpdateCounter[y] = new int[cells_in_x_dir];
    for (int x = 0; x < cells_in_x_dir; x++) {
      gradientStrengths[y][x] = new float[gradientBinSize];
      cellUpdateCounter[y][x] = 0;

      for (int bin = 0; bin < gradientBinSize; bin++)
        gradientStrengths[y][x][bin] = 0.0;
    }
  }

  // nr of blocks = nr of cells - 1
  // since there is a new block on each cell (overlapping blocks!) but the
  // last one
  int blocks_in_x_dir = cells_in_x_dir - 1;
  int blocks_in_y_dir = cells_in_y_dir - 1;

  // compute gradient strengths per cell
  int descriptorDataIdx = 0;
  int cellx = 0;
  int celly = 0;

  for (int blockx = 0; blockx < blocks_in_x_dir; blockx++) {
    for (int blocky = 0; blocky < blocks_in_y_dir; blocky++) {
      // 4 cells per block ...
      for (int cellNr = 0; cellNr < 4; cellNr++) {
        // compute corresponding cell nr
        cellx = blockx;
        celly = blocky;
        if (cellNr == 1)
          celly++;
        if (cellNr == 2)
          cellx++;
        if (cellNr == 3) {
          cellx++;
          celly++;
        }

        for (int bin = 0; bin < gradientBinSize; bin++) {
          float gradientStrength = descriptor[descriptorDataIdx];
          descriptorDataIdx++;

          gradientStrengths[celly][cellx][bin] += gradientStrength;

        } // for (all bins)

        // note: overlapping blocks lead to multiple updates of this sum!
        // we therefore keep track how often a cell was updated,
        // to compute average gradient strengths
        cellUpdateCounter[celly][cellx]++;

      } // for (all cells)

    } // for (all block x pos)
  }   // for (all block y pos)

  // compute average gradient strengths
  for (celly = 0; celly < cells_in_y_dir; celly++) {
    for (cellx = 0; cellx < cells_in_x_dir; cellx++) {
      float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

      // compute average gradient strenghts for each gradient bin direction
      for (int bin = 0; bin < gradientBinSize; bin++) {
        gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
      }
    }
  }

  // draw cells
  for (celly = 0; celly < cells_in_y_dir; celly++) {
    for (cellx = 0; cellx < cells_in_x_dir; cellx++) {
      int drawX = cellx * cellSize;
      int drawY = celly * cellSize;

      int mx = drawX + cellSize / 2;
      int my = drawY + cellSize / 2;

      rectangle(visu, Point((int)(drawX * zoomFac), (int)(drawY * zoomFac)),
                Point((int)((drawX + cellSize) * zoomFac),
                      (int)((drawY + cellSize) * zoomFac)),
                Scalar(100, 100, 100), 1);

      // draw in each cell all 9 gradient strengths
      for (int bin = 0; bin < gradientBinSize; bin++) {
        float currentGradStrength = gradientStrengths[celly][cellx][bin];

        // no line to draw?
        if (currentGradStrength == 0)
          continue;

        float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

        float dirVecX = cos(currRad);
        float dirVecY = sin(currRad);
        float maxVecLen = (float)(cellSize / 2.f);
        float scale =
            2.5; // just a visualization scale, to see the lines better

        // compute line coordinates
        float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
        float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
        float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
        float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

        // draw gradient visualization
        line(visu, Point((int)(x1 * zoomFac), (int)(y1 * zoomFac)),
             Point((int)(x2 * zoomFac), (int)(y2 * zoomFac)), Scalar(0, 255, 0),
             1);

      } // for (all bins)

    } // for (cellx)
  }   // for (celly)

  // don't forget to free memory allocated by helper data structures!
  for (int y = 0; y < cells_in_y_dir; y++) {
    for (int x = 0; x < cells_in_x_dir; x++) {
      delete[] gradientStrengths[y][x];
    }
    delete[] gradientStrengths[y];
    delete[] cellUpdateCounter[y];
  }
  delete[] gradientStrengths;
  delete[] cellUpdateCounter;

  return visu;
}

/*
 * This is the main function and processes one image of a video frame
 * the input is the path to the image and '-V' for visualize
 */
int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  if ((argc == 1) || (argc > 3)) { // parse for file path
    LOG(FATAL) << "Bad Arguments\n";
  }

  LOG(INFO) << "Argv[1]: " << argv[1];
  char *imgPath = argv[1];
  Mat origImage = imread(imgPath, IMREAD_COLOR); // Read the file
  int rows = origImage.rows;
  int cols = origImage.cols;

  if (rows < WINDOW_HEIGHT | cols < WINDOW_WIDTH) {
    LOG(FATAL) << "Image is too small";
  }

  std::vector<std::vector<double>> hogRes;

  // sliding window
  int jumpFactor = 5;
  int rowIterJump = (rows - WINDOW_HEIGHT) / jumpFactor;
  int colIterJump = (cols - WINDOW_WIDTH) / jumpFactor;
  for (int i = 0; i < rows - WINDOW_HEIGHT; i += rowIterJump) {
    for (int j = 0; j < cols - WINDOW_WIDTH; j += colIterJump) {
      Rect roi = Rect(j, i, WINDOW_WIDTH, WINDOW_HEIGHT); // crop image
      Mat croppedImg = origImage(roi);
      auto hogVector = getHOGForWindow(croppedImg);
      hogRes.push_back(hogVector);
      if ((argc == 3) && (strcmp(argv[2], "-V") == 0)) {
        Mat hogVis =
            visualize(croppedImg, hogVector, Size(WINDOW_WIDTH, WINDOW_HEIGHT));
        char *buffer = (char *)malloc(strlen(argv[1]) + 1);
        strcpy(buffer, argv[1]);
        std::string fileName = buffer;
        LOG(INFO) << "Writing File: "
                  << "res-" + std::to_string(i) + "-" + std::to_string(j) +
                         fileName;
        imwrite("res-" + std::to_string(i) + "-" + std::to_string(j) + fileName,
                hogVis);
      }
    }
  }
  // hogRes can be fed to a SVM now

  return 0;
};