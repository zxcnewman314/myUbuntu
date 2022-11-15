#ifndef _REFERENCE_H_
#define _REFERENCE_H_
#include <opencv2/opencv.hpp>
using namespace cv;
void cvtGray(Mat &m1);
void satu(int contrastV,int brightV,Mat m1);
void CopyImage(Mat sour,Mat com);
void TwoValue(int control,Mat m1);
void TwoValue2(int control,Mat m1,Mat &m2);
void chooseChannel(Mat m1,int ch);
void saturability (int saturation, Mat m1);
int GetBestTH_OTSU(Mat& grayImg, int nBestTH);
void distanceTrans(Mat srcImage, Mat dstImage);
void spicedSalt(Mat &src);
double generateGaussianNoise(double mu, double sigma);
void  addGaussianNoise(Mat& srcImage);
void AverageWave(Mat &src,int n);
void swap(uchar &n1,uchar &n2);
uchar MediumValue(uchar* p,int n);
void MediumWave(Mat &src,int n);
void Random_noise(Mat &src);
Mat GuassTemplate(Mat &src,int n);
void gray2pseudocolor1(Mat& Image);
void gray2pseudocolor2(Mat& Image);
void  gray2pseudocolor3(Mat& Image);
void ZoomOutEqualInterval(Mat& src, double x_k,double y_k);
void ZoomOutLocalMean(Mat& src, double x_k, double y_k);
void ZoomInNearestNeighborInterpolation(Mat& src, double x_k, double y_k);
void ZoomInBilinearInterpolation(Mat& src, double x_k, double y_k);
int isAllSame(uchar *p,int n);
uchar MaxValue(uchar *p,int n);
uchar MinimumValue(uchar* p,int n);
void LaplaceSharpen_1(Mat &src);
void LaplaceSharpen_2(Mat &src);
void SobelSharpen_1(Mat &src);
void TwoValueErodeImage_1(Mat &src,int n);
Mat TwoValueErodeImage_2(Mat &src,int n);
void TwoValueErodeImage_3(Mat &src,int n);
void TwoValueDilateImage_1(Mat &src,int n);
Mat TwoValueDilateImage_2(Mat &src,int n);
void TwoValueDilateImage_3(Mat &src,int n);
void GrayErodeImage_1(Mat &src,int n);
void CannyEdge(Mat &img);
void UsetoRotateImage(Mat &src,Mat &dst,double angle);
void realizeCanny(Mat &img);
void showCanny(Mat &img);
#endif
