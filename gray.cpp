#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "repo.h"
#include <stdlib.h>
#include <unistd.h>
using namespace cv;
int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: gray.out <Image_Path>\n");
        return -1;
    }
    Mat image;
    int BrightV=80;
    int ContrastV=80;
    int words=0;
    //image = imread( argv[1],IMREAD_GRAYSCALE);
    image = imread( argv[1]);
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    printf("input 1 to use grayErodeImage method.input 2 to use sharpen method.input 3 to use give edge method.\n");
    scanf("%d",&words);
    if(words==1){
     GrayErodeImage_1(image,7);
    }
    else if(words==2){
    printf("input 1 to use Laplace positive corner template,input 2 to use Laplace opposite angel method.\n");
    printf("input 3 to use sobel method.\n");
    scanf("%d",&words);
    if(words==1)
     LaplaceSharpen_1(image);
    else if(words==2)
	     LaplaceSharpen_2(image);
    else if (words==3)
	     SobelSharpen_1(image);
    }
    else if (words==3)
	     CannyEdge( image);
    printf("input 1 to use pseudo color method 1,input 2 to use pseudo color method 2, input 3 to use pseudo method 3.\n");
     scanf("%d",&words);
   if(words<4&&words>0){
         Mat imgg=image.clone();
	 cvtGray(imgg);
	   if(words==1)
		   gray2pseudocolor1(imgg);

      else if(words==2)
	      gray2pseudocolor2(imgg);
	   else
		   gray2pseudocolor3(imgg);
   }
   printf("input 1 to erode Picture ,input 2 to dilate picture.input 3 to rotate image.\n");
    scanf("%d",&words);
   if(words==1){
printf("input 1 to use TwoValueErodeImage_8Neigbor method.Input 2 to use cross method.Input 3 to use Lower right method.\n");
 scanf("%d",&words);
   if(words==1)
  TwoValueErodeImage_1(image,5);
   else if(words==2)
	   TwoValueErodeImage_2(image,5);
   else if(words==3)
	   TwoValueErodeImage_3(image,5);
   }
   else if(words==2){
	   printf("input 1 to use TwoValueDilateImage_8Neigbor method.Input 2 to use cross method.Input 3 to use Lower right method.\n");
scanf("%d",&words);
   if(words==1)
  TwoValueDilateImage_1(image,5);
    else if(words==2)
           TwoValueDilateImage_2(image,5);
   else if(words==3)
           TwoValueDilateImage_3(image,5);

   }
   else if(words==3){
      printf("input the angle  you want the picture to rotate.input 0 to exit.\n");
    double angle=0;
    Mat dst=image.clone();
    scanf("%lf",&angle);
    if(angle!=0)
      UsetoRotateImage(image,dst, angle); 
   }
   printf("input 1 to zoom out pics, input 2 to zoom in.\n");
  scanf("%d",&words);
   if(words==1){
    printf("input 1 to use local mean method, input 2 to use equal interval method.And you also need to input x_k and y_k\n");
     double lx=0;
     double ly=0;
      scanf("%d%lf%lf",&words,&lx,&ly);
     if(words==1)
	     ZoomOutLocalMean(image,lx,ly);
     else if(words==2)
	     ZoomOutEqualInterval(image,lx,ly);
   }
   else if(words==2){
     printf("input 1 to use NearestNeighborInterpolation method, input 2 to use BilinearInterpolation method.And you also need to input x_k and y_k\n");
      double lx=0;
     double ly=0;
      scanf("%d%lf%lf",&words,&lx,&ly);
       if(words==1)
	       ZoomInNearestNeighborInterpolation(image,lx,ly);
       else if(words==2)
	       ZoomInBilinearInterpolation(image,lx,ly);
   }
    printf("input 1 to add guass noise.input 2 to add spiced salt noise in the picture.input 3 to add random noise in the picture.\n");
     scanf("%d",&words);
   if(words==1){
     addGaussianNoise(image);
   }
   else if(words==2){
      spicedSalt(image);
   }
   else if(words==3)
   { Random_noise(image);
   }
  printf("input 1 to use guass template.input 2 to use medium method.input 3 to use averag method.\n");
  scanf("%d",&words);
   if(words==1)
  {GuassTemplate(image,3);
}
else if(words==2){
     MediumWave(image,3);
   }
else  if(words==3){
     AverageWave(image,3);
   }
    printf("do you want to change the picture's contrast value and bright value? if so,please input 1 .input 2 to use canny method.\n");
   scanf("%d",&words);
   if(words==1){
	   Mat img1=Mat::zeros(image.size(),image.type());
	   CopyImage(image,img1);
    printf("please input the Contrast Value and Bright Value.\n");
    scanf("%d%d",&BrightV,&ContrastV);
    satu(BrightV,ContrastV,img1);
   }
   else if(words==2){
     // realizeCanny(image);
    showCanny(image);    
   }
   printf("do you want to make distance transform in this picture?if so ,input 1.\n");
    scanf("%d",&words);
     if(words==1){
    Mat imaget=Mat::zeros(image.size(),image.type());
        CopyImage(image,imaget);
	    distanceTrans( image,imaget);
    
     }
    //imshow(argv[1], image) ;
   printf("do you want to change color to gray ? if so, input 1 else to skip.\n");
     scanf("%d",&words);
     if(words==1){
   Mat image2=Mat::zeros(image.size(),image.type());
    image2=image.clone();
    cvtGray(image);}
     printf("do you want to choose the picture's channel? if so,0 is blue,1 is green,2 is red.And the other to skip.\n");
     scanf("%d",&words);
     if(words>=0&&words<3){
	     Mat img2=Mat::zeros(image.size(),image.type());
           CopyImage(image,img2);
	     chooseChannel( img2,words);
     }
     printf("do you want to make the picture in 2 value?if so,input 1.\n");
     scanf("%d",&words);
     if(words==1){
   printf("input the tap value.\n");
   int tap=0;
   scanf("%d",&tap);
   TwoValue(tap,image);}
     printf("do you want to change the picture's saturation? if so, input 1.\n");
      scanf("%d",&words);
     if(words==1){
     int satu=0;
     printf("input the saturation value.\n");
     scanf("%d",&satu);
   saturability ( satu, image);}
    printf("input 1 to change picture's best 2 value.\n");
      scanf("%d",&words);
     if(words==1){
     int best=0; 
   best=  GetBestTH_OTSU(image, best);
   TwoValue(best,image);}
    waitKey(0);
  destroyAllWindows();
    return 0;
}
