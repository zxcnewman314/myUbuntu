#include<stdio.h>
#include <opencv2/opencv.hpp>
#include "repo.h"
#include "math.h"
#include <cstdlib>
#include <limits>
#include <iostream>
#include <cmath>
#include <numeric>   // std::accumulate
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
using namespace cv;
using namespace std;
#define max(a, b) ( ((a) > (b))  ? ( a) : (b))
#define max3(a,b,c)   a > b  ?  max( a, c) : max( b, c ) 

#define mmin(a, b)   ( ((a) < (b))  ? ( a) : (b))
#define min3(a,b,c)   a < b  ?  mmin( a, c ) : mmin( b, c ) 
const int max_Increment = 200;
int Increment_value;
void swap(uchar &n1,uchar &n2){
   uchar n3=n2;
  n2=n1;
   n1=n3;
   
}
double generateGaussianNoise(double mu, double sigma)
{
	//定义一个特别小的值
	const double epsilon = std::numeric_limits<double>::min();//返回目标数据类型能表示的最逼近1的正数和1的差的绝对值
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	//flag为假，构造高斯随机变量
	if (!flag)
		return z1*sigma + mu;
	double u1, u2;
	//构造随机变量

	do
	{
		u1 = rand()*(1.0 / RAND_MAX);
		u2 = rand()*(1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	//flag为真构造高斯随机变量X
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI * u2);
	z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI * u2);
	return z1*sigma + mu;
}
//为图像添加高斯噪声
void  addGaussianNoise(Mat& srcImage)
{
	Mat resultImage = srcImage.clone();    //深拷贝,克隆
	int channels = resultImage.channels();    //获取图像的通道
	int nRows = resultImage.rows;    //图像的行数

	int nCols = resultImage.cols*channels;   //图像的总列数
	//判断图像的连续性
	if (resultImage.isContinuous())    //判断矩阵是否连续，若连续，我们相当于只需要遍历一个一维数组
	{
		nCols *= nRows;
		nRows = 1;
	}
	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nCols; j++)
		{	//添加高斯噪声
			int val = resultImage.ptr<uchar>(i)[j] + generateGaussianNoise(4, 0.8) * 32;
			if (val < 0)
				val = 0;
			if (val > 255)
				val = 255;
			resultImage.ptr<uchar>(i)[j] = (uchar)val;
		}
	}
	 namedWindow("srcImage", WINDOW_FREERATIO);
   imshow("srcImage",resultImage);

}
uchar MediumValue(uchar* p,int n){
  if(n<3||n%2==0)
	  return 1;
  int divide=(n-1)/2;
  for(int gap=divide;gap>0;gap/=2){
     for(int i=gap;i<n;++i)
	 for(int j=i-gap;j>=0&&p[j]>p[j+gap];j=j-gap){
                 swap(p[j],p[j+gap]);
	 }
  }
  return p[divide];
}
uchar MinimumValue(uchar* p,int n){
  if(n<3||n%2==0)
          return 1;
  int divide=(n-1)/2;
  for(int gap=divide;gap>0;gap/=2){
     for(int i=gap;i<n;++i)
         for(int j=i-gap;j>=0&&p[j]>p[j+gap];j=j-gap){
                 swap(p[j],p[j+gap]);
         }
  }
  return p[0];
}
uchar MaxValue(uchar *p,int n){
  if(n<3||n%2==0)
          return 1;
  int divide=(n-1)/2;
  for(int gap=divide;gap>0;gap/=2){
     for(int i=gap;i<n;++i)
         for(int j=i-gap;j>=0&&p[j]>p[j+gap];j=j-gap){
                 swap(p[j],p[j+gap]);
         }
  }
  return p[n-1];

}
int isAllSame(uchar *p,int n){
int result=1;
 for (int i=0;i<n-1;i++)
	 if(p[i]!=p[i+1])
      result=0;
 return result;
}
void MediumWave(Mat &src,int n){
	Mat dst=src.clone();
        if(!src.data||n%2==0)
           return;
   int n1=(n-1)/2;
  int  h=src.rows;
  int  w=src.cols;
   for(int i=0;i<h;i++)
     for(int j=0;j<w;j++){
        for(int k=0;k<3;k++){
           int li=i-n1;
           int mi=i+n1;
           int lj=j-n1;
           int mj=j+n1;
	   uchar *p;
	   int count=0;
                if(li>=0&&lj>=0&&mi<h&&mj<w)
            {      for(int i1=li;i1<mi+1;i1++)
                    for(int j1=lj;j1<mj+1;j1++)
                   p[count++]=src.at<Vec3b>(i1,j1)[k];
		     dst.at<Vec3b>(i,j)[k]=MediumValue(p,n*n);
	    }
		  else
                dst.at<Vec3b>(i,j)[k]=src.at<Vec3b>(i,j)[k];
        }}
         namedWindow("dst", WINDOW_FREERATIO);
   imshow("dst",dst);

}
void AverageWave(Mat &src,int n){
  Mat dst=src.clone();
       	if(!src.data||n%2==0)
	   return;
   int n1=(n-1)/2;
  int  h=src.rows;
  int  w=src.cols;
   for(int i=0;i<h;i++)
     for(int j=0;j<w;j++){
        for(int k=0;k<3;k++){
           int li=i-n1;
	   int mi=i+n1;
	   int lj=j-n1;
	   int mj=j+n1;
	       	if(li>=0&&lj>=0&&mi<h&&mj<w)
	    {  int count=0;
		    for(int i1=li;i1<mi+1;i1++)
		    for(int j1=lj;j1<mj+1;j1++)
                        count+=src.at<Vec3b>(i1,j1)[k];
		    dst.at<Vec3b>(i,j)[k]=  count/(n*n);
	         
	    }
	    else
                dst.at<Vec3b>(i,j)[k]=src.at<Vec3b>(i,j)[k];
	}
     }
     namedWindow("dst", WINDOW_FREERATIO);
   imshow("dst",dst);

}
Mat GuassTemplate(Mat &src,int n){
   Mat dst=src.clone();
        if(!src.data||n%2==0)
           return src;
   int n1=(n-1)/2;
  int  h=src.rows;
  int  w=src.cols;
   for(int i=0;i<h;i++)
     for(int j=0;j<w;j++){
        for(int k=0;k<3;k++){
           int li=i-n1;
           int mi=i+n1;
           int lj=j-n1;
           int mj=j+n1;
           uchar *p;
           int count=0;
                if(li>=0&&lj>=0&&mi<h&&mj<w)
            {      for(int i1=li;i1<mi+1;i1++)
                    for(int j1=lj;j1<mj+1;j1++)
		    { if(i1==i&&j1==j)   count+=4*src.at<Vec3b>(i1,j1)[k];
		      else if (i1!=i&&j1!=j)
                            count+=src.at<Vec3b>(i1,j1)[k];
                       else
			        count+=2*src.at<Vec3b>(i1,j1)[k];

		    }
                     dst.at<Vec3b>(i,j)[k]=count/16;
            }
                  else
                dst.at<Vec3b>(i,j)[k]=src.at<Vec3b>(i,j)[k];
        }}
         namedWindow("dst", WINDOW_FREERATIO);
   imshow("dst",dst);
    return dst;
}
void spicedSalt(Mat &src){
   RNG rng(12345);
   int h=src.rows;
   int w=src.cols;
   int nums=30000;
   for(int i=0;i<nums;i++){
      int x= rng.uniform(0,w);
     int y=rng.uniform(0,h);
	   if(i%2==0)
     src.at<Vec3b>(y,x)=Vec3b(0,0,0);
	   else
     src.at<Vec3b>(y,x)=Vec3b(255,255,255);
     
   }
    namedWindow("src", WINDOW_FREERATIO);
   imshow("src",src);

}
void Random_noise(Mat &src){
   RNG rng(12345);
   int h=src.rows;
   int w=src.cols;
   int nums=30000;
   for(int i=0;i<nums;i++){
      int x= rng.uniform(0,w);
     int y=rng.uniform(0,h);
       int co=rng.uniform(0,255);  
     src.at<Vec3b>(y,x)=Vec3b(co,co,co);

   }
    namedWindow("src", WINDOW_FREERATIO);
   imshow("src",src);

}
void LaplaceSharpen_1(Mat &src){//positive corner
 Mat dst=src.clone();
  int h=src.rows;
   int w=src.cols;
  for(int i=0;i<h;i++)
     for(int j=0;j<w;j++)
        for(int k=0;k<3;k++){
       uchar  	   sum=0;
		if(i>0&&j>0&&i+1<h&&j+1<w)
		{ for(int li=i-1;li<=i+1;li++)
		    sum-=src.at<Vec3b>(li,j)[k];
                   for(int lj=j-1;lj<=j+1;lj++)
                    sum-=src.at<Vec3b>(i,lj)[k];
		   sum+=7*src.at<Vec3b>(i,j)[k];
		   dst.at<Vec3b>(i,j)[k]=sum;
		}
	}
     namedWindow("dst", WINDOW_FREERATIO);
   imshow("dst",dst);
}
void LaplaceSharpen_2(Mat &src){//opposite angle
 Mat dst=src.clone();
  int h=src.rows;
   int w=src.cols;
  for(int i=0;i<h;i++)
     for(int j=0;j<w;j++)
        for(int k=0;k<3;k++){
          uchar  sum=0;
                if(i>0&&j>0&&i+1<h&&j+1<w)
                { for(int li=i-1;li<=i+1;li++)
                   for(int lj=j-1;lj<=j+1;lj++)
                    sum-=src.at<Vec3b>(li,lj)[k];
                   sum+=11*src.at<Vec3b>(i,j)[k];
                   dst.at<Vec3b>(i,j)[k]=sum;
                }
        }
     namedWindow("dst", WINDOW_FREERATIO);
   imshow("dst",dst);
}
void SobelSharpen_1(Mat &src){
    Mat dst=src.clone();
  int h=src.rows;
   int w=src.cols;
  for(int i=0;i<h;i++)
     for(int j=0;j<w;j++)
        for(int k=0;k<3;k++){
        int li=i-1;
           int mi=i+1;
           int lj=j-1;
           int mj=j+1;
           uchar sumx=0;
	   uchar sumy=0;
                if(li>=0&&lj>=0&&mi<h&&mj<w)
           { sumx+=-src.at<Vec3b>(li,lj)[k]-2*src.at<Vec3b>(i,lj)[k]-src.at<Vec3b>(mi,lj)[k]+src.at<Vec3b>(li,mj)[k]+2*src.at<Vec3b>(i,mj)[k]+src.at<Vec3b>(mi,mj)[k];
             sumy+=-src.at<Vec3b>(li,lj)[k]-2*src.at<Vec3b>(li,j)[k]-src.at<Vec3b>(li,mj)[k]+src.at<Vec3b>(mi,lj)[k]+2*src.at<Vec3b>(mi,j)[k]+src.at<Vec3b>(mi,mj)[k];
	     uchar sum=sqrt(sumx*sumx+sumy*sumy);
                   sum+=src.at<Vec3b>(i,j)[k];
	        dst.at<Vec3b>(i,j)[k]=sum;
	   }
     	}
      namedWindow("dst", WINDOW_FREERATIO);
   imshow("dst",dst);

}
float calcEuclideanDiatance(int x1, int y1, int x2, int y2)  
{  
    return sqrt(float((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1)));  
}  
int calcChessboardDistance(int x1, int y1, int x2, int y2)  
{  
    return max(abs(x1 - x2), abs(y1 - y2));  
}  
int calcBlockDistance(int x1, int y1, int x2, int y2)
{
    return abs(x1 - x2) + abs(y1 - y2);
}
void distanceTrans(Mat srcImage, Mat dstImage)
{
	   if (!srcImage.data || srcImage.data == NULL)
        {
                printf("error!\n");
                return;
        }

    //定义灰度图像的二值图像
    Mat  binaryImage=srcImage.clone();
    imshow("binary",binaryImage);
    //将灰度图像转换为二值图像
    int best=0;
   best=  GetBestTH_OTSU(binaryImage, best);
   TwoValue(best,binaryImage);
    imshow("my two value", binaryImage);

    int rows = binaryImage.rows;
    int cols = binaryImage.cols;
    uchar *pDataOne;
    uchar *pDataTwo;
    float disPara = 0;
    float fDisMIn = 0;

    //第一遍遍历图像，使用左模板更新像素值
    for (int i = 1; i < rows - 1; i++)
    {
        //图像的行指针的获取
        pDataOne = binaryImage.ptr<uchar>(i);
        for (int j = 1; j < cols; j++)
        {
            //分别计算左模板掩码的相关距离
            //PL  PL
            //PL  P
            //PL
            pDataTwo = binaryImage.ptr<uchar>(i - 1);
            disPara = calcEuclideanDiatance(i, j, i - 1, j - 1);
            fDisMIn = mmin((float)pDataOne[j], disPara + pDataTwo[j - 1]);
            disPara = calcEuclideanDiatance(i, j, i - 1, j);
            fDisMIn = mmin(fDisMIn, disPara + pDataTwo[j]);
            pDataTwo = binaryImage.ptr<uchar>(i);
            disPara = calcEuclideanDiatance(i, j, i, j - 1);
            fDisMIn = mmin(fDisMIn, disPara + pDataTwo[j-1]);
            pDataTwo = binaryImage.ptr<uchar>(i+1);
            disPara = calcEuclideanDiatance(i, j, i+1,j-1);
            fDisMIn = mmin(fDisMIn, disPara + pDataTwo[j - 1]);
            pDataOne[j] = (uchar)cvRound(fDisMIn);
        }
    }

    //第二遍遍历图像，使用右模板更新像素值
    for (int i = rows - 2; i > 0; i--)
    {
        pDataOne = binaryImage.ptr<uchar>(i);
        for (int j = cols - 2; j >= 0; j--)
        {
            //分别计算右模板掩码的相关距离
            //pR  pR
            //pR  p
            //pR
            pDataTwo = binaryImage.ptr<uchar>(i + 1);
            disPara = calcEuclideanDiatance(i, j, i + 1, j);
            fDisMIn = mmin((float)pDataOne[j], disPara + pDataTwo[j]);
            disPara = calcEuclideanDiatance(i, j, i + 1, j + 1);

            fDisMIn = min(fDisMIn, disPara + pDataTwo[j+1]);
            pDataTwo = binaryImage.ptr<uchar>(i);
            disPara = calcEuclideanDiatance(i, j, i, j +1);
            fDisMIn = mmin(fDisMIn, disPara + pDataTwo[j + 1]);
            pDataTwo = binaryImage.ptr<uchar>(i - 1);
            disPara = calcEuclideanDiatance(i, j, i - 1, j + 1);
            fDisMIn = mmin(fDisMIn, disPara + pDataTwo[j + 1]);
            pDataOne[j] = (uchar)cvRound(fDisMIn);
        }
    } 
    dstImage=binaryImage.clone();
      imshow("dstImage",dstImage);
}
void cvtGray(Mat &m1){
	int col=m1.cols;
	int row=m1.rows;
	int b=0;
	int g=0;
	int r=0;
	for(int i=0;i<row;i++)
	{
	for(int j=0;j<col;j++){
	Vec3b pixel =m1.at<Vec3b>(i,j);
	b=pixel[0];
	g=pixel[1];
	r=pixel[2];
        m1.at<Vec3b>(i,j)[0]=b*0.114+g*0.587+r*0.299;
	m1.at<Vec3b>(i,j)[1]=b*0.114+g*0.587+r*0.299;
	m1.at<Vec3b>(i,j)[2]=b*0.114+g*0.587+r*0.299;
	}
	}
	 namedWindow("cvtGray", WINDOW_FREERATIO);
	imshow("cvtGray",m1);
}
void chooseChannel(Mat m1,int ch){
	int col=m1.cols;
	int row=m1.rows;
	Mat img2=m1.clone();
          for(int i=0;i<row;i++)
        {
        for(int j=0;j<col;j++){
		for(int k=0;k<3;k++)
	  if(k!=ch)
           img2.at<Vec3b>(i,j)[k]=0;
	}
	}
	   namedWindow("chooseChannel", WINDOW_FREERATIO);
	   imshow("chooseChannel",img2);

}
void satu(int contrastV,int brightV,Mat m1){
	
	int col=m1.cols;
	int row=m1.rows;
	int b=0;
	int g=0;
	int r=0;
	for(int i=0;i<row;i++)
	{
	for(int j=0;j<col;j++){
	Vec3b pixel =m1.at<Vec3b>(i,j);
	b=pixel[0];
	g=pixel[1];
	r=pixel[2];
        m1.at<Vec3b>(i,j)[0]=saturate_cast<uchar>(contrastV*b+brightV);
	m1.at<Vec3b>(i,j)[1]=saturate_cast<uchar>(contrastV*g+brightV);
	m1.at<Vec3b>(i,j)[2]=saturate_cast<uchar>(contrastV*r+brightV);
	}
	}
	 namedWindow("satu", WINDOW_FREERATIO);
	imshow("satu",m1);
}
void CopyImage(Mat sour,Mat com){
	int row=sour.rows;

	int col=sour.cols;
	int ch=sour.channels();
	int b=0;
	int g=0;
	int r=0;
	if(ch!=3)
	return;
	for(int i=0;i<row;i++){
          uchar *p1=sour.ptr<uchar>(i);
        uchar *p2=com.ptr<uchar>(i);
	for(int j=0;j<col;j++){
        b=*p1++; g=*p1++; r=*p1++;
	*p2++=b;
        *p2++=g;
        *p2++=r;
        }
	 }

}
void TwoValue(int control,Mat m1){
  cvtGray(m1);
   int row=m1.rows;
   int col=m1.cols;
   int max=0;
   int min=0;
   int count=0;
   for(int i=0;i<row;i++)
	   for(int j=0;j<col;j++){
        count= m1.at<Vec3b>(i,j)[0];
	if(count>max)
		max=count;
	if(count<min)
		min=count;
}
   if(control<min||control>max)
       control=(min+max)/2;	
for(int i=0;i<row;i++)
           for(int j=0;j<col;j++){
	         count= m1.at<Vec3b>(i,j)[0];
                if(count>control)
		for(int k=0;k<3;k++)
			m1.at<Vec3b>(i,j)[k]=max;
		else 
			 for(int k=0;k<3;k++)
                        m1.at<Vec3b>(i,j)[k]=0;
	  
	   }
     namedWindow("2Values", WINDOW_FREERATIO);
        imshow("2Values",m1);
  }
void TwoValue2(int control,Mat m1,Mat &m2){
  cvtGray(m1);
   int row=m1.rows;
   int col=m1.cols;
   int max=0;
   int min=0;
   int count=0;
   for(int i=0;i<row;i++)
           for(int j=0;j<col;j++){
        count= m1.at<Vec3b>(i,j)[0];
        if(count>max)
                max=count;
        if(count<min)
                min=count;
}
   if(control<min||control>max)
       control=(min+max)/2;
for(int i=0;i<row;i++)
           for(int j=0;j<col;j++){
                 count= m1.at<Vec3b>(i,j)[0];
                if(count>control)
                for(int k=0;k<3;k++)
                        m1.at<Vec3b>(i,j)[k]=max;
                else
                         for(int k=0;k<3;k++)
                        m1.at<Vec3b>(i,j)[k]=0;

           }
     namedWindow("2Values", WINDOW_FREERATIO);
        imshow("2Values",m1);
      m2= m1.clone();
  }

void saturability (int saturation, Mat m1)
{     Mat m2=Mat::zeros(m1.size(),m1.type());
	float increment = (saturation - 80) * 1.0 / max_Increment;
	for (int col = 0; col < m1.cols; col++)
	{
		for (int row = 0; row < m1.rows; row++)
		{
			// R,G,B 分别对应数组中下标的 2,1,0
			uchar r = m1.at<Vec3b> (row, col)[2];		
			uchar g = m1.at<Vec3b> (row, col)[1];
			uchar b = m1.at<Vec3b> (row, col)[0];

			float maxn = max (r, max (g, b));
			float minn = min (r, min (g, b));

			float delta, value;
			delta = (maxn - minn) / 255;
			value = (maxn + minn) / 255;

			float new_r, new_g, new_b;

			if (delta == 0)		 // 差为 0 不做操作，保存原像素点
			{
				m2.at<Vec3b> (row, col)[0] = new_b;
				m2.at<Vec3b> (row, col)[1] = new_g;
				m2.at<Vec3b> (row, col)[2] = new_r;
				continue;
			}

			float light, sat, alpha;
			light = value / 2;

			if (light < 0.5)
				sat = delta / value;
			else
				sat = delta / (2 - value);

			if (increment >= 0)
			{
				if ((increment + sat) >= 1)
					alpha = sat;
				else
				{
					alpha = 1 - increment;
				}
				alpha = 1 / alpha - 1;
				new_r = r + (r - light * 255) * alpha;
				new_g = g + (g - light * 255) * alpha;
				new_b = b + (b - light * 255) * alpha;
			}
			else
			{
				alpha = increment;
				new_r = light * 255 + (r - light * 255) * (1 + alpha);
				new_g = light * 255 + (g - light * 255) * (1 + alpha);
				new_b = light * 255 + (b - light * 255) * (1 + alpha);
			}
	                          m2.at<Vec3b> (row, col)[0] = new_b;
                                m2.at<Vec3b> (row, col)[1] = new_g;
                                m2.at<Vec3b> (row, col)[2] = new_r;

		}
	}
	 namedWindow("saturibility", WINDOW_FREERATIO);
	imshow ("saturability", m2);
}
int GetBestTH_OTSU(Mat& grayImg, int nBestTH)
{       int returnValue=-1;
	//【1】安全性检查
	if (!grayImg.data || grayImg.data == NULL)
	{
		printf("error!\n");
		return returnValue;
	}
 
	//【2】参数准备
	double sum = 0.0;			//所有像素灰度之和
	double w0 = 0.0;			//背景像素所占比例
	double w1 = 0.0;			//前景像素所占比例
	double u0_temp = 0.0;
	double u1_temp = 0.0;
	double u0 = 0.0;			//背景平均灰度
	double u1 = 0.0;			//前景平均灰度
	double delta_temp = 0.0;	//类间方差
	double delta_max = 0.0;		//最大类间方差
	const int GrayScale = 256;
 
	//src_image灰度级  
	int pixel_count[GrayScale] = { 0 };		//每个灰度级的像素数目
	float pixel_pro[GrayScale] = { 0 };		//每个灰度级的像素数目占整幅图像的比例  
 
	int height = grayImg.rows;
	int width = grayImg.cols;
	//统计每个灰度级中像素的个数  
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int index = i * width + j;
			pixel_count[(int)grayImg.data[index]]++;		//每个灰度级的像素数目
			sum += (int)grayImg.data[index];				//灰度之和
		}
	}
    printf("average gray value:%f\n",sum / (height * width));
 
	//计算每个灰度级的像素数目占整幅图像的比例  
	int imgArea = height * width;
	for (int i = 0; i < GrayScale; i++)
	{
		pixel_pro[i] = (float)pixel_count[i] / imgArea;
	}
 
	//遍历灰度级[0,255],寻找合适的threshold  
	for (int i = 0; i < GrayScale; i++)
	{
		w0 = w1 = u0_temp = u1_temp = u0 = u1 = delta_temp = 0;
		for (int j = 0; j < GrayScale; j++)
		{
			if (j <= i)   //背景部分  
			{
				w0 += pixel_pro[j];			//背景像素比例
				u0_temp += j * pixel_pro[j];
			}
			else		 //前景部分  
			{
				w1 += pixel_pro[j];			//前景像素比例
				u1_temp += j * pixel_pro[j];
			}
		}
		u0 = u0_temp / w0;		//背景像素点的平均灰度
		u1 = u1_temp / w1;		//前景像素点的平均灰度
 
		delta_temp = (float)(w0 * w1 * pow((u0 - u1), 2));		//类间方差 g=w0*w1*(u0-u1)^2
 
		//当类间方差delta_temp最大时，对应的i就是阈值T
		if (delta_temp > delta_max)
		{
			delta_max = delta_temp;
			nBestTH = i;
		}
	}
        returnValue=1; 
	return nBestTH;
}
void  gray2pseudocolor1(Mat& Image) {

		Mat pseudocolor=Image.clone();
		unsigned char grayValue; //用来记录的当前像素的灰度
		for (int r = 0; r < Image.rows; r++) {
			for (int c = 0; c < Image.cols; c++) {
				grayValue = Image.at<Vec3b>(r, c)[0];
				//对每个像素进行灰度变换
				Vec3b& pixel = pseudocolor.at<Vec3b>(r, c);
				pixel[0] = abs(255 - grayValue);
				pixel[1] = abs(127 - grayValue);
				pixel[2] = abs(0 - grayValue);
			}
		}

	    namedWindow("pseudocolor", WINDOW_FREERATIO);
        imshow ("pseudocolor", pseudocolor);
	}

	//gray2pseudocolor
	//灰度级-彩色变换法2
void  gray2pseudocolor2(Mat& Image) {

                      Mat pseudocolor=Image.clone();
		unsigned char grayValue; //用来记录的当前像素的灰度
		for (int r = 0; r < Image.rows; r++) {
			for (int c = 0; c < Image.cols; c++) {
				grayValue = Image.at<Vec3b>(r, c)[0];
				//对每个像素进行灰度变换
				Vec3b& pixel = pseudocolor.at<Vec3b>(r, c);
				if (grayValue <= 63) {
					pixel[0] = 255; //B
					pixel[1] = 254 - 4 * grayValue; //G
					pixel[2] = 0; //R
				}
				else if (grayValue <= 127) {
					pixel[0] = 510 - 4 * grayValue; //B
					pixel[1] = 4 * grayValue - 254; //G
					pixel[2] = 0; //R
				}
				else if (grayValue <= 191) {
					pixel[0] = 0; //B
					pixel[1] = 255; //G
					pixel[2] = 4 * grayValue - 510; //R
				}
				else {
					pixel[0] = 0; //B
					pixel[1] = 1022 - 4 * grayValue; //G
					pixel[2] = 255; //R
				}
			}
		}
           
		  namedWindow("pseudocolor", WINDOW_FREERATIO);
        imshow ("pseudocolor", pseudocolor);

	}
void  gray2pseudocolor3(Mat& Image){
     Mat pseudocolor=Image.clone();
     int row=Image.rows;
     int col=Image.cols;
     for(int r=0;r<row;r++)
     {  
	     for(int c=0;c<col;c++)
         {
		 unsigned char grayValue = Image.at<Vec3b>(r, c)[0];
                                //对每个像素进行灰度变换
                                Vec3b& pixel = pseudocolor.at<Vec3b>(r, c);
                                if (grayValue <= 51) {
                                        pixel[2] = 255; //B
                                        pixel[1] = 5 * grayValue; //G
                                        pixel[0] = 0; //R
                                }
                                else if (grayValue <= 102) {
                                        pixel[2] = 510-5*grayValue; //B
                                        pixel[1] = 255; //G
                                        pixel[0] = 0; //R
                                }
                                else if (grayValue <= 153) {
                                        pixel[2] = 5* grayValue - 510;
                                        pixel[1] = 255; //G
                                        pixel[0] = 510-4*grayValue; //R
                                }
                                else if(grayValue<=204){
                                        pixel[2] = 0; //B
                                        pixel[1] = 1022 - 5 * grayValue; //G
                                        pixel[0] = 255; //R
                                }
				else {
					 pixel[2] = 0; //B
                                        pixel[1] = 1275 - 5 * grayValue; //G
                                        pixel[0] = 5*grayValue-1022; //R
				}

	 }
     }
      namedWindow("pseudocolor", WINDOW_FREERATIO);
        imshow ("pseudocolor", pseudocolor);

}
void ZoomOutEqualInterval(Mat& src, double x_k,double y_k)
{
   namedWindow("src",1);
   imshow("src",src);
     	int i0=src.rows;
   int i1=src.cols;
     printf("Inputpicture's row is %d,col is %d\n",i0,i1);
    Mat dst(src.rows*y_k, src.cols*x_k, src.type(), Scalar(0));
    x_k = 1 / x_k;//将缩小率转换为采样间隔
    y_k = 1 / y_k;
    if (src.channels() == 3)
    {
        for (int i = 0; i < dst.rows; i++)
        {
            for (int j = 0; j < dst.cols; j++)
            {

                int x = j * x_k + 0.5;
                int y = i * y_k + 0.5;
                if (x >= src.cols)x = src.cols - 1;
                if (y >= src.rows)y = src.rows - 1;
                dst.at<Vec3b>(i, j) = src.at<Vec3b>(y, x);
            }
        }
    }
    int a1=dst.rows;
    int a2=dst.cols;
    printf("Outputpicture's row is %d,col is %d\n",a1,a2);
     namedWindow("ZoomOut", 1);
//     cvResizeWindow("ZoomOut",500,500);
        imshow ("ZoomOut", dst);

}
void ZoomOutLocalMean(Mat& src, double x_k, double y_k)
{    namedWindow("src",1);
   imshow("src",src);

     int i0=src.rows;
   int i1=src.cols;
     printf("Inputpicture's row is %d,col is %d\n",i0,i1);

    Mat dst(src.rows*y_k, src.cols*x_k, src.type(), Scalar(0));
    x_k = 1 / x_k;//将缩小率转换为采样间隔
    y_k = 1 / y_k;
     cout << "dst x y:" << dst.cols << " " << dst.rows << endl;
    cout << "src x y:" <<src.cols << " " << src.rows << endl;
    if (src.channels() == 3)
    {
        for (int i = 0; i < dst.rows; i++)
        {
            for (int j = 0; j < dst.cols; j++)
            {
                int j_start = (j-1) * x_k+1;
                if (j_start < 0)j_start = 0;

                int j_end = j* x_k;
                if (j_end >=src.cols )j_end = src.cols;
                int i_start= (i-1) * y_k +1;
                if (i_start < 0)i_start = 0;
                int i_end = i * y_k;
                if (i_end >= src.rows)
			i_end = src.rows;
                int pix[3] = { 0,0,0 };

                int count = (j_end - j_start)*(i_end - i_start);
                for (int n = i_start; n < i_end; n++)
                    for (int m = j_start; m < j_end; m++) {
                        pix[0]+= src.at<Vec3b>(n, m)[0];
                        pix[1]+= src.at<Vec3b>(n, m)[1];
                        pix[2]+= src.at<Vec3b>(n, m)[2];
                    }
                if (count >0) {
		    
                      dst.at<Vec3b>(i, j) = Vec3b(pix[0]/count,pix[1]/count,pix[2]/count);
                }
                else
		{ dst.at<Vec3b>(i, j) = src.at<Vec3b>(i,j);
		}
            }
        }
    }
      int a1=dst.rows;
    int a2=dst.cols;
      printf("Outputpicture's row is %d,col is %d\n",a1,a2);


     namedWindow("ZoomOut", 1);
        imshow ("ZoomOut", dst);

}
void ZoomInNearestNeighborInterpolation(Mat& src, double x_k, double y_k)
     {   namedWindow("src",1);
   imshow("src",src);

	     int i0=src.rows;
   int i1=src.cols;
     printf("Inputpicture's row is %d,col is %d\n",i0,i1);

         Mat dst(src.rows*y_k, src.cols*x_k, src.type(), Scalar(0));
         x_k = 1 / x_k;
         y_k = 1 / y_k;
         if (src.channels() == 3)
         {
                   for (int i = 0; i < dst.rows; i++)
                   {
                            for (int j = 0; j < dst.cols; j++)
                                     dst.at<Vec3b>(i, j) = src.at<Vec3b>(y_k*i, x_k*j);
                   }
         }
	 int a1=dst.rows;
    int a2=dst.cols;
      printf("Outputpicture's row is %d,col is %d\n",a1,a2);

         namedWindow("ZoomIn", 1);
        imshow ("ZoomIn", dst);

}
void ZoomInBilinearInterpolation(Mat& src, double x_k, double y_k)
  {  namedWindow("src",1);
   imshow("src",src);
     int i0=src.rows;
   int i1=src.cols;
     printf("Inputpicture's row is %d,col is %d\n",i0,i1);
    Mat dst(src.rows*y_k, src.cols*x_k, src.type(), Scalar(0));
    x_k = 1 / x_k;
    y_k = 1 / y_k;
    if (src.channels() == 3)
    {
        for (int i = 0; i < dst.rows; i++){
            for (int j = 0; j < dst.cols; j++)
            {
                double x0 = x_k * j;
                double y0 = y_k * i;
                int x1 = int(x0);
                int y1 = int(y0);


                double s1 = y0 - y1;
                double s4 = x0 - x1;
                double s2 = 1 - s4;
                double s3 = 1 - s1;
                if (x1 >= src.cols - 1)x1 = src.cols - 2;
                if (y1 >= src.rows - 1)y1 = src.rows - 2;
                dst.at<Vec3b>(i, j) = src.at<Vec3b>(y1, x1)*s1*s4 + src.at<Vec3b>(y1, x1 + 1)*s1*s2 + src.at<Vec3b>(y1 + 1, x1 + 1)*s2*s3 + src.at<Vec3b>(y1 + 1, x1)*s3*s4;
            }
        }
    }
    int a1=dst.rows;
    int a2=dst.cols;
      printf("Outputpicture's row is %d,col is %d\n",a1,a2);
         namedWindow("ZoomIn", 1);
        imshow ("ZoomIn", dst);

}
void TwoValueErodeImage_1(Mat &src,int n){
    if(!src.data||n%2==0)
           return;
       	int best=0;
   best=  GetBestTH_OTSU(src, best);
   Mat dst=src.clone();
	   TwoValue2(best,src,dst);
   int n1=(n-1)/2;
  int  h=dst.rows;
  int  w=dst.cols;
  Mat m1=dst.clone();
  int control=0;
   for(int i=0;i<h;i++)
     for(int j=0;j<w;j++){
        for(int k=0;k<3;k++){
           int li=i-n1;
           int mi=i+n1;
           int lj=j-n1;
           int mj=j+n1;
            uchar *p=new uchar[n*n];
           int count=0;
                if(li>=0&&lj>=0&&mi<h&&mj<w)
            {      for(int i1=li;i1<mi+1;i1++)
                    for(int j1=lj;j1<mj+1;j1++)
                   p[count++]=dst.at<Vec3b>(i1,j1)[k];
                  int ss=isAllSame(p,n*n);
		
		    if(!ss) 
	   	    m1.at<Vec3b>(i,j)[k]=MinimumValue(p,n*n);
		    else
			    m1.at<Vec3b>(i,j)[k]=dst.at<Vec3b>(i,j)[k];
            }
                  else
                m1.at<Vec3b>(i,j)[k]=dst.at<Vec3b>(i,j)[k];
        }}
         namedWindow("out", WINDOW_FREERATIO);
   imshow("out",m1);

}
Mat TwoValueErodeImage_2(Mat &src,int n){
    if(!src.data||n%2==0)
           return src;
        int best=0;
   best=  GetBestTH_OTSU(src, best);
   Mat dst=src.clone();
           TwoValue2(best,src,dst);
   printf("channels:%d\n",dst.channels());
    int n1=(n-1)/2;
  int  h=dst.rows;
  int  w=dst.cols;
  Mat m1=dst.clone();
  int control=0;
for(int i=0;i<h;i++)
     for(int j=0;j<w;j++){
        for(int k=0;k<3;k++){
           int li=i-n1;
           int mi=i+n1;
           int lj=j-n1;
           int mj=j+n1;
           uchar *p=new uchar[n*n];
           int count=0;
		   p[count++]= dst.at<Vec3b>(i,j)[k];
		   if(li>=0&&lj>=0&&(mi<h)&&(mj<w)){
		   for(int i1=li;i1<mi+1;i1++){
		   if(i1==i)
		    continue;
                      p[count++]=dst.at<Vec3b>(i1,j)[k];
		   }
		   for(int j1=lj;j1<mj+1;j1++)
		   {  if(j1==j)
			   continue;
		       p[count++]=dst.at<Vec3b>(i,j1)[k];
		   }
                   p[count++]=dst.at<Vec3b>(i,j)[k];
                     int ss=isAllSame(p,2*n-1);
                    if(!ss) 
                    m1.at<Vec3b>(i,j)[k]=MinimumValue(p,2*n-1);
                    else
                            m1.at<Vec3b>(i,j)[k]=dst.at<Vec3b>(i,j)[k];

		   }
		    else
                m1.at<Vec3b>(i,j)[k]=dst.at<Vec3b>(i,j)[k];

} }
                 namedWindow("out", WINDOW_FREERATIO);
   imshow("out",m1);
   return m1;
}
void TwoValueErodeImage_3(Mat &src,int n){
    if(!src.data||n%2==0)
           return;
        int best=0;
   best=  GetBestTH_OTSU(src, best);
   Mat dst=src.clone();
           TwoValue2(best,src,dst);
   printf("channels:%d\n",dst.channels());
    int n1=(n-1)/2;
  int  h=dst.rows;
  int  w=dst.cols;
  Mat m1=dst.clone();
  int control=0;
for(int i=0;i<h;i++)
     for(int j=0;j<w;j++){
        for(int k=0;k<3;k++){
           int mi=i+n1;
           int mj=j+n1;
           uchar *p=new uchar[n*n];
           int count=0;
                   p[count++]= dst.at<Vec3b>(i,j)[k];
                   if((mi<h)&&(mj<w)){
                   for(int i1=i;i1<mi+1;i1++){
                   if(i1==i)
                    continue;
                      p[count++]=dst.at<Vec3b>(i1,j)[k];
                   }
                   for(int j1=j;j1<mj+1;j1++)
                   {  if(j1==j)
                           continue;
                       p[count++]=dst.at<Vec3b>(i,j1)[k];
                   }
                   p[count++]=dst.at<Vec3b>(i,j)[k];
                     int ss=isAllSame(p,n);
                    if(!ss)
                    m1.at<Vec3b>(i,j)[k]=MinimumValue(p,n);
                    else
                            m1.at<Vec3b>(i,j)[k]=dst.at<Vec3b>(i,j)[k];
   }
                    else
                m1.at<Vec3b>(i,j)[k]=dst.at<Vec3b>(i,j)[k];

} }
                 namedWindow("out", WINDOW_FREERATIO);
   imshow("out",m1);

}
void TwoValueDilateImage_1(Mat &src,int n){
         if(!src.data||n%2==0)
           return;
        int best=0;
   best=  GetBestTH_OTSU(src, best);
   Mat dst=src.clone();
           TwoValue2(best,src,dst);
   int n1=(n-1)/2;
  int  h=dst.rows;
  int  w=dst.cols;
  Mat m1=dst.clone();
  int control=0;
   for(int i=0;i<h;i++)
     for(int j=0;j<w;j++){
        for(int k=0;k<3;k++){
           int li=i-n1;
           int mi=i+n1;
           int lj=j-n1;
           int mj=j+n1;
            uchar *p=new uchar[n*n];
           int count=0;
                if(li>=0&&lj>=0&&mi<h&&mj<w)
            {      for(int i1=li;i1<mi+1;i1++)
                    for(int j1=lj;j1<mj+1;j1++)
                   p[count++]=dst.at<Vec3b>(i1,j1)[k];
                  int ss=isAllSame(p,n*n);

                    if(!ss)
                    m1.at<Vec3b>(i,j)[k]=MaxValue(p,n*n);
                    else
                            m1.at<Vec3b>(i,j)[k]=dst.at<Vec3b>(i,j)[k];
            }
                  else
                m1.at<Vec3b>(i,j)[k]=dst.at<Vec3b>(i,j)[k];
        }}
         namedWindow("out", WINDOW_FREERATIO);
   imshow("out",m1);

}
Mat TwoValueDilateImage_2(Mat &src,int n){
    if(!src.data||n%2==0)
           return src;
        int best=0;
   best=  GetBestTH_OTSU(src, best);
   Mat dst=src.clone();
           TwoValue2(best,src,dst);
   printf("channels:%d\n",dst.channels());
    int n1=(n-1)/2;
  int  h=dst.rows;
  int  w=dst.cols;
  Mat m1=dst.clone();
  int control=0;
for(int i=0;i<h;i++)
     for(int j=0;j<w;j++){
        for(int k=0;k<3;k++){
           int li=i-n1;
           int mi=i+n1;
           int lj=j-n1;
           int mj=j+n1;
           uchar *p=new uchar[n*n];
           int count=0;
                   p[count++]= dst.at<Vec3b>(i,j)[k];
                   if(li>=0&&lj>=0&&(mi<h)&&(mj<w)){
                   for(int i1=li;i1<mi+1;i1++){
                   if(i1==i)
                    continue;
                      p[count++]=dst.at<Vec3b>(i1,j)[k];
                   }
                   for(int j1=lj;j1<mj+1;j1++)
                   {  if(j1==j)
                           continue;
                       p[count++]=dst.at<Vec3b>(i,j1)[k];
                   }
                   p[count++]=dst.at<Vec3b>(i,j)[k];
                     int ss=isAllSame(p,2*n-1);
                    if(!ss)
                    m1.at<Vec3b>(i,j)[k]=MaxValue(p,2*n-1);
                    else
                            m1.at<Vec3b>(i,j)[k]=dst.at<Vec3b>(i,j)[k];

                   }
                    else
                m1.at<Vec3b>(i,j)[k]=dst.at<Vec3b>(i,j)[k];

} }
                 namedWindow("out", WINDOW_FREERATIO);
   imshow("out",m1);
   return m1;
}
void TwoValueDilateImage_3(Mat &src,int n){
    if(!src.data||n%2==0)
           return;
        int best=0;
   best=  GetBestTH_OTSU(src, best);
   Mat dst=src.clone();
           TwoValue2(best,src,dst);
   printf("channels:%d\n",dst.channels());
    int n1=(n-1)/2;
  int  h=dst.rows;
  int  w=dst.cols;
  Mat m1=dst.clone();
  int control=0;
for(int i=0;i<h;i++)
     for(int j=0;j<w;j++){
        for(int k=0;k<3;k++){
           int mi=i+n1;
           int mj=j+n1;
           uchar *p=new uchar[n*n];
           int count=0;
                   p[count++]= dst.at<Vec3b>(i,j)[k];
                   if((mi<h)&&(mj<w)){
                   for(int i1=i;i1<mi+1;i1++){
                   if(i1==i)
                    continue;
                      p[count++]=dst.at<Vec3b>(i1,j)[k];
                   }
                   for(int j1=j;j1<mj+1;j1++)
                   {  if(j1==j)
                           continue;
                       p[count++]=dst.at<Vec3b>(i,j1)[k];
                   }
                   p[count++]=dst.at<Vec3b>(i,j)[k];
                     int ss=isAllSame(p,n);
                    if(!ss)
                    m1.at<Vec3b>(i,j)[k]=MaxValue(p,n);
                    else
                            m1.at<Vec3b>(i,j)[k]=dst.at<Vec3b>(i,j)[k];
   }
                    else
                m1.at<Vec3b>(i,j)[k]=dst.at<Vec3b>(i,j)[k];

} }
                 namedWindow("out", WINDOW_FREERATIO);

imshow("out",m1);

}
int getDir(int i,int n1,int row){
 int dir=i-n1;
 if(dir<0)
 {if(i+n1>=row)
	 dir=-1;
	 else dir=i+n1;
 }
   return dir;
}

void GrayErodeImage_1(Mat &src,int n){
   if(!src.data||n%2==0)
           return;
   Mat dst=src.clone();
           cvtGray(dst);
   int n1=(n-1)/2;
  int  h=dst.rows;
  int  w=dst.cols;
  Mat m1=dst.clone();
  int control=0;
   for(int i=0;i<h;i++)
     for(int j=0;j<w;j++)
        for(int k=0;k<3;k++){
           int li=i-n1;
           int mi=i+n1;
           int lj=j-n1;
           int mj=j+n1;
            uchar *p=new uchar[n*n];
           int count=0;
                   for(int i1=li;i1<mi+1;i1++)
                    for(int j1=lj;j1<mj+1;j1++)
		    {     if(i1>=0&&j1>=0&&i1<h&&j1<w)
			    p[count++]=dst.at<Vec3b>(i1,j1)[k];}
                  int ss=isAllSame(p,n*n);

                    if(!ss)
                    m1.at<Vec3b>(i,j)[k]=MinimumValue(p,count);
            
        }
         namedWindow("out", WINDOW_FREERATIO);
   imshow("out",m1);


}
void PreWittEdge(Mat &src){
 
}
void CannyEdge(Mat &img){
    int best=0;
   best=  GetBestTH_OTSU(img, best);

       	Mat  gradXY = img.clone();
	double sum=0;
    TwoValue2(best,img,gradXY);
    for (int j = 0; j < img.rows-1; j++) 
        for (int i= 0; i < img.cols-1; i++) {
		for(int k=0;k<3;k++){
            double p = (double)(img.at<Vec3b>(j,i+1)[k] - img.at<Vec3b>(j,i)[k] + img.at<Vec3b>(j+1,i+1)[k] - img.at<Vec3b>(j+1,i+1)[k])/2;
            double q = (double)(img.at<Vec3b>(j+1,i)[k] - img.at<Vec3b>(j,i)[k] + img.at<Vec3b>(j+1,i+1)[k] - img.at<Vec3b>(j,i+1)[k])/2;
              sum=sqrt(p*p + q*q);
	      if(sum>=50)
	    gradXY.at<Vec3b>(j,i)[k] = 255; 
	      else 
		      gradXY.at<Vec3b>(j,i)[k]=0;
        }
    }
    namedWindow("grad", WINDOW_FREERATIO);
   imshow("grad",gradXY);
gradXY=   GuassTemplate(gradXY,3);
   namedWindow("grad1", WINDOW_FREERATIO);
   imshow("grad1",gradXY);

}
void rotatePoint(Point& point1, Point& point2, Point& newPoint, double angle)
{
	int dx, dy;
	double dx1, dy1;
	dy1 = -((double)point2.x - point1.x) * sin(angle) + ((double)point2.y - point1.y) * cos(angle);
	dx1 = ((double)point2.x - point1.x) * cos(angle) + ((double)point2.y - point1.y) * sin(angle);
	if (dx1 - (int)dx1 > 0.5)    //做一个四舍五入
		dx = (int)dx1 + 1;
	else
	{
		if (dx1 - (int)dx1 < -0.5)
			dx = (int)dx1 - 1;
		else
			dx = (int)(dx1);
	}
	if (dy1 - (int)dy1 > 0.5)   //做一个四舍五入
		dy = (int)dy1 + 1;
	else
	{
		if (dy1 - (int)dy1 < -0.5)
			dy = (int)dy1 - 1;
		else
			dy = (int)(dy1);
	}
	newPoint.x = point1.x + dx;
	newPoint.y = point1.y + dy;
}
void translationPoint(cv::Point& point, int x, int y) //平移运算
{
	point.x = point.x + x;
	point.y = point.y + y;
}

int Max4(int a[4])    //获取四个数中的最大值
{
	int max = a[0];
	for (int i = 1; i < 4; i++)
	{
		if (max < a[i])
			max = a[i];
	}
	return max;
}

int Min4(int a[4])    //获取四个数中的最小值
{
	int min = a[0];
	for (int i = 1; i < 4; i++)
	{
		if (min > a[i])
			min = a[i];
	}
	return min;
}
int absMax4(int a[4])
{
	int max = 0, m;
	for (int i = 0; i < 4; i++)
	{
		if (a[i] < 0)
			m = -a[i];
		else m = a[i];
		if (max < m)
			max = m;
	}
	return max;
}

void rotateImage(cv::Mat inputMat, cv::Mat& outputMat, std::vector<cv::Point> points, cv::Point point, double angle)
{
	std::vector<cv::Point> newPoints;
	cv::Point newP;
	for (int i = 0; i < 4; ++i)
	{
		if (points[i] != point)  //判断输入的4个顶点是否与旋转点point相同
		{
			rotatePoint(point, points[i], newP, angle);  //顶点points[i]与旋转点point不同，则进行旋转计算
			newPoints.push_back(newP);
		}
		else
		{
			newPoints.push_back(points[i]);
		}
	}
	//获取经旋转后，新图像的大小，其中w表示图像宽长，h表示图像高长。
	int w = 0, h = 0;
	int suw[4] = { newPoints[1].x - newPoints[0].x,newPoints[1].x - newPoints[3].x,
		newPoints[2].x - newPoints[0].x,newPoints[2].x - newPoints[3].x };
	int suh[4] = { newPoints[2].y - newPoints[0].y ,newPoints[2].y - newPoints[1].y,
		newPoints[3].y - newPoints[0].y,newPoints[3].y - newPoints[1].y };
	w = absMax4(suw);
	h = absMax4(suh);
	//获取需要旋转的四边形区域的外接矩形表示区域范围（x_min,y_min)、(x_max,y_max)
	int y_max, y_min, x_max, x_min;
	int points_x[4] = { points[0].x,points[1].x,points[2].x,points[3].x };
	int points_y[4] = { points[0].y,points[1].y,points[2].y,points[3].y };
	y_max = Max4(points_y);
	y_min = Min4(points_y);
	x_max = Max4(points_x);
	x_min = Min4(points_x);
	//计算向x轴的平移量dx,向y轴的平移量dy
	int dx, dy;
	int a[4] = { newPoints[0].x,newPoints[1].x,newPoints[2].x,newPoints[3].x };
	int b[4] = { newPoints[0].y,newPoints[1].y,newPoints[2].y,newPoints[3].y };
	dx = Min4(a);
	dy = Min4(b);
	//初始化输出矩阵
	if(inputMat.type() == CV_8UC1)
		cv::Mat(h, w, CV_8UC1, cv::Scalar::all(255)).copyTo(outputMat);
	if(inputMat.type() == CV_8UC3)
		cv::Mat(h, w, CV_8UC3, cv::Scalar(255, 255, 255)).copyTo(outputMat);
 //实现I(x',y')=I(x,y)
	double z1, z2, z3, z4;
	for (int i = y_min; i < y_max; ++i)
	{
		for (int j = x_min; j < x_max; ++j)
		{
				//四边形顶点A为points[0],顶点B为points[1],顶点C为points[2],顶点D为points[3].
      //直线AB
			z1 = i - (double)points[0].y -
				(j - (double)points[0].x) * ((double)points[0].y - points[1].y) / ((double)points[0].x - points[1].x);
			//直线BC
			z2 = j - (double)points[1].x -
				(i - (double)points[1].y) * ((double)points[1].x - points[2].x) / ((double)points[1].y - points[2].y);
		  //直线CD
			z3 = i - (double)points[2].y -
				(j - (double)points[2].x) * ((double)points[2].y - points[3].y) / ((double)points[2].x - points[3].x);
			//直线AD
			z4 = j - (double)points[0].x -
				(i - (double)points[0].y) * ((double)points[0].x - points[3].x) / ((double)points[0].y - points[3].y);
			if (z1 >= 0 && z2 <= 0 && z3 <= 0 && z4 >= 0)
			{
				cv::Point point0(j, i);
				rotatePoint(point, point0, point0, angle);  //将点point0绕点point旋转angle度得到新的点point0
				translationPoint(point0, -dx, -dy);
				if (point0.x >= 0 && point0.x < w && point0.y >= 0 && point0.y < h)
				{
					if (inputMat.type() == CV_8UC1)
					{
						uchar* str = inputMat.ptr<uchar>(i);
						outputMat.at<uchar>(point0.y, point0.x) = str[j];
					}
					if (inputMat.type() == CV_8UC3)
					{
						cv::Vec3b* str = inputMat.ptr<cv::Vec3b>(i);
						outputMat.at<cv::Vec3b>(point0.y, point0.x) = str[j];
					}
				}
			}
		}
	}
	if (inputMat.type() == CV_8UC1)
	{   //插值
		for (int i = 1; i < outputMat.rows - 1; ++i)
		{
			for (int j = 1; j < outputMat.cols - 1; ++j)
			{
				if (outputMat.at<uchar>(i, j) == 255)
				{
					int sum = 0;
					uchar* str1 = outputMat.ptr<uchar>(i - 1);
					sum = str1[j - 1] + str1[j] + str1[j + 1];
					uchar* str2 = outputMat.ptr<uchar>(i);
					sum = sum + str2[j - 1] + str2[j + 1];
					uchar* str3 = outputMat.ptr<uchar>(i + 1);
					sum = sum + str3[j - 1] + str3[j] + str3[j + 1];
					sum = sum / 8;
					outputMat.at<uchar>(i, j) = (uchar)sum;
				}
			}
		}
	}
	if (inputMat.type() == CV_8UC3)
	{   //插值
		for (int i = 1; i < outputMat.rows - 1; ++i)
		{
			for (int j = 1; j < outputMat.cols - 1; ++j)
			{
				if (outputMat.at<cv::Vec3b>(i, j) == cv::Vec3b(255, 255, 255))
				{
					int sum[3] = { 0,0,0 };
					uchar r, g, b;
					for (int k = 0; k < 3; k++)
					{
						cv::Vec3b* str1 = outputMat.ptr<cv::Vec3b>(i - 1);
						sum[k] = str1[j - 1][k] + str1[j][k] + str1[j + 1][k];
						cv::Vec3b* str2 = outputMat.ptr<cv::Vec3b>(i);
						sum[k] = sum[k] + str2[j - 1][k] + str2[j + 1][k];
						cv::Vec3b* str3 = outputMat.ptr<cv::Vec3b>(i + 1);
						sum[k] = sum[k] + str3[j - 1][k] + str3[j][k] + str3[j + 1][k];
						sum[k] = sum[k] / 8;
					}
					r = (uchar)sum[0];
					g = (uchar)sum[1];
					b = (uchar)sum[2];
					outputMat.at<cv::Vec3b>(i, j) = cv::Vec3b(r, g, b);
				}
			}
		}
	}
}
void UsetoRotateImage(Mat &src,Mat &dst,double angle){
if (src.empty())
	{
		printf("Cannot read this pic！\n");
		return ;
	}
    std::vector<cv::Point> points;  
	points.push_back(cv::Point(0, 0));
	points.push_back(cv::Point(src.cols, 0));
	points.push_back(cv::Point(src.cols, src.rows));
	points.push_back(cv::Point(0, src.rows));

	rotateImage(src, dst, points, cv::Point(40, 70), angle);
       namedWindow("src", WINDOW_FREERATIO);
   imshow("src",src);
      namedWindow("dst", WINDOW_FREERATIO);
   imshow("dst",dst);


}
int GaussianKernel(int kernel_size, std::vector<std::vector<float>> &kernel)
{
    kernel.clear();
    kernel.resize(kernel_size);
    for (auto &it : kernel)
    {
        it.resize(kernel_size);
    }

    std::vector<int> coord_val(kernel_size, -kernel_size / 2);  //  坐标取值范围,一般取0左右对称的相反数
    for (int i = 1; i < kernel_size; ++i)
    {
        coord_val[i] = coord_val[i - 1] + 1;
    }

    const float kSigma = 0.5f;     // sigma,σ,Standard Deviation
    float val1 = 1 / (2 * M_PI * kSigma * kSigma);
    float val2 = -1 / (2 * kSigma * kSigma);
    for (int i = 0; i < kernel_size; ++i)
    {
        for (int j = 0; j < kernel_size; ++j)
        {
            kernel[i][j] = val1 * exp(val2 * (pow(coord_val[i], 2) + pow(coord_val[j], 2)));
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i)
    {
        sum += std::accumulate(kernel[i].begin(), kernel[i].end(), 0.0);
    }

    for (int i = 0; i < kernel_size; ++i)
    {
        for (int j = 0; j < kernel_size; ++j)
        {
            kernel[i][j] /= sum;
        }
    }

    return 0;
}
int GaussianBlur(const std::vector<std::vector<float>> &kernel, const cv::Mat &src, cv::Mat &formula_gaussian)
{
    int kernel_size = kernel.size();

    /* 对原图灰度图补pad */
    int pad = kernel_size >> 1;
    cv::Mat formula_pad;
    formula_pad.convertTo(src, CV_32FC1, 1 / 255.0);
    memset(formula_pad.data, 0, formula_pad.rows * formula_pad.cols * sizeof(float));
  //  cv::Rect rect(pad, pad, src.cols, src.rows);
   // src.copyTo(formula_pad(rect));
    /* 执行高斯滤波 */
    memset(formula_gaussian.data, 0, formula_gaussian.rows * formula_gaussian.cols * sizeof(float));
    for (int i = 0; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
            float tmp = 0.0f;
            for (int m = 0; m < kernel_size; ++m)
            {
                for (int n = 0; n < kernel_size; ++n)
                {
                    tmp += kernel[m][n] * formula_pad.at<float>(i + m, j + n);
                }
            }

            formula_gaussian.at<float>(i, j) = tmp;
        }
    }

    return 0;
}
int DoSobel(const cv::Mat &formula_gaussian, cv::Mat &sobel_xy, cv::Mat &sobel_angle, float &mean_sobel_xy)
{
    /* Sobel滤波 */
    const int kSobelKernel = 3;
    int pad = kSobelKernel >> 1;   // Sobel的kernel大小是3
    std::vector<std::vector<int> > kernel_x{ {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    std::vector<std::vector<int> > kernel_y{ {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };
   // cv::Mat gaussian_pad(formula_gaussian.rows , formula_gaussian.cols, CV_32FC1);
    Mat gaussian_pad;
     gaussian_pad.convertTo(formula_gaussian, CV_32FC1, 1 / 255.0);
    memset(gaussian_pad.data, 0, gaussian_pad.rows * gaussian_pad.cols * sizeof(float));
    cv::Rect rect = cv::Rect(pad, pad, formula_gaussian.cols, formula_gaussian.rows);
   // formula_gaussian.copyTo(gaussian_pad(rect));

    cv::Mat sobel_x(formula_gaussian.rows, formula_gaussian.cols, CV_32FC1);
    cv::Mat sobel_y(formula_gaussian.rows, formula_gaussian.cols, CV_32FC1);
    sobel_xy = cv::Mat(formula_gaussian.rows, formula_gaussian.cols, CV_32FC1);
    sobel_angle = cv::Mat(formula_gaussian.rows, formula_gaussian.cols, CV_32FC1);  // 这里角度用tan值表示
    const float kEps = 0.000001f;
    float sum_sobel_xy = 0.0f;
    mean_sobel_xy = 0.0f;
    for (int i = 0; i < formula_gaussian.rows; ++i)
    {
        for (int j = 0; j < formula_gaussian.cols; ++j)
        {
            float gradient_x = 0.0f;
            float gradient_y = 0.0f;
            float gradient_xy = 0.0f;
            for (int m = 0; m < kSobelKernel; ++m)
            {
                for (int n = 0; n < kSobelKernel; ++n)
                {
                    gradient_x += kernel_x[m][n] * gaussian_pad.at<float>(i + m, j + n);
                    gradient_y += kernel_y[m][n] * gaussian_pad.at<float>(i + m, j + n);
                }
            }
            gradient_xy = sqrt(pow(gradient_x, 2) + pow(gradient_y, 2));
            sobel_x.at<float>(i, j) = gradient_x;
            sobel_y.at<float>(i, j) = gradient_y;
            sobel_xy.at<float>(i, j) = gradient_xy;
            sobel_angle.at<float>(i, j) = gradient_y / (fabs(gradient_x) > kEps ? gradient_x : kEps);

            sum_sobel_xy += gradient_xy;
        }
    }
    mean_sobel_xy = sum_sobel_xy / (sobel_xy.rows * sobel_xy.cols);

    return 0;
}
int DoNMS(const cv::Mat &sobel_xy, const cv::Mat &sobel_angle, cv::Mat &nms_sobel_xy)
{
    nms_sobel_xy = cv::Mat(sobel_xy.rows, sobel_xy.cols, CV_32FC1);
    memset(nms_sobel_xy.data, 0, nms_sobel_xy.rows * nms_sobel_xy.cols * sizeof(float));

    for (int i = 1; i < nms_sobel_xy.rows - 1; ++i)        // 图像第一行、最后一行不认为是边缘
    {
        for (int j = 1; j < nms_sobel_xy.cols - 1; ++j)    // 图像第一列、最后一列不认为是边缘
        {
            // 8邻域各像素的梯度
            float en0 = sobel_xy.at<float>(i - 1, j - 1);
            float en1 = sobel_xy.at<float>(i - 1, j);
            float en2 = sobel_xy.at<float>(i - 1, j + 1);
            float en3 = sobel_xy.at<float>(i, j - 1);
            float en4 = sobel_xy.at<float>(i, j);          // 当前像素梯度
            float en5 = sobel_xy.at<float>(i, j + 1);
            float en6 = sobel_xy.at<float>(i + 1, j - 1);
            float en7 = sobel_xy.at<float>(i + 1, j);
            float en8 = sobel_xy.at<float>(i + 1, j + 1);

            // 插值interpolation
            float grad_inter1 = 0.0f;
            float grad_inter2 = 0.0f;
            float ratio = 0.0f;   // 插值时的权重比例,[0.0, 1.0f]
            float angle = sobel_angle.at<float>(i, j);  // 注意角度使用dy/dx表示的,因此ratio是|angle|或1/|angle|
            if (angle >= 0)  // 第一、三象限
            {
                if (angle >= 1)         // [45°, 90°]使用en1、en2得到插值一，[225°, 270°]使用en6、en7得到插值二
                {
                    ratio = 1.0f / angle;  // angle越大越靠近Y轴,en1、en7的占比就越大,ratio越小,1-ratio越大;
                                           // 当angele=1时是45°或225°方向此时正好en2或en6的占比是1;
                                           // 当angle无穷大时是90°或270°方向此时正好en1、en7的占比是1
                    grad_inter1 = (1 - ratio) * en1 + ratio * en2;
                    grad_inter2 = (1 - ratio) * en7 + ratio * en6;
                }
                else  // angle∈[0,1)   // [0°, 45°)使用en2、en5得到插值一，[180°, 225°)使用en3、en6得到插值二
                {
                    ratio = angle;         // angle越小越靠近X轴,en5、en3的占比就越大,ratio越小,1-ratio越大;
                                           // 当angle=0时是0°或180°方向此时正好en5、en3的占比是1;
                                           // 当angle=1时是45°或225°方向此时正好en2、en6的占比是1
                    grad_inter1 = (1 - ratio) * en5 + ratio * en2;
                    grad_inter2 = (1 - ratio) * en3 + ratio * en6;
                }
            }
            else             // 第二、四象限
            {
                if (fabs(angle) >= 1)  // (90°, 135°]使用en0、en1得到插值一，(270°, 315°]使用en7、en8得到插值二
                {
                    ratio = 1.0f / fabs(angle);  // fabs(angle)越大越靠近Y轴,en1、en7的占比就越大,ratio越小,1-ratio越大;
                                                 // 当fabs(angle)=1时是135°或315°方向此时正好en0或en8的占比是1;
                                                 // 当fabs(angle)无穷大时是90°或270°方向此时正好en1、en7的占比是1
                    grad_inter1 = (1 - ratio) * en1 + ratio * en0;
                    grad_inter2 = (1 - ratio) * en7 + ratio * en8;
                }
                else  // fabs(angle)∈[0,1)   // (135°, 180°]使用en0、en3得到插值一，(315°, 360°]使用en5、en8得到插值二
                {
                    ratio = fabs(angle);         // fabs(angle)越小越靠近X轴,en3、en5的占比就越大,ratio越小,1-ratio越大;
                                                 // 当fabs(angle)=0时是180°或360°方向此时正好en3或en5的占比是1;
                                                 // 当fabs(angle)=1时是135°或315°方向此时正好en0、en8的占比是1
                    grad_inter1 = (1 - ratio) * en3 + ratio * en0;
                    grad_inter2 = (1 - ratio) * en5 + ratio * en8;
                }
            }
            if (en4 > grad_inter1 && en4 > grad_inter2)
            {
                nms_sobel_xy.at<float>(i, j) = en4;
            }
        }
    }

    return 0;
}
int DoBinaryThresh(const cv::Mat &nms_sobel_xy, float mean_sobel_xy, cv::Mat &img_canny)
{
    cv::Mat binary_thresh_canny = cv::Mat(nms_sobel_xy.rows, nms_sobel_xy.cols, CV_32FC1);
    memset(binary_thresh_canny.data, 0, binary_thresh_canny.rows * binary_thresh_canny.cols * sizeof(float));

    float low_thresh = mean_sobel_xy * 0.5f;
    float high_thresh = low_thresh * 3.0f;
    for (int i = 1; i < binary_thresh_canny.rows - 1; ++i)      // 图像第一行、最后一行不认为是边缘
    {
        for (int j = 1; j < binary_thresh_canny.cols - 1; ++j)  // 图像第一列、最后一列不认为是边缘
        {
            float en0 = nms_sobel_xy.at<float>(i - 1, j - 1);
            float en1 = nms_sobel_xy.at<float>(i - 1, j);
            float en2 = nms_sobel_xy.at<float>(i - 1, j + 1);
            float en3 = nms_sobel_xy.at<float>(i, j - 1);
            float en4 = nms_sobel_xy.at<float>(i, j);              // 当前像素梯度
            float en5 = nms_sobel_xy.at<float>(i, j + 1);
            float en6 = nms_sobel_xy.at<float>(i + 1, j - 1);
            float en7 = nms_sobel_xy.at<float>(i + 1, j);
            float en8 = nms_sobel_xy.at<float>(i + 1, j + 1);
            if (en4 >= high_thresh)       // 强边缘
            {
                binary_thresh_canny.at<float>(i, j) = 255.0f;
            }
            else if (en4 <= low_thresh)  // 不是边缘
            {
                binary_thresh_canny.at<float>(i, j) = 0.0f;
            }
            else                        // 弱边缘,继续8邻域的判断
            {
                if (en0 >= high_thresh || en1 >= high_thresh || en2 >= high_thresh
                    || en3 >= high_thresh || en5 >= high_thresh
                    || en6 >= high_thresh || en7 >= high_thresh || en8 >= high_thresh)
                {
                    binary_thresh_canny.at<float>(i, j) = 255.0f;
                }
            }
        }
    }

    binary_thresh_canny.convertTo(img_canny, CV_8UC1);

    return 0;
}
void realizeCanny(Mat &img){
    cv::Mat imgf;
    img.convertTo(imgf, CV_32FC1, 1 / 255.0);

    /* 计算高斯滤波器 */
    int kKernelSize = 5;   // 滤波器大小,可自定义,但应当是奇数
    std::vector<std::vector<float>> kernel;
    int ret = GaussianKernel(kKernelSize, kernel);
    if (0 != ret)
    {
        
        return ;
    }

    /* 高斯滤波 */
    cv::Mat formula_gaussian;
    ret = GaussianBlur(kernel, imgf, formula_gaussian);
    if (0 != ret)
    {
        return ;
    }
   namedWindow("img_1", WINDOW_FREERATIO);
    imshow("img_1", formula_gaussian);
    cv::Mat sobel_xy, sobel_angle;
    float mean_sobel_xy = 0.0f;
    ret = DoSobel(formula_gaussian, sobel_xy, sobel_angle, mean_sobel_xy);
    if (0 != ret)
    {
        return ;
    }

    cv::Mat nms_sobel_xy;
    ret = DoNMS(sobel_xy, sobel_angle, nms_sobel_xy);
    if (0 != ret)
    {
        return ;
    }

    cv::Mat img_canny;
    ret = DoBinaryThresh(nms_sobel_xy, mean_sobel_xy, img_canny);
    if (0 != ret)
    {
        return ;
    }
         namedWindow("img_canny", WINDOW_FREERATIO);
    imshow("img_canny", img_canny);

    return ;
}
void showCanny(Mat &img){
     int best=0;
   best=  GetBestTH_OTSU(img, best);

        Mat  gradXY = img.clone();
        double sum=0;
    TwoValue2(best,img,gradXY);
    blur(gradXY,gradXY,Size(3,3));
    Mat edge;
    Canny(gradXY,edge,150,100,3);
    Mat dst;
    dst.create(gradXY.size(),gradXY.type());
    dst=Scalar::all(0);
    gradXY.copyTo(dst,edge);
     namedWindow("img_canny", WINDOW_FREERATIO);
    imshow("img_canny", dst);


}
