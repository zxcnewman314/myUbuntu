#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
int main( )
 {   VideoCapture capture(0);
	 if(!capture.isOpened())
         { printf("haven't turn on the camera!\n");
		 waitKey(80);
		 return 0;
	 }
	 Size mySize(capture.get(cv::CAP_PROP_FRAME_WIDTH),capture.get(cv::CAP_PROP_FRAME_HEIGHT));
	 VideoWriter vTest("vTest.avi",VideoWriter::fourcc('M','J','P','G'),30,mySize);
	 int Key=0;
	 bool Cend=true;
	 char flag=0;
	 Mat fn;
	 while(true){
	capture >> fn;
	 Key=waitKey(80);
	 if(Key==27)
		 break;
	 else if(Key==32){
		 Cend=!Cend;
		 if(!Cend){
			 flag=1;
		          }
	                 }
	   else  if(!Cend&&flag){
                  capture >> fn ;
		  vTest << fn;
	//	  vTest.write(fn);
		  printf("recording.\n");
	     }
	 else  if(Cend){
		 flag=0;
		 printf("recording paused\n");
	 }	 
	 imshow("vTest",fn);
	 } 
	 capture.release();
	 vTest.release();
	 destroyAllWindows();
    return 0;
}
