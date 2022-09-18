#include <stdio.h>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
int main( )
 {    std::string path="am.webm";
       	 VideoCapture capture(path);
         Size mySize(capture.get(cv::CAP_PROP_FRAME_WIDTH),capture.get(cv::CAP_PROP_FRAME_HEIGHT));
           if(!capture.isOpened())
         { printf("haven't turn on the camera!\n");
                 waitKey(80);
                 return 0;
         } 
	 int count=0;
         char flag=0;
	 Mat fn;
	  namedWindow("Watch", WINDOW_FREERATIO);
	  waitKey(2000);
          while(1){
	capture >> fn;
	count=waitKey(80);
          if(fn.empty())
	  break;
           imshow("video",fn);
	   if(count==27)
			   break;
	  }
	  capture.release();
         destroyAllWindows();
    return 0;
}

