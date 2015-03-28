#include <opencv2\opencv.hpp>
#include <opencv2\ml\ml.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat sampleVectors(Mat& img)
{
	Mat tempimg;
	img.convertTo(tempimg,CV_32FC3);
	Mat samples = tempimg.reshape(1,img.rows*img.cols);
	return samples;
}

int main()
{
	int hmin=140,hmax=180;
	int smin=50,smax=255;
	int vmin=0,vmax=255;

	Mat image,hsv_image,t_image;
	namedWindow("image",CV_WINDOW_AUTOSIZE);
	namedWindow("Thresholded",CV_WINDOW_AUTOSIZE);
	namedWindow("detected",CV_WINDOW_AUTOSIZE);
	//namedWindow("Control",CV_WINDOW_AUTOSIZE);

	/*
	createTrackbar("hue low","Control",&hmin,180);
	createTrackbar("hue high","Control",&hmax,180);
	createTrackbar("saturation low","Control",&smin,255);
	createTrackbar("saturation high","Control",&smax,255);
	createTrackbar("value low","Control",&vmin,255);
	createTrackbar("value high","Control",&vmax,255);
	*/

	image=imread("RealRoadSigns.JPG",CV_LOAD_IMAGE_COLOR); //Enter name of image to detect road signs in
	double t1=getTickCount();
	cvtColor(image,hsv_image,CV_BGR2HSV);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	
	//while(true)
	//{
		Mat edges=Mat::zeros(image.rows,image.cols,image.type());
		Mat temp=Mat::zeros(image.rows,image.cols,image.type());

		inRange(hsv_image,Scalar(hmin,smin,vmin),Scalar(hmax,smax,vmax),t_image);
		dilate(t_image, t_image, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)) );
		//erode(t_image, t_image, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)) );

		findContours(t_image,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
		vector<vector<int> >hull(1);
		vector<vector<Point> >hull_pt(1);
		vector<Vec4i> defects;

		if(!contours.empty())
		{
				for(int i=0;i<contours.size();i++)
				{
					if(contourArea(contours[i])>500)
					{
					Rect myRect=boundingRect(contours[i]);
					drawContours(edges,contours,i,Scalar(255,255,255),-1);
					convexHull( Mat(contours[i]), hull[0], false );
					convexHull( Mat(contours[i]), hull_pt[0], false );
					convexityDefects(contours[i], hull[0], defects);
					
					vector<float> distances;
					for(int x=0;x<defects.size();x++)
					{
						distances.push_back(defects[x][3]);
					}
					cv::sort(distances,distances,CV_SORT_DESCENDING);
					
					if(distances[0]/256<10)
					{
					Mat image_roi=image(myRect);
					Mat edges_roi=edges(myRect);
					Mat temp_roi=temp(myRect);

					image_roi.copyTo(temp_roi,edges_roi);
					Mat samples=sampleVectors(temp_roi);
					Mat bestLabels, centers, clustered;
					int K = 4;
					cv::kmeans(samples, K, bestLabels,TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),3, KMEANS_PP_CENTERS, centers);
					centers=centers.reshape(3,K);
					centers.convertTo(centers,CV_8UC3);
					bestLabels=bestLabels.reshape(1,image_roi.rows);
					bestLabels.convertTo(bestLabels,CV_8UC1);
					//cvtColor(centers,centers,CV_BGR2GRAY);
					int min=255,index=0;

					for(int x=0;x<4;x++)
					{
						if((float)centers.at<Vec3b>(x,0)[2]>(((float)centers.at<Vec3b>(x,0)[0]*(3))/(2)) && (float)centers.at<Vec3b>(x,0)[2]>(((float)centers.at<Vec3b>(x,0)[1]*(3))/(2)))
						{
							centers.at<Vec3b>(x,0)=Vec3b(0,0,255);
							break;
						}
					}
					for(int x=0;x<4;x++)
					{
						if(centers.at<Vec3b>(x,0)[0]>1 && centers.at<Vec3b>(x,0)[0]<min)
						{
							min=centers.at<Vec3b>(x,0)[0];
							index=x;
						}
					}
					centers.at<Vec3b>(index,0)=Vec3b(0,0,0);
					for(int x=0;x<4;x++)
					{
						if((float)centers.at<Vec3b>(x,0)[2]>((float)centers.at<Vec3b>(x,0)[0]+20) && (float)centers.at<Vec3b>(x,0)[2]>((float)centers.at<Vec3b>(x,0)[1]+20))
						{
							centers.at<Vec3b>(x,0)=Vec3b(0,0,255);
							break;
						}
					}
					for(int x=0;x<4;x++)
					{
						if(centers.at<Vec3b>(x,0)[0]>1)
						{
							centers.at<Vec3b>(x,0)=Vec3b(255,255,255);
							break;
						}
					}
					//cout<<centers<<"\n";
					
					for(int x=0;x<edges_roi.rows;x++)
					{
						for(int y=0;y<edges_roi.cols;y++)
						{
							temp_roi.at<Vec3b>(x,y)=centers.at<Vec3b>(bestLabels.at<uchar>(x,y),0);
						}
					}
					rectangle(temp,myRect,Scalar(255,0,0),2);
					drawContours( temp, hull_pt, 0, Scalar(0,255,0), 2, 8, vector<Vec4i>(), 0, Point() );
					imshow("detected",temp);
					waitKey(1);
					}}
				}
		}

		if(!edges.empty())
			imshow("Thresholded",edges);
		imshow("image",image);
		imshow("detected",temp);
		
		double t2 = (getTickCount()-t1)/getTickFrequency();
		cout<<"Time taken = "<<t2;
		waitKey(0);
		//contours.clear();
		//hierarchy.clear();
	//}

	return 0;
}
