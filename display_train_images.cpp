#include "template.h"
#include <opencv2/opencv.hpp>

using namespace cv;
ifstream train;
inline ll trainscan()
{
    ll n=0;
    char c;
    train.get(c);
    bool check=0;
    if(c=='-')check=1;
    while(c<'0'||c>'9')
    {
        train.get(c);
        if(c=='-')check=1;
    }
    while(c<='9'&&c>='0'){
        n=n*10+c-'0';
        train.get(c);
    }
    return n+(-2*check*n);
}
int main(){
    train.open("dataset/train.csv");
    string s;
    train>>s;
    short number_of_images=10;
    while(number_of_images--){
        int emotion;
        emotion=trainscan();
        Mat image(48,48,CV_8UC1);
        short i,j;
        f(i,48){
            f(j,48){
                int x;
                x=trainscan();
                // cout<<x<<' ';
                image.at<uchar>(i,j) = x;
            }
            // cout<<'\n';
        }
        string s;
        switch(emotion){
            case 0: s="anger";
                    break;
            case 1: s="disgust";
                    break;
            case 2: s="fear";
                    break;
            case 3: s="happy";
                    break;
            case 4: s="sad";
                    break;
            case 5: s="surprise";
                    break;
            case 6: s="neutral";
                    break;
        }
        //anger=0, disgust=1, fear=2, happy=3, sad=4, surprise=5, neutral=6
        putText(image, s, Point(0,10), FONT_HERSHEY_DUPLEX, 0.2, Scalar(0), 0.2);
        Size size(128,128);//the dst image size,e.g.100x100
        resize(image,image,size);//resize image
        namedWindow("image");
        imshow("image",image);
        waitKey(0);
    }
    return 0;
}