#include "template.h"
#include <opencv2/opencv.hpp>
using namespace cv;
ld w1[5][5][1][10],w2[5][5][10][10],w4[5][5][10][7],wFC[700][7];
void read_weights(){
    ifstream weights("weights");
    short i,j,k,u;
    f(i,5)f(j,5)f(k,10)weights>>w1[i][j][0][k];
    f(i,5)f(j,5)f(k,10)f(u,10)weights>>w2[i][j][k][u];
    f(i,5)f(j,5)f(k,10)f(u,7)weights>>w4[i][j][k][u];
    f(i,700)f(j,7)weights>>wFC[i][j];
}



void forward_prop(int image_no,vvvld network[7]){
    // ofstream xxx("network");
    network[0][0] = train[image_no];
    
    vvvld temp;
    short i,j,k;

    //network[1]
    temp.resize(1,vvld(50,vld(50,0)));
    f(i,50)
        f(j,50){
            temp[0][i][j]=0;
            if((i-1)>=0&&(j-1)>=0&&(i-1)<48&&(j-1)<48)
                temp[0][i][j]=network[0][0][i-1][j-1];
        }

    f(i,46){
        f(j,46){
            f(k,10){
                short u;
                network[1][k][i][j]=0;
                f(u,25){
                    network[1][k][i][j]+=w1[u/5][u%5][0][k]*temp[0][i+(u/5)][j+(u%5)];
                }
                network[1][k][i][j] = activation(network[1][k][i][j]);
                // xxx<<"net1"<<'\n';
                // xxx<<network[1][k][i][j]<<'\n';
            }
        }
    }
    // network 2
    temp.resize(10,vvld(48,vld(48,0)));
    f(k,10)
    f(i,48)
        f(j,48){
            temp[k][i][j]=0;
            if(i-1>=0&&j-1>=0&&i-1<46&&j-1<46)
                temp[k][i][j]=network[1][k][i-1][j-1];
        }

    f(i,44){
        f(j,44){
            f(k,10){
                short u,u2;
                network[2][k][i][j]=0;
                f(u2,10)
                    f(u,25){
                        network[2][k][i][j]+=w2[u/5][u%5][u2][k]*temp[u2][i+(u/5)][j+(u%5)];
                    }
                network[2][k][i][j] = activation(network[2][k][i][j]);
                // xxx<<"net2"<<'\n';
                // xxx<<network[2][k][i][j]<<'\n';
            }
        }
    }

    // network 3
    f(i,22){
        f(j,22){
            f(k,10){
                network[3][k][i][j]=max(max(network[2][k][2*i][2*j],network[2][k][2*i+1][2*j]),max(network[2][k][2*i][2*j+1],network[2][k][2*i+1][2*j+1]));

                // xxx<<"net3"<<'\n';
                // xxx<<network[3][k][i][j]<<'\n';
            }
        }
    }

    // network 4
    temp.resize(10,vvld(24,vld(24,0)));
    f(k,10)
    f(i,24)
        f(j,24){
            temp[k][i][j]=0;
            if(i-1>=0&&j-1>=0&&i-1<22&&j-1<22)
                temp[k][i][j]=network[3][k][i-1][j-1];
        }

    f(i,10){
        f(j,10){
            f(k,7){
                short u,u2;
                network[4][k][i][j]=0;
                f(u2,10)
                    f(u,25){
                        network[4][k][i][j]+=w4[u/5][u%5][u2][k]*temp[u2][(2*i)+(u/5)][(2*j)+(u%5)];
                    }
                network[4][k][i][j] = activation(network[4][k][i][j]);
                // xxx<<"net4"<<'\n';
                // xxx<<network[4][k][i][j]<<'\n';
            }
        }
    }

    // network 5 or o/p
    f(i,7){
        network[5][0][0][i]=0;
        f(j,700)
            network[5][0][0][i]+=network[4][j/100][(j/10)%10][j%10]*wFC[j][i];
        network[5][0][0][i] = softmax(i,network[5][0][0]);
        // xxx<<"net5"<<'\n';
        // xxx<<network[5][0][0][i]<<'\n';
    }
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade, 
                CascadeClassifier& nestedCascade, double scale );
string cascadeName, nestedCascadeName;
 
int main( int argc, const char** argv )
{
    read_weights();
    // VideoCapture class for playing video for which faces to be detected
    VideoCapture capture; 
    Mat frame, image;
 
    // PreDefined trained XML classifiers with facial features
    CascadeClassifier cascade, nestedCascade; 
    double scale=1;
 
    // Load classifiers from "opencv/data/haarcascades" directory 
    nestedCascade.load( "../../haarcascade_eye_tree_eyeglasses.xml" ) ;
 
    // Change path before execution 
    cascade.load( "../../haarcascade_frontalcatface.xml" ) ; 
 
    // Start Video..1) 0 for WebCam 2) "Path to Video" for a Local Video
    capture.open(0); 
    if( capture.isOpened() )
    {
        // Capture frames from video and detect faces
        cout << "Face Detection Started...." << endl;
        while(1)
        {
            capture >> frame;
            if( frame.empty() )
                break;
            Mat frame1 = frame.clone();
            detectAndDraw( frame1, cascade, nestedCascade, scale ); 
            char c = (char)waitKey(10);
         
            // Press q to exit from window
            if( c == 27 || c == 'q' || c == 'Q' ) 
                break;
        }
    }
    else
        cout<<"Could not Open Camera";
    return 0;
}
 
void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale)
{
    vector<Rect> faces, faces2;
    Mat gray, smallImg;
 
    cvtColor( img, gray, COLOR_BGR2GRAY ); // Convert to Gray Scale
    double fx = 1 / scale;
 
    // Resize the Grayscale Image 
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR ); 
    equalizeHist( smallImg, smallImg );
 
    // Detect faces of different sizes using cascade classifier 
    cascade.detectMultiScale( smallImg, faces, 1.1, 
                            2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
 
    // Draw circles around the faces
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = Scalar(255, 0, 0); // Color for Drawing tool
        int radius;
 
        double aspect_ratio = (double)r.width/r.height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
                    cvPoint(cvRound((r.x + r.width-1)*scale), 
                    cvRound((r.y + r.height-1)*scale)), color, 3, 8, 0);
        if( nestedCascade.empty() )
            continue;
        smallImgROI = smallImg( r );
         
        // Detection of eyes int the input image
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects, 1.1, 2,
                                        0|CASCADE_SCALE_IMAGE, Size(30, 30) ); 
         
        // Draw circles around eyes
        for ( size_t j = 0; j < nestedObjects.size(); j++ ) 
        {
            Rect nr = nestedObjects[j];
            center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
            center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
            radius = cvRound((nr.width + nr.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
    }
 
    // Show Processed Image with detected faces
    imshow( "Face Detection", img ); 
}