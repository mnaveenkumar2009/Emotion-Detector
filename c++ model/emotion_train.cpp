#include "template.h"
#include <omp.h>
// #include <opencv2/opencv.hpp>
// using namespace cv;
#define vvvld vector < vector < vector <ld> > >
#define vvld vector <  vector <ld> >
#define vld vector < ld >
/*
model : 6 layers
        1st layer : i/p : 48*48*1
        2nd layer : 5*5*1 conv(p=1,s=1) with 10 filters : 46*46*10
        3rd layer : 5*5*10 conv(p=1,s=1) with 10 filters : 44*44*10
        4th layer : max pool (/2): 22*22*10
        5th layer : 5*5*10 conv(p=1,s=2) with 7 filters : 10*10*7 = FC 700
        6th layer : o/p
*/
#define relu(x) ((ld)(x>0)*x)
#define sigmoid(x) (1/(1+exp(-x)))
#define activation(x) relu(x)

ld softmax(int xxx,vector <ld> n){
    ld a=exp(n[xxx]);
    int i;
    ld b=0;
    f(i,7)
        b+=exp(n[i]);
    return (a/b);
}
#define number_of_images 4178

vvvld train(4178,vvld (48,vld (48)));
int label[4178];
short train_size=2800;
ld alpha = 4;
ld w1[5][5][1][10],w2[5][5][10][10],w4[5][5][10][7],wFC[700][7];
void encode_weights(){
    ofstream weights("weights");
    short i,j,k,u;
    f(i,5)f(j,5)f(k,10)weights<<w1[i][j][0][k]<<'\n';
    f(i,5)f(j,5)f(k,10)f(u,10)weights<<w2[i][j][k][u]<<'\n';
    f(i,5)f(j,5)f(k,10)f(u,7)weights<<w4[i][j][k][u]<<'\n';
    f(i,700)f(j,7)weights<<wFC[i][j]<<'\n';
}
void read_weights(){
    ifstream weights("weights");
    short i,j,k,u;
    f(i,5)f(j,5)f(k,10)weights>>w1[i][j][0][k];
    f(i,5)f(j,5)f(k,10)f(u,10)weights>>w2[i][j][k][u];
    f(i,5)f(j,5)f(k,10)f(u,7)weights>>w4[i][j][k][u];
    f(i,700)f(j,7)weights>>wFC[i][j];
}
void random_initialize(){
    
    srand(time(NULL));
    short i,j,k,u;
    f(i,5)
        f(j,5)
            f(k,1)
                f(u,10)
                    w1[i][j][k][u]=((ld)(rand()-rand()))/1e11;
    f(i,5)
        f(j,5)
            f(k,10)
                f(u,10)
                    w2[i][j][k][u]=((ld)(rand()-rand()))/1e11;
    f(i,5)
        f(j,5)
            f(k,10)
                f(u,7)
                    w4[i][j][k][u]=((ld)(rand()-rand()))/1e11;
    f(i,700)
        f(j,7)
            wFC[i][j]=((ld)(rand()-rand()))/1e11;
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

void back_prop(){
    

   short number_of_iterations=200;
    // random initialisation
    bool random=0;
    if(random)
        random_initialize();
    else
        read_weights();
    // do the back prop
    while(number_of_iterations--){
        short pred_label[4178];
        short image_no;
        ld accuracy = 0;
        ld Dw1[5][5][1][10]={0},Dw2[5][5][10][10]={0},Dw4[5][5][10][7]={0},DwFC[700][7]={0};
        omp_set_num_threads(700);
        short xxx;
        f(xxx,4){
            #pragma omp parallel private(image_no)
            {
                short i,j;
                image_no = omp_get_thread_num() + xxx*700;
                vvvld network[7];
                // network
                network[0].resize(1,vvld(48,vld(48,0)));
                network[1].resize(10,vvld(46,vld(46,0)));
                network[2].resize(10,vvld(44,vld(44,0)));
                network[3].resize(10,vvld(22,vld(22,0)));
                network[4].resize(7,vvld(10,vld(10,0))); // FC 700
                network[5].resize(1,vvld(1,vld(7,0)));
                vvvld error[7];
                error[0].resize(1,vvld(48,vld(48,0))); // useless
                error[1].resize(10,vvld(46,vld(46,0)));
                error[2].resize(10,vvld(44,vld(44,0)));
                error[3].resize(10,vvld(22,vld(22,0)));
                error[4].resize(7,vvld(10,vld(10,0))); // FC 700
                error[5].resize(1,vvld(1,vld(7,0)));
                //forward propogation
                
                forward_prop(image_no,network);
                
                //find errors
                error[5]=network[5];// output error
                error[5][0][0][label[image_no]]-=1.0;

                // error 4
                f(i,700){
                    ld temp = network[4][i/100][(i/10)%10][i%10];
                    f(j,7)
                        error[4][i/100][(i/10)%10][i%10]+=wFC[i][j]*error[5][0][0][j];
                    error[4][i/100][(i/10)%10][i%10]*=temp*(1-temp);
                }
                
                short k,u;
                // error 3
                f(i,10){
                    f(j,10){
                        f(k,7){
                            short u,u2;
                            f(u2,10)
                                f(u,25){
                                    if(((2*i)+(u/5))-1<22&&((2*i)+(u/5))-1>=0&&((2*j)+(u%5))-1<22&&((2*j)+(u%5))-1>=0)
                                        error[3][u2][((2*i)+(u/5))-1][((2*j)+(u%5))-1]+=error[4][k][i][j]*w4[u/5][u%5][u2][k];
                                }
                        }
                    }
                }

                f(i,10)
                    f(j,22)
                        f(k,22)
                            error[3][i][j][k]*=(ld)(network[3][i][j][k]>0);

                // error 2
                f(i,44){
                    f(j,44){
                        f(k,10){
                            if(network[2][k][i][j]==network[3][k][i/2][j/2]){
                                error[2][k][i][j]=error[3][k][i/2][j/2];
                            }
                        }
                    }
                }
                
                // error 1

                f(i,44){
                    f(j,44){
                        f(k,10){
                            short u,u2;
                            f(u2,10)
                                f(u,25){
                                    if((i+(u/5)-1>=0)&&(i+(u/5)-1<46)&&(j+(u%5)-1)>=0&&(j+(u%5)-1)<46)
                                        error[1][u2][i+(u/5)-1][j+(u%5)-1]+=error[2][k][i][j]*w2[u/5][u%5][u2][k];
                                }
                        }
                    }
                }

                f(i,10)
                    f(j,46)
                        f(k,46)
                            error[1][i][j][k]*=(ld)(network[1][i][j][k]>0);

                ld Dw1[5][5][1][10]={0},Dw2[5][5][10][10]={0},Dw4[5][5][10][7]={0},DwFC[700][7]={0};
                // f(u,7)
                // f(i,error[u].size()){
                //     f(j,error[u][i].size()){
                //         f(k,error[u][i][j].size()){
                //             if(isnan(error[u][i][j][k])){
                //                 assert(0);
                //             }
                //         }
                //     }
                // }

                vvvld temp;
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
                            f(u,25){
                                Dw1[u/5][u%5][0][k]+=error[1][k][i][j]*temp[0][i+(u/5)][j+(u%5)];
                            }
                        }
                    }
                }

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
                            short u2;
                            f(u2,10)
                                f(u,25){
                                    Dw2[u/5][u%5][u2][k]+=error[2][k][i][j]*temp[u2][i+(u/5)][j+(u%5)];
                                }
                        }
                    }
                }
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
                            f(u2,10)
                                f(u,25){
                                    Dw4[u/5][u%5][u2][k]+=error[4][k][i][j]*temp[u2][(2*i)+(u/5)][(2*j)+(u%5)];
                                }
                        }
                    }
                }

                f(i,700){
                    f(j,7){
                        DwFC[i][j]+= error[5][0][0][j]*network[4][i/100][(i/10)%10][i%7];
                    }
                }

                if(image_no%100==0)
                    cout<<image_no<<'\n';
            }
        }
        short k,u,i,j;
        // f(i,5){
        //     f(j,5){
        //         f(k,10){
        //             w1[i][j][0][k]-=alpha*Dw1[i][j][0][k];
        //         }
        //     }
        // }

        // f(i,5){
        //     f(j,5){
        //         f(k,10){
        //             f(u,10){
        //                 w2[i][j][u][k]-=alpha*Dw2[i][j][u][k];
        //             }
        //         }
        //     }
        // }

        // f(i,5){
        //     f(j,5){
        //         f(k,7){
        //             f(u,10){                        
        //                 w4[i][j][u][k]-=alpha*Dw4[i][j][u][k];
        //             }
        //         }
        //     }
        // }
        
        f(i,700){
            f(j,7){
                wFC[i][j]-=alpha*DwFC[i][j];
            }
        }
        
        ofstream pred("pred");
        f(xxx,2){
            omp_set_num_threads(689);
            #pragma omp parallel private(image_no)
            {
                image_no=omp_get_thread_num()+train_size+xxx*689;
                vvvld network[7];
                // network
                network[0].resize(1,vvld(48,vld(48,0)));
                network[1].resize(10,vvld(46,vld(46,0)));
                network[2].resize(10,vvld(44,vld(44,0)));
                network[3].resize(10,vvld(22,vld(22,0)));
                network[4].resize(7,vvld(10,vld(10,0))); // FC 700
                network[5].resize(1,vvld(1,vld(7,0)));
                forward_prop(image_no,network);
                ld maxi=0;
                int emotion=-1;
                f(i,7){
                    maxi=max(maxi,network[5][0][0][i]);
                    pred<<network[5][0][0][i]<<' ';
                }pred<<'\n';
                f(i,7)
                    if(maxi==network[5][0][0][i])
                        emotion=i;
                pred_label[image_no]=emotion;
            }
            cout<<"689 predictions found\n";
        }
        f(image_no,number_of_images-train_size){
            accuracy+=(label[image_no+train_size]==pred_label[image_no+train_size]);
        }
        // accuracy/=(ld)(number_of_images-train_size);
        cout<<accuracy<<'\n';
        encode_weights();
    }

}

int main(){
    ifstream train_data_file("dataset/train.csv");
    string s;
    getline(train_data_file,s);
    ll cur=0;
    // ll anger=0,disgust=0,fear=0,happy=0,sad=0,surprise=0,neutral=0,d=0;
    while(getline(train_data_file,s)){
        label[cur]=s[0]-'0';
        ll i=1,n=s.length(),j=0,kkk=48*48;
        while(kkk--){
            ll n=0,c=s[i++];
            while(c<'0'||c>'9')
                c=s[i++];        
            while(c<='9'&&c>='0'){
                n=n*10+c-'0';
                c=s[i++];
            }
            train[cur][j/48][j%48]= n;
            // train[cur][j/48][j%48]/=1000.0;
            j++;
        }
        cur++;
    }
    // short i;
    // f(i,10)cout<<label[i]<<'\n';
    // Mat image(48,48,CV_8UC1);
    // short i,j;
    // f(i,48){
    //     f(j,48){
    //         image.at<uchar>(i,j) = train[0][i][j];
    //     }
    // }
    // Size size(128,128);//the dst image size,e.g.100x100
    // resize(image,image,size);//resize image
    // namedWindow("image");
    // imshow("image",image);
    // waitKey(0);
    // input done
    
    // tested input


    // random_initialize();
    vvvld network[7];
    // network
    network[0].resize(1,vvld(48,vld(48,0)));
    network[1].resize(10,vvld(46,vld(46,0)));
    network[2].resize(10,vvld(44,vld(44,0)));
    network[3].resize(10,vvld(22,vld(22,0)));
    network[4].resize(7,vvld(10,vld(10,0))); // FC 700
    network[5].resize(1,vvld(1,vld(7,0)));
    forward_prop(1,network);
    short i;
    f(i,7)
    cout<<network[5][0][0][i]<<' ';
    cout<<'\n';
    forward_prop(0,network);
    f(i,7)
    cout<<network[5][0][0][i]<<' ';
    short k;
    back_prop();
}