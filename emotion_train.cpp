#include "template.h"
#define vvvld vector < vector < vector <ld> > >
#define vvld vector <  vector <ld> >
#define vld vector < ld >
/*
model : 7 layers
        1st layer : i/p : 48*48*1
        2nd layer : 5*5*1 conv(p=1,s=1) with 10 filters : 46*46*10
        3rd layer : 5*5*10 conv(p=1,s=1) with 10 filters : 44*44*10
        4th layer : max pool (/2): 22*22*10
        5th layer : 5*5*10 conv(p=1,s=2) with 7 filters : 10*10*7 = FC 700
        6th layer : o/p
*/
#define econst (ld)2.71828182845904523536
#define relu(x) ((ld)(x>0)*x)
#define sigmoid(x) (1/(1+pow(econst,-x)))
#define activation(x) sigmoid(x)

short number_of_images = 4178;
int train[48][48][4178],label[4178];
short train_size=0.75*4178.0;

vvvld network[7],error[7];



void back_prop(){
    // weights
    ld w1[5][5][1][10],w2[5][5][10][10],w4[5][5][10][7];

    // network
    network[0].resize(48,vvld(48,vld(1,0)));
    network[1].resize(46,vvld(46,vld(10,0)));
    network[2].resize(44,vvld(44,vld(10,0)));
    network[3].resize(22,vvld(22,vld(10,0)));
    network[4].resize(10,vvld(10,vld(7,0))); // FC 700
    network[5].resize(7,vvld(1,vld(1,0)));
    
    error[0].resize(48,vvld(48,vld(1,0)));
    error[1].resize(46,vvld(46,vld(10,0)));
    error[2].resize(44,vvld(44,vld(10,0)));
    error[3].resize(22,vvld(22,vld(10,0)));
    error[4].resize(10,vvld(10,vld(7,0))); // FC 700
    error[5].resize(7,vvld(1,vld(1,0)));

    short i,j,number_of_iterations=100;
    while(number_of_iterations--){
        short image_no;
        double accuracy = 0;
        f(image_no,train_size){

        }

        cout<<accuracy<<'\n';
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
            train[j/48][j%48][cur]= n;
            cout<<n<<'\n';
            j++;
        }
        cur++;
    }
    // input done
    
    back_prop();
    
}