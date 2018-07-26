#include "template.h"

int main(){
    ifstream train;
    train.open("dataset/train.csv");
    string s;
    getline(train,s);
    ll number_of_images=0;
    ll anger=0,disgust=0,fear=0,happy=0,sad=0,surprise=0,neutral=0,d=0;
    while(getline(train,s)){
        number_of_images++;
        switch(s[0]-'0'){
            case 0: anger++;
                    break;
            case 1: disgust++;
                    break;
            case 2: fear++;
                    break;
            case 3: happy++;
                    break;
            case 4: sad++;
                    break;
            case 5: surprise++;
                    break;
            case 6: neutral++;
                    break;
            default: d++;
        }
    }

    cout<<"number of images: "<<number_of_images<<'\n';
    cout<<"anger "<<anger<<'\n';
    cout<<"disgust "<<disgust<<'\n';
    cout<<"fear "<<fear<<'\n';
    cout<<"happy "<<happy<<'\n';
    cout<<"sad "<<sad<<'\n';
    cout<<"surprise "<<surprise<<'\n';
    cout<<"neutral "<<neutral<<'\n';
    cout<<anger+disgust+fear+happy+sad+surprise+neutral<<'\n';
    return 0;
}