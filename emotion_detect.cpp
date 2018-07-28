#include "template.h"
ld w1[5][5][1][10],w2[5][5][10][10],w4[5][5][10][7],wFC[700][7];
void read_weights(){
    ifstream weights("weights");
    short i,j,k,u;
    f(i,5)f(j,5)f(k,10)weights>>w1[i][j][0][k];
    f(i,5)f(j,5)f(k,10)f(u,10)weights>>w2[i][j][k][u];
    f(i,5)f(j,5)f(k,10)f(u,7)weights>>w4[i][j][k][u];
    f(i,700)f(j,7)weights>>wFC[i][j];
}
int main(){
    
}