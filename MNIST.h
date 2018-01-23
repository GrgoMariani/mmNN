#pragma once
#include <fstream>
#include <math.h>
/*  https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
       code by dariush
    Download MNIST from
        http://yann.lecun.com/exdb/mnist/
*/
//! READ IMAGES
// C:/MNIST/t10k-images.idx3-ubyte
// C:/MNIST/train-images.idx3-ubyte
typedef unsigned char uchar;
using namespace std;

uchar** read_mnist_images(string full_path, int& number_of_images, int& image_size) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        uchar** _dataset = new uchar*[number_of_images];
        for(int i = 0; i < number_of_images; i++) {
            _dataset[i] = new uchar[image_size];
            file.read((char *)_dataset[i], image_size);
        }
        return _dataset;
    } else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}


//! READ LABELS
// C:/MNIST/t10k-labels.idx1-ubyte
// C:/MNIST/train-labels.idx1-ubyte

uchar* read_mnist_labels(string full_path, int& number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        return _dataset;
    } else {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
}


//! IMAGE OPERATIONS
//Randomize images slightly, either by rotating or moving the whole image
vector<double> MoveImageR(vector<double> image){
    int howmuch=rand()%5;
    vector<double> result;
    for(int i=0; i<28; i++){
        int j=0;
        while((j++)<howmuch)
            result.push_back(0.);
        for(j=0; j<28-howmuch; j++)
            result.push_back(image[i*28+j]);
    }
    return result;
}
vector<double> MoveImageL(vector<double> image){
    int howmuch=rand()%5;
    vector<double> result;
    for(int i=0; i<28; i++){
        for(int j=0; j<28-howmuch; j++)
            result.push_back(image[i*28+j+howmuch]);
        int j=0;
        while((j++)<howmuch)
            result.push_back(0.);
    }
    return result;
}
vector<double> MoveImageU(vector<double> image){
    int howmuch=rand()%5;
    vector<double> result;
    for(int i=0; i<28-howmuch; i++){
        for(int j=0; j<28; j++)
            result.push_back(image[(i+howmuch)*28+j]);
    }
    int j=0;
    while((j++)<howmuch)
        for(int i=0; i<28; i++)
            result.push_back(0.);
    return result;
}
vector<double> MoveImageD(vector<double> image){
    int howmuch=rand()%5;
    vector<double> result;
    int j=0;
    while((j++)<howmuch)
        for(int i=0; i<28; i++)
            result.push_back(0.);
    for(int i=0; i<28-howmuch; i++){
        for(int j=0; j<28; j++)
            result.push_back(image[(i)*28+j]);
    }
    return result;
}
double PixelOnXY(double x, double y, vector<double> image){
    if( x<0 || y<0 || x>27 || y>27)
        return 0.;
    int xcoord=(int)x;
    int ycoord=(int)y;
    double xperc=xcoord+1.-x;
    double yperc=ycoord+1.-y;
    double result=xperc*image[28*ycoord+xcoord]*yperc;
    result+=(1.-xperc)*image[28*ycoord+xcoord+1]*yperc;
    result+=xperc*image[28*ycoord+xcoord+28]*(1.-yperc);
    result+=(1.-xperc)*image[28*ycoord+xcoord+28+1]*(1.-yperc);
    return result;
}
vector<double> RotateImage(vector<double> image, int angle){
    double _angle=angle*PI/180.;
    vector<double> result;
    for(int i=0; i<28; i++)
        for(int j=0; j<28; j++)
            result.push_back(PixelOnXY(13.5-(i-13.5)*sin(_angle)+(j-13.5)*cos(_angle), 13.5+(i-13.5)*cos(_angle)+(j-13.5)*sin(_angle), image));
    return result;
}
vector<double> DistortImage(vector<double> image){
    vector<double> result;
    result=RotateImage(image, rand()%70-35);
    if(rand()%2==0) result=MoveImageL(result);
    else result=MoveImageR(result);
    if(rand()%2==0) result=MoveImageU(result);
    else result=MoveImageD(result);
    return result;
}
void PrintImage(vector<double> image){
    for(int i=0; i<image.size(); i++){
        int x=(int)image[i];
        if(x==0) cout<<" ";
        else if(x<64) cout<<(char)176;
        else if(x<128) cout<<(char)177;
        else if(x<192) cout<<(char)178;
        else cout<<(char)219;
        if(i%28==0 && i!=0 )
            cout<<endl;
    }
    cout<<endl;
}

vector<double> ReadRawImage(uchar **images, int index){
    vector<double> result;
    for(int i=0; i<28*28; i++){
        result.push_back((double)images[index][i]);
    }
    return result;
}
vector<double> ReadImage(uchar **images, int index){
    return DistortImage(ReadRawImage(images, index));
}
