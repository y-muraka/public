#include <vector>
#include <fstream>

using namespace::std;

void save_datfile(string filename, vector<vector<double> > data)
{
    ofstream outFile;
    int size_x = data.size();
    int size_y = data.at(0).size();
    
    outFile.open(filename, std::ios::binary);
    for (int ii=0; ii < size_x; ii++)
        for (int jj=0; jj < size_y; jj++)
            outFile.write(reinterpret_cast<const char*>(&data[ii][jj]), sizeof(double));
    outFile.close();
}

void save_datfile(string filename, vector<double> data)
{
    ofstream outFile;
    int size_x = data.size();
    
    outFile.open(filename, std::ios::binary);
    for (int ii=0; ii < size_x; ii++)
            outFile.write(reinterpret_cast<const char*>(&data[ii]), sizeof(double));
    outFile.close();
}

vector<double> mat_to_mat1d(vector<vector<double> > mat)
{
    int size_x = mat.size();
    int size_y = mat.at(0).size();
    vector<double> mat1d(size_x*size_y);

    for (int ii = 0; ii < size_x; ii++)
        for (int jj = 0; jj < size_y; jj++)
            mat1d[ii*mat[0].size() + jj] = mat[ii][jj];

    return mat1d;
}