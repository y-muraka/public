#include "../CochlearModel_1D.h"

using namespace std;

extern "C" int dgetrf_(int *m, int *n, double *a, int * lda, int *ipiv, int *info);
extern "C" int dgetri_(int *n, double *a, int *lda, int	*ipiv, double *work, int *lwork, int *info);


vector<double> inv(vector<double> mat1d_input)
{
    vector<double> mat1d_output = mat1d_input;

    double* ptr_output = mat1d_output.data();

    int size = (int)sqrt(mat1d_output.size());
    int m = size;
    int n = size;
    int lda = size;
    int info;
    int* ipiv = new int[size];
    int lwork = size;
    double* work = new double[size];

    dgetrf_( &m, &n, ptr_output, &lda, ipiv, &info);
    dgetri_(&n, ptr_output, &lda, ipiv, work, &lwork, &info);

    delete[] ipiv;
    delete[] work;

    return mat1d_output;
}

void matvec_mul(vector<double> mat1d_in, vector<double> vec_in, vector<double>& vec_out)
{
    double alpha = 1.0;
    double beta = 0.0;

    int dim = vec_in.size();

    cblas_dgemv(CblasRowMajor, CblasNoTrans, dim, dim, alpha, mat1d_in.data(), dim, vec_in.data(), 1, beta, vec_out.data(), 1);
}

void CochlearModel_1D::get_g(vector<double> vb, vector<double> ub, vector<double> vt, vector<double> ut, vector<double>& gb, vector<double>& gt)
{
    double uc_lin, vc_lin;
    double uc, vc, coshx;

    for (int ii=0; ii < N; ii++){
        gb[ii] = c1c3[ii]*vb[ii] + k1k3[ii]*ub[ii] - c3[ii]*vt[ii] - k3[ii]*ut[ii];
        gt[ii] = -c3[ii]*vb[ii] - k3[ii]*ub[ii] + c2c3[ii]*vt[ii] + k2k3[ii]*ut[ii];
        uc_lin = ub[ii] - ut[ii];
        vc_lin = vb[ii] - vt[ii];

        uc_lin = ub[ii] - ut[ii];
        vc_lin = vb[ii] - vt[ii];

        uc = lookup(table_tanh, uc_lin);
        coshx = lookup(table_cosh, uc_lin);
        vc = vc_lin/(coshx*coshx);

        gb[ii] -= gamma[ii] * (c4[ii] * vc + k4[ii] * uc);
    }
}

void CochlearModel_1D::solve_time_domain(vector<double> vec_f, vector<vector<double> >& mat_vb, vector<vector<double> >& mat_ub, vector<vector<double> >& mat_p)
{
    int num_time = (int)round(vec_f.size()/2);
    double alpha2 = 2*rho/H/m1;
    vector<vector<double> > mat_vt(num_time, vector<double>(N,0));
    vector<vector<double> > mat_ut(num_time, vector<double>(N,0));
    vector<vector<double> > mat_F(N, vector<double>(N,0));

    struct timeval tv_start, tv_stop;
    double deltaT;
    double dx2 = pow(dx, 2.0);

    mat_F[0][0] = -2/dx2 - alpha2;
    mat_F[0][1] = 2/dx2;
    for (int mm = 1; mm < N-1; mm++){
        mat_F[mm][mm-1] = 1/dx2;
        mat_F[mm][mm] = -2/dx2 - alpha2;
        mat_F[mm][mm+1] = 1/dx2;
    }
    mat_F[N-1][N-2] = 1/dx2;
    mat_F[N-1][N-1] = -2/dx2 - alpha2;

    vector<double> mat1d_F = mat_to_mat1d(mat_F);

    vector<double> mat1d_iF = inv(mat1d_F);

    vector<double> vec_gb(N);
    vector<double> vec_gt(N);
    vector<double> vec_k(N);

    vector<double> vec_dvb1(N), vec_dvb2(N), vec_dvb3(N), vec_dvb4(N);
    vector<double> vec_dvt1(N), vec_dvt2(N), vec_dvt3(N), vec_dvt4(N);
    vector<double> vec_vb1(N), vec_vb2(N), vec_vb3(N);
    vector<double> vec_vt1(N), vec_vt2(N), vec_vt3(N);
    vector<double> vec_ub1(N), vec_ub2(N), vec_ub3(N);
    vector<double> vec_ut1(N), vec_ut2(N), vec_ut3(N);
    vector<double> vec_p1(N), vec_p2(N), vec_p3(N);

    int mm;
    gettimeofday(&tv_start, NULL);
    for (int ii = 0; ii < num_time-1; ii++){
        // RK4

        // (ii)
        get_g(mat_vb[ii], mat_ub[ii], mat_vt[ii], mat_ut[ii], vec_gb, vec_gt);

        for (mm = 0 ; mm < N; mm++)
            vec_k[mm] = -alpha2*vec_gb[mm];
        vec_k[0] -= vec_f[ii*2] * 2/dx;

        // (iii)
        matvec_mul(mat1d_iF, vec_k, mat_p[ii]);

        /// (iv)
        for ( mm=0; mm < N; mm++){
            vec_dvb1[mm] = (mat_p[ii][mm] - vec_gb[mm])/m1;
            vec_ub1[mm] = mat_ub[ii][mm] + 0.5*dt*mat_vb[ii][mm];
            vec_vb1[mm] = mat_vb[ii][mm] + 0.5*dt*vec_dvb1[mm];

            vec_dvt1[mm] = -vec_gt[mm]/m2;
            vec_ut1[mm] = mat_ut[ii][mm] + 0.5*dt*mat_vt[ii][mm];
            vec_vt1[mm] = mat_vt[ii][mm] + 0.5*dt*vec_dvt1[mm];
        }

        // (ii)
        get_g(vec_vb1, vec_ub1, vec_vt1, vec_ut1, vec_gb, vec_gt);

        for (mm = 0 ; mm < N; mm++)
            vec_k[mm] = -alpha2*vec_gb[mm];
        vec_k[0] -= vec_f[ii*2+1] * 2/dx;

        // (iii)
        matvec_mul(mat1d_iF, vec_k, vec_p1);

        /// (iv)

        for ( mm=0; mm < N; mm++){
            vec_dvb2[mm] = (vec_p1[mm] - vec_gb[mm])/m1;
            vec_ub2[mm] = vec_ub1[mm] + 0.5*dt*vec_vb1[mm];
            vec_vb2[mm] = vec_vb1[mm] + 0.5*dt*vec_dvb2[mm];

            vec_dvt2[mm] = -vec_gt[mm]/m2;
            vec_ut2[mm] = vec_ut1[mm] + 0.5*dt*vec_vt1[mm];
            vec_vt2[mm] = vec_vt1[mm] + 0.5*dt*vec_dvt2[mm];
        }

        // (ii)
        get_g(vec_vb2, vec_ub2, vec_vt2, vec_ut2, vec_gb, vec_gt);


        for (mm = 0 ; mm < N; mm++)
            vec_k[mm] = -alpha2*vec_gb[mm];
        vec_k[0] -= vec_f[ii*2+1] * 2/dx;

        // (iii)
        matvec_mul(mat1d_iF, vec_k, vec_p2);

        /// (iv)

        for ( mm=0; mm < N; mm++){
            vec_dvb3[mm] = (vec_p2[mm] - vec_gb[mm])/m1;
            vec_ub3[mm] = vec_ub2[mm] + dt*vec_vb2[mm];
            vec_vb3[mm] = vec_vb2[mm] + dt*vec_dvb3[mm];

            vec_dvt3[mm] = -vec_gt[mm]/m2;
            vec_ut3[mm] = vec_ut2[mm] + dt*vec_vt2[mm];
            vec_vt3[mm] = vec_vt2[mm] + dt*vec_dvt3[mm];

        }

        // (ii)
        get_g(vec_vb3, vec_ub3, vec_vt3, vec_ut3, vec_gb, vec_gt);

        for (mm = 0 ; mm < N; mm++)
            vec_k[mm] = -alpha2*vec_gb[mm];
        vec_k[0] -= vec_f[ii*2+2] * 2/dx;

        // (iii)
        matvec_mul(mat1d_iF, vec_k, vec_p3);

        /// (iv)
        for ( mm=0; mm < N; mm++){
            vec_dvb4[mm] = (vec_p3[mm] - vec_gb[mm])/m1;
            vec_dvt4[mm] = -vec_gt[mm] / m2;

            mat_ub[ii+1][mm] = mat_ub[ii][mm] + dt/6*(mat_vb[ii][mm] + 2*vec_vb1[mm] + 2*vec_vb2[mm] + vec_vb3[mm]);
            mat_vb[ii+1][mm] = mat_vb[ii][mm] + dt/6*(vec_dvb1[mm] + 2*vec_dvb2[mm] + 2*vec_dvb3[mm] + vec_dvb4[mm]);
            mat_ut[ii+1][mm] = mat_ut[ii][mm] + dt/6*(mat_vt[ii][mm] + 2*vec_vt1[mm] + 2*vec_vt2[mm] + vec_vt3[mm]);
            mat_vt[ii+1][mm] = mat_vt[ii][mm] + dt/6*(vec_dvt1[mm] + 2*vec_dvt2[mm] + 2*vec_dvt3[mm] + vec_dvt4[mm]);
        }
    }
    gettimeofday(&tv_stop, NULL);
    deltaT =  tv_stop.tv_sec  - tv_start.tv_sec +
    1e-6 * (tv_stop.tv_usec - tv_start.tv_usec);

    printf("Duration of stimulation: %f [sec] \n", num_time*dt);
    printf("Elapsed time: %f [sec] \n", deltaT);
}

vector<double> get_f(string filename, int Lp)
{
    double Ap = pow(10, Lp/20)*1000;
    AudioFile<double> audioFile;
    audioFile.load(filename);

    int num_time_in = audioFile.getNumSamplesPerChannel();

    vector<double> vec_f(num_time_in, 0);

    for (int ii=0; ii < num_time_in; ii++)
        vec_f[ii] = Ap * audioFile.samples[0][ii];

    return vec_f;
}
/*
vector<double> get_f(double fp, double Ap, int num_time)
{
    vector<double> vec_f(num_time, 0);
    double dt = 5e-6;
    double t;

    for (int ii = 0; ii < num_time; ii++){
        t = dt*ii;
        vec_f[ii] = Ap*sin(2*M_PI*fp*t);
    }

    return vec_f;
}
*/

int main(int argc, char *argv[])
{
    string filename  = argv[1];
    int num_segment = stoi(argv[2]);
    int Lp = stoi(argv[3]);

    char filename_out[50];

    CochlearModel_1D cm(num_segment);
    vector<double> vec_f = get_f(filename, Lp);

    vector<vector<double> > mat_vb(int(vec_f.size()/2), vector<double>(num_segment));
    vector<vector<double> >  mat_ub(int(vec_f.size()/2), vector<double>(num_segment));
    vector<vector<double> > mat_p(int(vec_f.size()/2), vector<double>(num_segment));
    
    cm.solve_time_domain(vec_f, mat_vb, mat_ub, mat_p);

    sprintf(filename_out, "../tmp/direct/mat_vb_%d_%d.dat", num_segment, Lp);
    save_datfile(filename_out, mat_vb);
}


/*
int main(void)
{
    openblas_set_num_threads(8);
    int num_segment = 500;
    int num_time = 10000;
    int num_time2 = num_time * 2;
    double fp = 1000.0;

    vector<double> gamma(num_segment,1.0);

    CochlearModel_1D cm(num_segment, gamma);

    vector<vector<double> > mat_vb(num_time, vector<double>(num_segment));
    vector<vector<double> >  mat_ub(num_time, vector<double>(num_segment));
    vector<vector<double> > mat_p(num_time, vector<double>(num_segment));
    vector<double> vec_f = get_f(fp, num_time2);

    cm.solve_time_domain(vec_f, mat_vb, mat_ub, mat_p);

    save_datfile("mat_vb.dat", mat_vb);
}
*/