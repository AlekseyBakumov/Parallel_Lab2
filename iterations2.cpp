#include <iostream>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <vector>
#include <fstream>
#include <math.h>

#define EPSILON 0.0001
#define THETA 0.01

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

void matrix_vector_product_omp(std::shared_ptr<double[]>a, 
                               std::shared_ptr<double[]>b, 
                               std::shared_ptr<double[]>c, 
                               size_t m, size_t n, int k)
{
#pragma omp parallel num_threads(k)
{
    int nthreads = omp_get_num_threads();
    int threadid = omp_get_thread_num();
    int items_per_thread = m / nthreads;
    int lb = threadid * items_per_thread;
    int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
    for (int i = lb; i <= ub; i++)
    {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}
}

double vectorLen(std::shared_ptr<double[]>u, int n)
{
    double result = 0.0;

    for (int i = 0; i < n; i++)
        result += u[i] * u[i];

    result = sqrt(result);
    return result;
}

double criteria(std::shared_ptr<double[]>a, 
                std::shared_ptr<double[]>b,
                std::shared_ptr<double[]>c,
                size_t m, size_t n, int k)
{
    std::shared_ptr<double[]> u(new double[n]);
    matrix_vector_product_omp(a,c,u,n,m,k);

    for (int i = 0; i < n; i++)
        u[i] -= b[i];
    double result = vectorLen(u,n) / vectorLen(b,n);
    return result;
}

double run_parallel(size_t n, size_t m, int k)
{
    std::shared_ptr<double[]> a(new double[m * n]);
    std::shared_ptr<double[]> ax(new double[n]);
    std::shared_ptr<double[]> b(new double[n]);
    std::shared_ptr<double[]> c(new double[m]);
    std::shared_ptr<double[]> u(new double[n]);

    if (a == NULL || b == NULL || c == NULL)
    {
        std::cout << "Error allocate memory!\n" << std::endl;
        exit(1);
    }

    for (int i = 0; i < m; i++)
        c[i] = 0.0; 

    for (int i = 0; i < n; i++)
        b[i] = n + 1;

    for (int i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
            if (i == j) a[i * n + j] = 2.0;
            else        a[i * n + j] = 1.0;
    }

    double cr = 1000;
    double t = cpuSecond();
    while (cr > EPSILON)
    {    
        //matrix_vector_product_omp(a,c,ax,m,n,k);
        #pragma omp parallel for num_threads(k) schedule(guided,k)
        for (int i = 0; i < m; i++)
        {
            ax[i] = 0.0;
            for (int j = 0; j < n; j++)
                ax[i] += a[i * n + j] * c[j];
        }

        #pragma omp parallel for num_threads(k) schedule(guided,k)
        for (int i = 0; i < m; i++)
            c[i] = c[i] - THETA * (ax[i] - b[i]);

        //cr = criteria(a, b, c, m, n, k);
        //matrix_vector_product_omp(a,c,u,n,m,k);
        #pragma omp parallel for num_threads(k) schedule(guided,k)
        for (int i = 0; i < n; i++)
        {
            u[i] = 0.0;
            for (int j = 0; j < m; j++)
                u[i] += a[i * m + j] * c[j];
        }    

        #pragma omp parallel for num_threads(k) schedule(guided,k)
        for (int i = 0; i < n; i++)
            u[i] -= b[i];
        cr = vectorLen(u,n) / vectorLen(b,n);
        //printf("Error: %.8f.\n", cr);
    }
    t = cpuSecond() - t;
    
    //double mean_error = 0.0;
    //for (int i = 0; i < m; i++)
    //   mean_error += abs(c[i] - 1.0);
    //mean_error /= m;

    printf("Elapsed time (parallel %d threads): %.6f sec.\n", k, t);
    //printf("Avarage error on x[i]: %.8f.\n", mean_error);

    return t;
}

double avg_time_parallel(size_t n, size_t m, int k, int runs)
{
    double time = 0;
    run_parallel(n,m,k);
    for (int i = 0; i < runs; i++)
    {
            time += run_parallel(n,m,k);
    }

    return time / runs;
}

int main(int argc, char *argv[])
{
    std::cout << "Start..." << std::endl;

    int runs = 3;

    size_t M = 3000;
    size_t N = 3000;
    if (argc > 1)
        M = atoi(argv[1]);
    if (argc > 2)
        N = atoi(argv[2]);

    int threads[] = {2,4,7,8,16,20,40,60,80};
    double single_thread_time = avg_time_parallel(M, N, 1, runs);
    double time;

    std::ofstream out_file;
    out_file.open("results_itrs_for.csv");

    out_file << 1 << "," << single_thread_time << "," << 1 << std::endl;

    for (int tr : threads)
    {
        time = avg_time_parallel(M, N, tr, runs);
        out_file << tr << "," << time << "," << single_thread_time / time << std::endl;
    }

    return 0;
}