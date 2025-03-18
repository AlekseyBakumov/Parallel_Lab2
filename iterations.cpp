#include <iostream>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <vector>
#include <fstream>
#include <math.h>

#define EPSILON 0.0001//e-4
#define THETA 0.01//e-3

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

/*
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
*/

double vectorLen(std::shared_ptr<double[]>u, int n)
{
    double result = 0.0;

    for (int i = 0; i < n; i++)
        result += u[i] * u[i];

    result = sqrt(result);
    return result;
}

/*
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
*/

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

    double cr = 1000;
    double t = cpuSecond();
    int iterations = 0;
    
    #pragma omp parallel num_threads(k)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();

        int items_per_thread = n / nthreads;
        int lb_n = threadid * items_per_thread;
        int ub_n = (threadid == nthreads - 1) ? (n - 1) : (lb_n + items_per_thread - 1);

        
        items_per_thread = m / nthreads;
        int lb_m = threadid * items_per_thread;
        int ub_m = (threadid == nthreads - 1) ? (m - 1) : (lb_m + items_per_thread - 1);
        

        for (int i = lb_n; i <= ub_n; i++)
            b[i] = n + 1;

        for (int i = lb_n; i <= ub_n; i++)
        {
            for (size_t j = 0; j < m; j++)
                if (i == j) a[i * m + j] = 2.0;
                else        a[i * m + j] = 1.0;
        }

        for (int i = lb_m; i <= ub_m; i++)
                c[i] = 0.0; 

        //printf("N: (%d , %d), thread №%d.\n", lb_n, ub_n, threadid);
        //printf("M: (%d , %d), thread №%d.\n", lb_m, ub_m, threadid);

        #pragma omp barrier 
        while (cr > EPSILON)
        {
            if (threadid == 0) iterations++;

            //matrix_vector_product_omp(a,c,ax,m,n,k);
            for (int i = lb_n; i <= ub_n; i++)
            {
                ax[i] = 0.0;
                for (int j = 0; j < m; j++)
                    ax[i] += a[i * m + j] * c[j];
            }
            #pragma omp barrier



            for (int i = lb_m; i <= ub_m; i++)
                c[i] = c[i] - THETA * (ax[i] - b[i]);
            #pragma omp barrier


            
            //cr = criteria(a, b, c, m, n, k);
            //matrix_vector_product_omp(a,c,u,n,m,k);
            for (int i = lb_n; i <= ub_n; i++)
            {
                u[i] = -b[i]; // 0
                for (int j = 0; j < m; j++)
                    u[i] += a[i * m + j] * c[j];
            }    
            #pragma omp barrier



            //for (int i = lb_n; i <= ub_n; i++)
                //u[i] -= b[i];
            #pragma omp single
            cr = vectorLen(u,n) / vectorLen(b,n);
            #pragma omp barrier



            //printf("Error: %.8f.\n", cr);
        }
    }
    t = cpuSecond() - t;
    
    double mean_error = 0.0;
    for (int i = 0; i < m; i++)
        mean_error += abs(c[i] - 1.0);
    mean_error /= m;

    printf("Elapsed time (parallel %d threads): %.6f sec.\n", k, t);
    printf("Avarage error on x[i]: %.8f.\n", mean_error);
    printf("Iterations: %d.\n", iterations);

    return t;
}

double avg_time_parallel(size_t n, size_t m, int k, int runs)
{
    double time = 0;
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

    size_t M = 10000;
    size_t N = 10000;
    if (argc > 1)
        M = atoi(argv[1]);
    if (argc > 2)
        N = atoi(argv[2]);

    int threads[] = {2,4,7,8,16,20,40,60,80};
    double single_thread_time = avg_time_parallel(M, N, 1, runs);
    double time;

    std::ofstream out_file;
    out_file.open("results_itrs.csv");

    out_file << 1 << "," << single_thread_time << "," << 1 << std::endl;

    for (int tr : threads)
    {
        time = avg_time_parallel(M, N, tr, runs);
        out_file << tr << "," << time << "," << single_thread_time / time << std::endl;
    }

    return 0;
}