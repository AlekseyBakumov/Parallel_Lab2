#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <fstream>

const double PI = 3.14159265358979323846;
const double a = -4.0;
const double b = 4.0;
const int nsteps = 40000000;

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

double func(double x)
{
    return exp(-x * x);
}

double integrate(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

    for (int i = 0; i < n; i++)
        sum += func(a + h * (i + 0.5));

    sum *= h;

    return sum;
}

double integrate_omp(double (*func)(double), double a, double b, int n, int k)
{
    double h = (b - a) / n;
    double sum = 0.0;

#pragma omp parallel num_threads(k)
    {
        double sum_local = 0.0;

        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; i++)
            sum_local += func(a + h * (i + 0.5));

        #pragma omp atomic
        sum += sum_local;
    }
    sum *= h;

    return sum;
}

double run_serial()
{
    double t = cpuSecond();
    double res = integrate(func, a, b, nsteps);
    t = cpuSecond() - t;
    printf("Result (serial): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}
double run_parallel(int threads)
{
    double t = cpuSecond();
    double res = integrate_omp(func, a, b, nsteps, threads);
    t = cpuSecond() - t;
    //printf("Result (parallel): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

double avg_time_parallel(int threads, int runs)
{
    double time = 0;
    for (int i = 0; i < runs; i++)
    {
        time += run_parallel(threads);
    }

    return time / runs;
}

int main(int argc, char **argv)
{
    printf("Integration f(x) on [%.12f, %.12f], nsteps = %d\n", a, b, nsteps);
    double tserial = run_serial();
    double tparallel;
    printf("Execution time (serial): %.6f\n", tserial);


    int threads[] = {2,4,7,8,16,20,40};

    std::ofstream out_file;
    out_file.open("results_integral.csv");

    out_file << "threads" << "," << "time" << "," << "speedup" << std::endl;
    out_file << 1 << "," << tserial << "," << 1 << std::endl;

    for (int tr : threads)
    {
        tparallel = avg_time_parallel(tr, 30);
        printf("\nRunning on %d threads\n", tr);
        printf("Execution time (parallel): %.6f\n", tparallel);
        printf("Speedup: %.2f\n", tserial / tparallel);
        out_file << tr << "," << tparallel << "," << tserial / tparallel << std::endl;
    }

    
    return 0;
}