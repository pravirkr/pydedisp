
/*
  Simple test application for libdedisp
  By Paul Ray (2013)
*/

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <random>
#include <algorithm>

#include <dedisp/dedisp.hpp>

// Assume input is a 0 mean float and quantize to an unsigned 8-bit quantity
dedisp_byte bytequant(dedisp_float num) {
    dedisp_byte r;
    r = std::round(num + 127.5f);
    return std::max(static_cast<unsigned char>(0),
                    std::min(r, static_cast<unsigned char>(255)));
}

// Compute mean and standard deviation of an unsigned 8-bit array
void calc_stats_8bit(dedisp_byte* a,
                     dedisp_size n,
                     dedisp_float* mean,
                     dedisp_float* sigma) {
    // Use doubles to prevent rounding error
    double sum = 0.0, sum2 = 0.0;
    double mtmp = 0.0, vartmp;
    double v;
    dedisp_size i;

    // Use corrected 2-pass algorithm from Numerical Recipes
    sum = 0.0;
    for (i = 0; i < n; i++) {
        v = (double)a[i];
        sum += v;
    }
    mtmp = sum / n;

    sum  = 0.0;
    sum2 = 0.0;
    for (i = 0; i < n; i++) {
        v = (double)a[i];
        sum2 += (v - mtmp) * (v - mtmp);
        sum += v - mtmp;
    }
    vartmp = (sum2 - (sum * sum) / n) / (n - 1);
    *mean  = mtmp;
    *sigma = sqrt(vartmp);

    return;
}

// Compute mean and standard deviation of a float array
void calc_stats_float(dedisp_float* a,
                      dedisp_size n,
                      dedisp_float* mean,
                      dedisp_float* sigma) {
    // Use doubles to prevent rounding error
    double sum = 0.0, sum2 = 0.0;
    double mtmp = 0.0, vartmp;
    double v;
    dedisp_size i;

    // Use corrected 2-pass algorithm from Numerical Recipes
    sum = 0.0;
    for (i = 0; i < n; i++) {
        sum += a[i];
    }
    mtmp = sum / n;

    sum  = 0.0;
    sum2 = 0.0;
    for (i = 0; i < n; i++) {
        v = a[i];
        sum2 += (v - mtmp) * (v - mtmp);
        sum += v - mtmp;
    }
    vartmp = (sum2 - (sum * sum) / n) / (n - 1);
    *mean  = mtmp;
    *sigma = sqrt(vartmp);

    return;
}

int main(int argc, char* argv[]) {
    int device_idx = 0;

    // Base is 250 microsecond time samples
    dedisp_float sampletime_base = 250.0E-6;
    dedisp_float downsamp        = 1.0;
    dedisp_float Tobs            = 30.0;  // Observation duration in seconds
    dedisp_float dt    = downsamp * sampletime_base;  // s (0.25 ms sampling)
    dedisp_float f0    = 1581.0;                      // MHz (highest channel!)
    dedisp_float bw    = 100.0;                       // MHz
    dedisp_size nchans = 1024;
    dedisp_float df    = -1.0 * bw / nchans;  // MHz   (This must be negative!)

    dedisp_size nsamps   = Tobs / dt;
    dedisp_float datarms = 25.0;
    dedisp_float sigDM   = 41.159;
    dedisp_float sigT    = 3.14159;  // seconds into time series (at f0)
    dedisp_float sigamp  = 25.0;     // amplitude of signal

    dedisp_float dm_start    = 2.0;    // pc cm^-3
    dedisp_float dm_end      = 100.0;  // pc cm^-3
    dedisp_float pulse_width = 4.0;    // ms
    dedisp_float dm_tol      = 1.25;
    dedisp_size in_nbits     = 8;

    dedisp_plan plan;
    dedisp_error error;
    dedisp_size dm_count;
    dedisp_size max_delay;
    dedisp_size nsamps_computed;

    dedisp_byte* input   = 0;
    dedisp_float* output = 0;
    const dedisp_float* dmlist;
    std::vector<dedisp_float> rawdata(nsamps * nchans, 0);
    std::vector<dedisp_float> delay_s(nchans, 0);

    clock_t startclock;

    printf("----------------------------- INPUT DATA "
           "---------------------------------\n");
    printf("Frequency of highest chanel (MHz)            : %.4f\n", f0);
    printf("Bandwidth (MHz)                              : %.2f\n", bw);
    printf("NCHANS (Channel Width [MHz])                 : %lu (%f)\n", nchans,
           df);
    printf("Sample time (after downsampling by %.0f)        : %f\n", downsamp,
           dt);
    printf("Observation duration (s)                     : %f (%lu samples)\n",
           Tobs, nsamps);
    printf("Data RMS (%2lu bit input data)                 : %f\n", in_nbits,
           datarms);
    printf("Input data array size                        : %lu MB\n",
           (nsamps * nchans * sizeof(float)) / (1 << 20));
    printf("\n");

    /* First build 2-D array of floats with our signal in it */
    std::random_device rd;
    // Create and seed the generator (32-bit Mersenne Twister engine)
    std::mt19937 gen(rd());
    // Create distribution
    std::normal_distribution<> dist(0, datarms);

    for (size_t idata = 0; idata != rawdata.size(); ++idata) {
        rawdata[idata] = dist(gen);
    }

    /* Now embed a dispersed pulse signal in it */
    for (size_t ichan = 0; ichan != delay_s.size(); ++ichan) {
        dedisp_float a = 1.f / (f0 + ichan * df);
        dedisp_float b = 1.f / f0;
        delay_s[ichan] = sigDM * 4.15e3 * (a * a - b * b);
    }

    int ns;
    printf("Embedding signal\n");
    for (size_t ichan = 0; ichan < nchans; ++ichan) {
        ns = (int)((sigT + delay_s[ichan]) / dt);
        if (ns > nsamps) {
            printf("ns too big %u\n", ns);
            exit(1);
        }
        rawdata[ns * nchans + ichan] += sigamp;
    }

    printf("----------------------------- INJECTED SIGNAL  "
           "----------------------------\n");
    printf("Pulse time at f0 (s)                      : %.6f (sample %lu)\n",
           sigT, (dedisp_size)(sigT / dt));
    printf("Pulse DM (pc/cm^3)                        : %f \n", sigDM);
    printf("Signal Delays : %f, %f, %f ... %f\n", delay_s[0], delay_s[1],
           delay_s[2], delay_s[nchans - 1]);
    /*
       input is a pointer to an array containing a time series of length
       nsamps for each frequency channel in plan. The data must be in
       time-major order, i.e., frequency is the fastest-changing
       dimension, time the slowest. There must be no padding between
       consecutive frequency channels.
     */

    dedisp_float raw_mean, raw_sigma;
    calc_stats_float(&rawdata[0], nsamps * nchans, &raw_mean, &raw_sigma);
    printf("Rawdata Mean (includes signal)    : %f\n", raw_mean);
    printf("Rawdata StdDev (includes signal)  : %f\n", raw_sigma);
    printf("Pulse S/N (per frequency channel) : %f\n", sigamp / datarms);

    input = (dedisp_byte*)std::malloc(nsamps * nchans * (in_nbits / 8));

    printf("Quantizing array\n");
    /* Now fill array by quantizing rawdata */
    for (size_t ns = 0; ns < nsamps; ns++) {
        for (size_t nc = 0; nc < nchans; nc++) {
            input[ns * nchans + nc] = bytequant(rawdata[ns * nchans + nc]);
        }
    }

    dedisp_float in_mean, in_sigma;
    calc_stats_8bit(input, nsamps * nchans, &in_mean, &in_sigma);

    printf("Quantized data Mean (includes signal)    : %f\n", in_mean);
    printf("Quantized data StdDev (includes signal)  : %f\n", in_sigma);
    printf("\n");

    printf("Init GPU\n");
    // Initialise the GPU
    error = dedisp_set_device(device_idx);
    if (error != DEDISP_NO_ERROR) {
        printf("ERROR: Could not set GPU device: %s\n",
               dedisp_get_error_string(error));
        return -1;
    }

    printf("Create plan\n");
    // Create a dedispersion plan
    error = dedisp_create_plan(&plan, nchans, dt, f0, df);
    if (error != DEDISP_NO_ERROR) {
        printf("\nERROR: Could not create dedispersion plan: %s\n",
               dedisp_get_error_string(error));
        return -1;
    }

    printf("Gen DM list\n");
    // Generate a list of dispersion measures for the plan
    error
        = dedisp_generate_dm_list(plan, dm_start, dm_end, pulse_width, dm_tol);
    if (error != DEDISP_NO_ERROR) {
        printf("\nERROR: Failed to generate DM list: %s\n",
               dedisp_get_error_string(error));
        return -1;
    }

    // Find the parameters that determine the output size
    dm_count        = dedisp_get_dm_count(plan);
    max_delay       = dedisp_get_max_delay(plan);
    nsamps_computed = nsamps - max_delay;
    dmlist          = dedisp_get_dm_list(plan);
    // dt_factors = dedisp_get_dt_factors(plan);

    printf("----------------------------- DM COMPUTATIONS  "
           "----------------------------\n");
    printf("Computing %lu DMs from %f to %f pc/cm^3\n", dm_count, dmlist[0],
           dmlist[dm_count - 1]);
    printf("Max DM delay is %lu samples (%.f seconds)\n", max_delay,
           max_delay * dt);
    printf("Computing %lu out of %lu total samples (%.2f%% efficiency)\n",
           nsamps_computed, nsamps,
           100.0 * (dedisp_float)nsamps_computed / nsamps);
    printf("Output data array size : %lu MB\n",
           (dm_count * nsamps_computed * (32 / 8)) / (1 << 20));
    printf("\n");

    // Allocate space for the output data
    output = (dedisp_float*)std::malloc(nsamps_computed * dm_count * 32 / 8);
    if (output == NULL) {
        printf("\nERROR: Failed to allocate output array\n");
        return -1;
    }

    printf("Compute on GPU\n");
    startclock = clock();
    // Compute the dedispersion transform on the GPU
    error = dedisp_execute(plan, nsamps, input, in_nbits, output,
                           DEDISP_USE_DEFAULT);
    if (error != DEDISP_NO_ERROR) {
        printf("\nERROR: Failed to execute dedispersion plan: %s\n",
               dedisp_get_error_string(error));
        return -1;
    }
    printf("Dedispersion took %.2f seconds\n",
           (double)(clock() - startclock) / CLOCKS_PER_SEC);

    // Look for significant peaks
    dedisp_float out_mean, out_sigma;
    calc_stats_float(output, nsamps_computed * dm_count, &out_mean, &out_sigma);

    printf("Output RMS                               : %f\n", out_mean);
    printf("Output StdDev                            : %f\n", out_sigma);

    int i = 0;
    for (int nd = 0; nd < dm_count; nd++) {
        for (int ns = 0; ns < nsamps_computed; ns++) {
            dedisp_size idx  = nd * nsamps_computed + ns;
            dedisp_float val = output[idx];
            if (val - out_mean > 6.0 * out_sigma) {
                printf("DM trial %u (%.3f pc/cm^3), Samp %u (%.6f s): %f (%.2f "
                       "sigma)\n",
                       nd, dmlist[nd], ns, ns * dt, val,
                       (val - out_mean) / out_sigma);
                i++;
                if (i > 100)
                    break;
            }
        }
        if (i > 100)
            break;
    }

    // Clean up
    free(output);
    free(input);
    dedisp_destroy_plan(plan);
    printf("Dedispersion successful.\n");
    return 0;
}
