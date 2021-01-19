/*
  This file contains utils functions and physical equations.
*/

#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <dedisp/dedisp.hpp>

namespace dm_utils {

dedisp_float get_smearing(dedisp_float dt,
                          dedisp_float pulse_width,
                          dedisp_float f0,
                          dedisp_size nchans,
                          dedisp_float df,
                          dedisp_float DM,
                          dedisp_float deltaDM) {
    dedisp_float W         = pulse_width;
    dedisp_float BW        = nchans * std::abs(df);
    dedisp_float fc        = f0 - BW / 2;
    dedisp_float inv_fc3   = 1. / (fc * fc * fc);
    dedisp_float t_DM      = 8.3 * BW * DM * inv_fc3;
    dedisp_float t_deltaDM = 8.3 / 4 * BW * nchans * deltaDM * inv_fc3;
    dedisp_float t_smear
        = sqrt(dt * dt + W * W + t_DM * t_DM + t_deltaDM * t_deltaDM);
    return t_smear;
}

/**
 * @brief Generates a list of scrunch factors
 *
 * @param scrunch_list Output scrunch factor list
 * @param dt0
 * @param dm_list     List of dispersion measures
 * @param nchans      Number of frequency channels
 * @param f0          Centre frequency of first channel
 * @param df          Channel bandwidth
 * @param pulse_width Expected intrinsic width of the pulse in microseconds.
 * @param tol         The smearing tolerance factor between DM trials
 * (e.g, 1.25).
 */
void generate_scrunch_list(std::vector<dedisp_size>& scrunch_list,
                           dedisp_float dt0,
                           const std::vector<dedisp_float>& dm_list,
                           dedisp_size nchans,
                           dedisp_float f0,
                           dedisp_float df,
                           dedisp_float pulse_width,
                           dedisp_float tol) {
    // Note: This algorithm always starts with no scrunching and is only
    //         able to 'adapt' the scrunching by doubling in any step.
    // TODO: To improve this it would be nice to allow scrunch_list[0] > 1.
    //         This would probably require changing the output nsamps
    //           according to the mininum scrunch.

    scrunch_list.clear();
    scrunch_list.push_back(1);
    dedisp_float dm_count = dm_list.size();
    for (dedisp_size idm = 1; idm < dm_count; ++idm) {
        dedisp_size prev      = scrunch_list.back();
        dedisp_float dm       = dm_list[idm];
        dedisp_float delta_dm = dm - dm_list[idm - 1];

        dedisp_float smearing = get_smearing(prev * dt0, pulse_width * 1e-6, f0,
                                             nchans, df, dm, delta_dm);
        dedisp_float smearing2 = get_smearing(
            prev * 2 * dt0, pulse_width * 1e-6, f0, nchans, df, dm, delta_dm);
        if (smearing2 / smearing < tol) {
            scrunch_list.push_back(prev * 2);
        } else {
            scrunch_list.push_back(prev);
        }
    }
}

/**
 * @brief Generates a list of dispersion delays
 *
 * @param delay_table  Output delay list
 * @param nchans       Number of frequency channels
 * @param dt           Sampling time
 * @param f0           Centre frequency of first channel
 * @param df           Channel bandwidth
 */
void generate_delay_table(std::vector<dedisp_float>& delay_table,
                          dedisp_size nchans,
                          dedisp_float dt,
                          dedisp_float f0,
                          dedisp_float df) {
    delay_table.clear();
    for (dedisp_size ichan = 0; ichan < nchans; ++ichan) {
        dedisp_float a = 1.f / (f0 + ichan * df);
        dedisp_float b = 1.f / f0;
        // Note: To higher precision, the constant is 4.148741601e3
        dedisp_float delay = 4.15e3 / dt * (a * a - b * b);
        delay_table.push_back(delay);
    }
}

/**
 * @brief Generates a list of dispersion measures
 *
 * @param dm_list   Output dm list
 * @param dm_start  The lowest DM to use, in pc cm^-3.
 * @param dm_end   The highest DM to use, in pc cm^-3.
 * @param dt     Sampling time
 * @param ti     Expected intrinsic width of the pulse in microseconds.
 * @param f0     Centre frequency of first channel
 * @param df     Channel bandwidth
 * @param nchans Number of frequency channels
 * @param tol    The smearing tolerance factor between DM trials (e.g, 1.25).
 */
void generate_dm_list(std::vector<dedisp_float>& dm_list,
                      dedisp_float dm_start,
                      dedisp_float dm_end,
                      double dt,
                      double ti,
                      double f0,
                      double df,
                      dedisp_size nchans,
                      double tol) {
    // Note: This algorithm originates from Lina Levin
    // Note: Computation done in double precision to match MB's code

    dt *= 1e6;
    double f    = (f0 + ((nchans / 2) - 0.5) * df) * 1e-3;
    double tol2 = tol * tol;
    double a    = 8.3 * df / (f * f * f);
    double a2   = a * a;
    double b2   = a2 * (double)(nchans * nchans / 16.0);
    double c    = (dt * dt + ti * ti) * (tol2 - 1.0);

    dm_list.clear();
    dm_list.push_back(dm_start);
    while (dm_list.back() < dm_end) {
        double prev  = dm_list.back();
        double prev2 = prev * prev;
        double k     = c + tol2 * a2 * prev2;
        double dm    = ((b2 * prev + sqrt(-a2 * b2 * prev2 + (a2 + b2) * k))
                     / (a2 + b2));
        dm_list.push_back(dm);
    }
}

}  // namespace dm_utils
