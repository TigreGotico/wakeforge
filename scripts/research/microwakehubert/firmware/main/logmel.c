/* The trunk's causal log-mel in float32: 400-sample periodic Hann window, 160-sample hop, power spectrum by a
   mixed-radix FFT (400 = 2^4 * 5^2), 64 mel bands, log(x + eps), and the trunk's input batch norm. */
#include <math.h>
#include <string.h>

#include "logmel.h"
#include "logmel_consts.h"

typedef struct { float re, im; } cpx;

static cpx twiddle[MWH_N_FFT];
static const int factors[] = {5, 5, 2, 2, 2, 2, 1};

void mwh_logmel_init(void) {
    for (int k = 0; k < MWH_N_FFT; k++) {
        double a = -2.0 * M_PI * k / MWH_N_FFT;
        twiddle[k].re = (float)cos(a);
        twiddle[k].im = (float)sin(a);
    }
}

static void fft(cpx *out, const cpx *in, int n, int stride, const int *f) {
    int p = f[0], m = n / p;
    if (m == 1) {
        for (int q = 0; q < p; q++) out[q] = in[q * stride];
    } else {
        for (int q = 0; q < p; q++) fft(out + q * m, in + q * stride, m, stride * p, f + 1);
    }
    int tw = MWH_N_FFT / n;
    cpx y[5];
    for (int k = 0; k < m; k++) {
        for (int q = 0; q < p; q++) {
            cpx v = out[q * m + k], w = twiddle[(q * k * tw) % MWH_N_FFT];
            y[q].re = v.re * w.re - v.im * w.im;
            y[q].im = v.re * w.im + v.im * w.re;
        }
        for (int s = 0; s < p; s++) {
            cpx acc = {0.f, 0.f};
            for (int q = 0; q < p; q++) {
                cpx w = twiddle[(q * s * (MWH_N_FFT / p)) % MWH_N_FFT];
                acc.re += y[q].re * w.re - y[q].im * w.im;
                acc.im += y[q].re * w.im + y[q].im * w.re;
            }
            out[k + s * m] = acc;
        }
    }
}

void mwh_logmel_block(float *state, const float *wav, float *out) {
    static float buf[MWH_CTX + MWH_BLOCK];
    static cpx in[MWH_N_FFT], spec[MWH_N_FFT];
    static float power[MWH_N_BINS];
    memcpy(buf, state, sizeof(float) * MWH_CTX);
    memcpy(buf + MWH_CTX, wav, sizeof(float) * MWH_BLOCK);
    for (int t = 0; t < MWH_MEL_FRAMES; t++) {
        const float *x = buf + t * MWH_MEL_HOP;
        for (int i = 0; i < MWH_N_FFT; i++) {
            in[i].re = x[i] * MWH_WINDOW_FN[i];
            in[i].im = 0.f;
        }
        fft(spec, in, MWH_N_FFT, 1, factors);
        for (int b = 0; b < MWH_N_BINS; b++) power[b] = spec[b].re * spec[b].re + spec[b].im * spec[b].im;
        const float *w = MWH_FB;
        for (int m = 0; m < MWH_N_MELS; m++) {
            float acc = 0.f;
            for (int b = MWH_FB_LO[m]; b < MWH_FB_HI[m]; b++) acc += *w++ * power[b];
            out[t * MWH_N_MELS + m] = logf(acc + MWH_LOG_EPS) * MWH_NORM_SCALE[m] + MWH_NORM_SHIFT[m];
        }
    }
    memcpy(state, buf + MWH_BLOCK, sizeof(float) * MWH_CTX);
}
