#pragma once
#ifdef __cplusplus
extern "C" {
#endif

void mwh_logmel_init(void);
/* One 80 ms block: state holds the previous 240 samples and is updated; wav is 1280 samples in [-1, 1];
   out receives 8 frames x 64 normalised log-mel values, frame-major. */
void mwh_logmel_block(float *state, const float *wav, float *out);

#ifdef __cplusplus
}
#endif
