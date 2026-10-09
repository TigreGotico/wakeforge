// WakeHuBERT-tiny on the ESP32-S3: the streamed int8 trunk and a GRU head over fixed audio from flash.
//
// Every 80 ms block of the test audio goes through the C log-mel, the int8 trunk (state carried between calls)
// and the head over the last 75 trunk frames. Each block prints its times, the FNV-1a hash of its int8 feature
// bytes (comparable with expected_trace.json from export_tflm.py audio) and the head's logit. The plugin's rule for
// a ready model decides: sigmoid(calib_a * logit + calib_b) at or above the head's default threshold fires, the stream
// is reset, and the next 2.0 s are not scored. The audio runs
// twice: models read from flash, then models copied into PSRAM.
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "logmel.h"
#include "model_data.h"
#include "model_io.h"
#include "test_audio.h"

#define BLOCK 1280
#define MEL_VALUES (8 * 64)
#define QUIET_BLOCKS 25
#define TRUNK_ARENA (128 * 1024)
#define HEAD_ARENA (576 * 1024)

static float g_wav[BLOCK];
static float g_mel[MEL_VALUES];
static float g_fe_state[240];
static int8_t g_ring[MWH_WINDOW * MWH_FEATURE_DIM];
static int8_t *g_stage;

static uint32_t fnv1a(const int8_t *p, int n) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < n; i++) h = (h ^ (uint8_t)p[i]) * 16777619u;
    return h;
}

static inline int8_t quant(float x, float scale, int zp) {
    int v = (int)lrintf(x / scale) + zp;
    return (int8_t)(v < -128 ? -128 : v > 127 ? 127 : v);
}

static void requant(const int8_t *src, float ss, int sz, int8_t *dst, float ds, int dz, int n) {
    if (ss == ds && sz == dz) {
        memcpy(dst, src, n);
        return;
    }
    for (int i = 0; i < n; i++) dst[i] = quant((src[i] - sz) * ss, ds, dz);
}

static void *alloc_arena(size_t bytes, const char **where) {
    void *p = heap_caps_aligned_alloc(16, bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    *where = "internal";
    if (!p) {
        p = heap_caps_aligned_alloc(16, bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        *where = "psram";
    }
    return p;
}

typedef tflite::MicroMutableOpResolver<16> Resolver;

static void reset_stream(tflite::MicroInterpreter &trunk) {
    for (int s = 0; s < MWH_N_STATES; s++)
        memset(trunk.input(MWH_STATES[s].in)->data.int8, MWH_STATES[s].in_zp, MWH_STATES[s].bytes);
    memset(g_ring, quant(0.f, MWH_HEAD_IN_SCALE, MWH_HEAD_IN_ZP), sizeof(g_ring));
    memset(g_fe_state, 0, sizeof(g_fe_state));
}

static void add_ops(Resolver &r) {
    r.AddConcatenation();
    r.AddConv2D();
    r.AddDepthwiseConv2D();
    r.AddAdd();
    r.AddSub();
    r.AddMul();
    r.AddStridedSlice();
    r.AddQuantize();
    r.AddDequantize();
    r.AddReshape();
    r.AddFullyConnected();
    r.AddLogistic();
    r.AddTanh();
    r.AddRelu();
    r.AddMean();
    r.AddBatchMatMul();
}

static void run_pass(const char *label, const uint8_t *trunk_model, const uint8_t *head_model, Resolver &resolver) {
    const char *trunk_where, *head_where;
    uint8_t *trunk_arena = (uint8_t *)alloc_arena(TRUNK_ARENA, &trunk_where);
    uint8_t *head_arena = (uint8_t *)alloc_arena(HEAD_ARENA, &head_where);
    tflite::MicroInterpreter trunk(tflite::GetModel(trunk_model), resolver, trunk_arena, TRUNK_ARENA);
    tflite::MicroInterpreter head(tflite::GetModel(head_model), resolver, head_arena, HEAD_ARENA);
    if (trunk.AllocateTensors() != kTfLiteOk || head.AllocateTensors() != kTfLiteOk) {
        printf("MWH %s allocate failed\n", label);
        return;
    }
    printf("MWH %s arena trunk %u B (%s) head %u B (%s); free internal %u psram %u\n", label,
           (unsigned)trunk.arena_used_bytes(), trunk_where, (unsigned)head.arena_used_bytes(), head_where,
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    reset_stream(trunk);

    const int16_t *pcm = (const int16_t *)g_test_audio;
    const int blocks = MWH_TEST_AUDIO_SAMPLES / BLOCK;
    const int frame_bytes = MWH_FRAMES * MWH_FEATURE_DIM;
    int64_t sum_fe = 0, sum_trunk = 0, sum_head = 0, max_total = 0;
    int quiet = 0;
    for (int b = 0; b < blocks; b++) {
        for (int i = 0; i < BLOCK; i++) g_wav[i] = pcm[b * BLOCK + i] / 32768.0f;
        int64_t t0 = esp_timer_get_time();
        mwh_logmel_block(g_fe_state, g_wav, g_mel);
        int8_t *mel_in = trunk.input(MWH_MEL_IN)->data.int8;
        for (int i = 0; i < MEL_VALUES; i++) mel_in[i] = quant(g_mel[i], MWH_MEL_SCALE, MWH_MEL_ZP);
        int64_t t1 = esp_timer_get_time();
        if (trunk.Invoke() != kTfLiteOk) {
            printf("MWH trunk invoke failed\n");
            return;
        }
        const int8_t *feats = trunk.output(MWH_FEATURES_OUT)->data.int8;
        uint32_t h = fnv1a(feats, frame_bytes);
        memmove(g_ring, g_ring + frame_bytes, sizeof(g_ring) - frame_bytes);
        requant(feats, MWH_FEAT_SCALE, MWH_FEAT_ZP, g_ring + sizeof(g_ring) - frame_bytes, MWH_HEAD_IN_SCALE,
                MWH_HEAD_IN_ZP, frame_bytes);
        int off = 0;
        for (int s = 0; s < MWH_N_STATES; s++) {
            memcpy(g_stage + off, trunk.output(MWH_STATES[s].out)->data.int8, MWH_STATES[s].bytes);
            off += MWH_STATES[s].bytes;
        }
        off = 0;
        for (int s = 0; s < MWH_N_STATES; s++) {
            const mwh_state_io_t &st = MWH_STATES[s];
            requant(g_stage + off, st.out_scale, st.out_zp, trunk.input(st.in)->data.int8, st.in_scale, st.in_zp,
                    st.bytes);
            off += st.bytes;
        }
        int64_t t2 = esp_timer_get_time();
        memcpy(head.input(MWH_HEAD_IN)->data.int8, g_ring, sizeof(g_ring));
        if (head.Invoke() != kTfLiteOk) {
            printf("MWH head invoke failed\n");
            return;
        }
        float logit = (head.output(MWH_HEAD_OUT)->data.int8[0] - MWH_LOGIT_ZP) * MWH_LOGIT_SCALE;
        int64_t t3 = esp_timer_get_time();
        sum_fe += t1 - t0;
        sum_trunk += t2 - t1;
        sum_head += t3 - t2;
        if (t3 - t0 > max_total) max_total = t3 - t0;
        printf("MWH %s blk %d fe_us %d trunk_us %d head_us %d fnv %08lx logit %.3f\n", label, b, (int)(t1 - t0),
               (int)(t2 - t1), (int)(t3 - t2), (unsigned long)h, logit);
        float p = 1.f / (1.f + expf(-(MWH_CALIB_A * logit + MWH_CALIB_B)));
        if (quiet > 0) {
            quiet--;
        } else if (p >= MWH_THRESHOLD) {
            printf("MWH %s WAKE at %.2f s (block %d, p %.3f)\n", label, (b + 1) * 0.08f, b, p);
            reset_stream(trunk);
            quiet = QUIET_BLOCKS;
        }
        if (b % 8 == 7) vTaskDelay(1);
    }
    printf("MWH %s summary blocks %d mean_us fe %lld trunk %lld head %lld total %lld max_total_us %lld "
           "budget_us 80000 min_free_internal %u min_free_psram %u\n",
           label, blocks, sum_fe / blocks, sum_trunk / blocks, sum_head / blocks,
           (sum_fe + sum_trunk + sum_head) / blocks, max_total,
           (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));
    heap_caps_free(trunk_arena);
    heap_caps_free(head_arena);
}

extern "C" void app_main(void) {
    static Resolver resolver;
    add_ops(resolver);
    mwh_logmel_init();
    int stage_bytes = 0;
    for (int s = 0; s < MWH_N_STATES; s++) stage_bytes += MWH_STATES[s].bytes;
    g_stage = (int8_t *)heap_caps_malloc(stage_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    printf("MWH start: trunk model %u B, head model %u B, state %d B, audio %d samples\n", g_trunk_model_len,
           g_head_model_len, stage_bytes, MWH_TEST_AUDIO_SAMPLES);
    run_pass("flash", g_trunk_model, g_head_model, resolver);
    uint8_t *t = (uint8_t *)heap_caps_aligned_alloc(16, g_trunk_model_len, MALLOC_CAP_SPIRAM);
    uint8_t *h = (uint8_t *)heap_caps_aligned_alloc(16, g_head_model_len, MALLOC_CAP_SPIRAM);
    if (t && h) {
        memcpy(t, g_trunk_model, g_trunk_model_len);
        memcpy(h, g_head_model, g_head_model_len);
        run_pass("psram", t, h, resolver);
    }
    printf("MWH done\n");
    while (true) vTaskDelay(portMAX_DELAY);
}
