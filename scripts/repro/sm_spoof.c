// 1028 換機重現實驗：讓程序看到的 SM（運算核心群）數＝SPOOF_SM_COUNT，用來模擬 5090（170 個）的算法選擇。
// 只攔截「查詢」函式，不改任何計算；未設 SPOOF_SM_COUNT 時完全透明。
// 編譯（*.so 不進 git，clone 後要自己編）：
//   SITE=venv_matcha/lib/python3.12/site-packages
//   gcc -shared -fPIC -O2 -I$SITE/nvidia/cuda_runtime/include -I$SITE/triton/backends/nvidia/include \
//       scripts/repro/sm_spoof.c -o scripts/repro/sm_spoof2.so -ldl
// 用法：LD_PRELOAD=scripts/repro/sm_spoof2.so SPOOF_SM_COUNT=170（start_v1_baseline.sh 經 scripts/repro/hw_guard.sh 自動處理）。
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#undef cudaGetDeviceProperties
static int want(void) { const char *s = getenv("SPOOF_SM_COUNT"); return s ? atoi(s) : 0; }
cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp *p, int d) {
    static cudaError_t (*real)(struct cudaDeviceProp *, int) = 0;
    if (!real) real = dlsym(RTLD_NEXT, "cudaGetDeviceProperties_v2");
    cudaError_t r = real(p, d); int n = want();
    if (r == cudaSuccess && n > 0) p->multiProcessorCount = n;
    return r;
}
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *p, int d) {
    static cudaError_t (*real)(struct cudaDeviceProp *, int) = 0;
    if (!real) real = dlsym(RTLD_NEXT, "cudaGetDeviceProperties");
    cudaError_t r = real(p, d); int n = want();
    if (r == cudaSuccess && n > 0) p->multiProcessorCount = n;
    return r;
}
cudaError_t cudaDeviceGetAttribute(int *v, enum cudaDeviceAttr a, int d) {
    static cudaError_t (*real)(int *, enum cudaDeviceAttr, int) = 0;
    if (!real) real = dlsym(RTLD_NEXT, "cudaDeviceGetAttribute");
    cudaError_t r = real(v, a, d); int n = want();
    if (r == cudaSuccess && n > 0 && a == cudaDevAttrMultiProcessorCount) *v = n;
    return r;
}

// ── 第二層（1028）：cuBLAS 靜態連結 cudart、用 dlopen("libcuda.so.1")＋dlsym 取驅動函式，上面的攔截碰不到。
//    這裡攔 dlsym：取到 cuGetProcAddress(_v2)／cuDeviceGetAttribute 時換成包裝版，
//    包裝版只在查詢「SM 數」（CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT=16）時回報 SPOOF_SM_COUNT。
#include <string.h>
#include <stdio.h>
typedef int CUresult_t;
typedef unsigned long long cuu64_t;
static void *(*real_dlsym)(void *, const char *) = 0;
static CUresult_t (*real_devattr)(int *, int, int) = 0;
static CUresult_t (*real_getproc1)(const char *, void **, int, cuu64_t) = 0;
static CUresult_t (*real_getproc2)(const char *, void **, int, cuu64_t, void *) = 0;
static CUresult_t my_devattr(int *v, int a, int d) {
    CUresult_t r = real_devattr(v, a, d); int n = want();
    if (getenv("SPOOF_DEBUG") && a == 16) fprintf(stderr, "[spoof] devattr SM queried\n");
    if (r == 0 && a == 16 && n > 0) *v = n;
    return r;
}
static CUresult_t my_getproc1(const char *s, void **pfn, int ver, cuu64_t fl);
static CUresult_t my_getproc2(const char *s, void **pfn, int ver, cuu64_t fl, void *st);
static void *wrap_sym(const char *name, void *p) {
    if (!p || !name) return p;
    if (strcmp(name, "cuDeviceGetAttribute") == 0) { real_devattr = p; return (void *)my_devattr; }
    if (strcmp(name, "cuGetProcAddress_v2") == 0) { real_getproc2 = p; return (void *)my_getproc2; }
    if (strcmp(name, "cuGetProcAddress") == 0) { real_getproc1 = p; return (void *)my_getproc1; }
    return p;
}
static CUresult_t my_getproc1(const char *s, void **pfn, int ver, cuu64_t fl) {
    CUresult_t r = real_getproc1(s, pfn, ver, fl);
    if (getenv("SPOOF_DEBUG") && s) fprintf(stderr, "[spoof] getproc1 %s\n", s);
    if (r == 0 && pfn) *pfn = wrap_sym(s, *pfn);
    return r;
}
static CUresult_t my_getproc2(const char *s, void **pfn, int ver, cuu64_t fl, void *st) {
    CUresult_t r = real_getproc2(s, pfn, ver, fl, st);
    if (getenv("SPOOF_DEBUG") && s) fprintf(stderr, "[spoof] getproc2 %s\n", s);
    if (r == 0 && pfn) *pfn = wrap_sym(s, *pfn);
    return r;
}
static int dbg(void) { const char *s = getenv("SPOOF_DEBUG"); return s && *s == '1'; }
void *dlsym(void *h, const char *name) {
    if (!real_dlsym) real_dlsym = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
    void *p = real_dlsym(h, name);
    if (dbg() && name && strncmp(name, "cu", 2) == 0) fprintf(stderr, "[spoof] dlsym %s\n", name);
    if (!p || !name || want() <= 0) return p;
    return wrap_sym(name, p);
}
