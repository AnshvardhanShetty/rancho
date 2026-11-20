// Count later reads for the first reuse predictor
// Record which request introduced each block and count later reads in that session
// Y.i32 stores reads by interval and c.i32 stores distinct block counts by request
// The hourly trainer only uses intervals that have already ended

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  // 1. Read paths and interval settings, then load lengths, sessions, and times.
  const char *pfx = argv[1];
  int64_t NB = atoll(argv[2]) + 1;
  int64_t BUCK = atoll(argv[3]);
  int64_t NBK = atoll(argv[4]);
  const char *OUT = argv[5];
  char p[512];
  FILE *f;
  snprintf(p, sizeof p, "%s_nb.i32", pfx);
  f = fopen(p, "rb");
  fseek(f, 0, SEEK_END);
  int64_t n = ftell(f) / 4;
  fseek(f, 0, SEEK_SET);
  int32_t *nb = malloc(n * 4);
  if (fread(nb, 4, n, f) != (size_t)n)
    return 2;
  fclose(f);
  snprintf(p, sizeof p, "%s_sess.i32", pfx);
  f = fopen(p, "rb");
  int32_t *ss = malloc(n * 4);
  if (fread(ss, 4, n, f) != (size_t)n)
    return 2;
  fclose(f);
  snprintf(p, sizeof p, "%s_ts.i64", pfx);
  f = fopen(p, "rb");
  int64_t *ts = malloc(n * 8);
  if (fread(ts, 8, n, f) != (size_t)n)
    return 2;
  fclose(f);
  // 2. Track the last session, introducing request, and per-request duplicate marks.
  int32_t *stamp = malloc(NB * 4), *ownr = malloc(NB * 4), *mark = malloc(NB * 4);
  for (int64_t i = 0; i < NB; i++) {
    stamp[i] = -1;
    ownr[i] = -1;
    mark[i] = -1;
  }
  int32_t *Y = calloc((size_t)NBK * n, 4);
  if (!Y) {
    fprintf(stderr, "alloc %lld\n", (long long)((int64_t)NBK * n * 4));
    return 3;
  }
  int32_t *buf = malloc(4 * (1 << 21));
  int64_t bcap = 1 << 21;
  snprintf(p, sizeof p, "%s_ids.i32", pfx);
  FILE *fi = fopen(p, "rb");
  int64_t tot = 0;
  int32_t *offers = calloc(n, 4);
  // 3. Read requests in order; count each block once and assign reuse to its owner.
  for (int64_t r = 0; r < n; r++) {
    int32_t k = nb[r];
    if (k > bcap) {
      bcap = (int64_t)k * 2;
      buf = realloc(buf, bcap * 4);
    }
    if (k > 0 && fread(buf, 4, k, fi) != (size_t)k)
      return 4;
    int32_t sid = ss[r];
    int64_t bk = ts[r] / BUCK;
    if (bk < 0)
      bk = 0;
    if (bk >= NBK)
      bk = NBK - 1;
    for (int32_t j = 0; j < k; j++) {
      int32_t b = buf[j];
      if (b < 0 || b >= NB)
        continue;
      if (mark[b] == (int32_t)r)
        continue;
      mark[b] = (int32_t)r;
      if (stamp[b] != sid) {
        stamp[b] = sid;
        ownr[b] = (int32_t)r;
        offers[r]++;
      } else if (ownr[b] >= 0) {
        Y[bk * n + ownr[b]]++;
        tot++;
      }
    }
  }
  fclose(fi);
  // 4. Save read counts by interval and each request's introduced-block count.
  f = fopen(OUT, "wb");
  fwrite(Y, 4, (size_t)NBK * n, f);
  fclose(f);
  printf("{\"requests\":%lld,\"buckets\":%lld,\"credits_bucketed\":%lld}\n", (long long)n,
         (long long)NBK, (long long)tot);
  if (argc > 6) {
    f = fopen(argv[6], "wb");
    fwrite(offers, 4, n, f);
    fclose(f);
  }
  return 0;
}
