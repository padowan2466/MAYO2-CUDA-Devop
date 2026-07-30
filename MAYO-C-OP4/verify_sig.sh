#!/bin/bash
# Bit-exact signature verification harness.
# Reference signature captured from the serial/reference MAYO-2 pipeline
# (OP3 baseline, csk=zeros pattern from inputs.cu, msg={0xe,0,0,...}, mlen=32).
set -euo pipefail

BIN="${1:?usage: verify_sig.sh <path-to-mayo-2-gpu-fine>}"

REF="9e, 62, 97, 3f, 23, 74, 11, 32, ae, df, 4a, a5, fb, 45, c6, 02, d5, 0d, e6, 5d, f7, 94, 0f, 7c, c0, 2a, 84, 74, 62, a6, c4, bf, aa, ea, 4d, a4, 9b, dc, d1, 2f, 24, 98, ff, 66, ff, 9d, 99, a7, f2, 36, 73, 44, f5, c4, fa, ef, 2b, 55, 44, 55, 84, 41, 0c, 65, fa, 91, fe, c8, 05, c5, 1a, 55, 88, 56, 11, 2d, a8, 35, a3, a5, 44, 86, 82, 4f, 5b, 41, e8, 9e, 4c, 34, d8, ec, ef, fa, 83, 9e, 17, 4e, 3d, c3, 86, dc, 42, b3, 46, 3f, 21, bd, 92, 5c, a9, f6, ad, b1, a1, fd, 62, df, 65, a9, 08, d3, f0, 1c, 32, 5f, b3, cd, ed, df, 68, b9, 04, 86, d7, b4, bf, 0b, ca, a2, ba, 4a, 90, 75, 14, 95, 7c, 8d, 88, 47, ff, 44, bc, cb, c4, 68, e9, 3b, a5, 2e, 56, 1a, b5, 68, 2f, 9e, c2, 8a, 64, 3c, c7, af, 24, 65, 16, 2e, a4, 1e, 78, 75, 9f, d7, 52, 92, 01, f3,"

GOT=$("$BIN" 2>&1 | tr -d '\r' | grep -A1 "^Signature:$" | tail -1 | sed -e 's/[[:space:]]*$//')
REF=$(echo "$REF" | sed -e 's/[[:space:]]*$//')

if [ "$GOT" == "$REF" ]; then
    echo "PASS: signature bit-exact match"
    exit 0
else
    echo "FAIL: signature mismatch"
    echo "expected: $REF"
    echo "got:      $GOT"
    exit 1
fi
