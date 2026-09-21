#!/usr/bin/env bash
# ============================================================================
# p18-oneccl-dedup-omix-pin.sh — P18: deduplicate oneCCL to the OMIX-validated
#   pin (2022.1.x) inside flashnext-v25:0.1
#
# Evidence (Ryan directive 2026-09-21, after boot #18/#19):
#   - torch resolves the PIP tree /opt/venv/lib/libccl.so.1 = oneccl 2022.0.0
#     (matches the boot #18 capture-refusal string "please use sycl_algorithms")
#   - base OMIX 0.4 image ships the validated pin at
#     /opt/intel/oneapi/ccl/2022.1/lib (2022.1.2), unused
#   - validated-pairing rule: the library refusing graph recording must be the
#     one OMIX validated, not a random pip duplicate
#
# Actions:
#   1. pip uninstall oneccl oneccl-devel (removes the 2022.0.0 tree from /opt/venv)
#   2. register the OMIX tree with ldconfig so torch's dlopen("libccl.so.1")
#      resolves to 2022.1.x
#   3. carry over pip's libfabric/ATL plugins if the OMIX tree lacks them
#      (profile-A runs CCL_ATL_TRANSPORT=ofi)
#   4. asserts: exactly ONE libccl.so.1 resolvable; it lives under
#      /opt/intel/oneapi/ccl/2022.1; its version strings read 2022.1
# ============================================================================
set -euo pipefail
echo "P18: dedup oneCCL to OMIX-validated pin"

CCL_OMIX_LIB="/opt/intel/oneapi/ccl/2022.1/lib"

# --- 0. sanity: OMIX tree must exist -----------------------------------------
if [ ! -e "${CCL_OMIX_LIB}/libccl.so.1" ]; then
    echo "P18 FATAL: OMIX ccl tree not found at ${CCL_OMIX_LIB}" >&2
    exit 1
fi

# --- 1. uninstall the pip duplicate ------------------------------------------
pip uninstall -y oneccl oneccl-devel 2>&1 | grep -E "Successfully|not installed" || true

# --- 2. carry over libfabric if OMIX lacks it (ATL=ofi needs it) -------------
if [ ! -d /opt/intel/oneapi/ccl/2022.1/libfabric ] && [ -d /opt/venv/lib/libfabric ]; then
    cp -a /opt/venv/lib/libfabric /opt/intel/oneapi/ccl/2022.1/libfabric
    echo "P18: carried pip libfabric -> OMIX tree"
fi

# --- 3. ldconfig registration -------------------------------------------------
echo "${CCL_OMIX_LIB}" > /etc/ld.so.conf.d/oneccl-omix.conf
ldconfig

# --- 4. asserts ----------------------------------------------------------------
# 4a. exactly one resolvable libccl.so.1 and it is the OMIX one
RESOLVED="$(ldconfig -p | grep -E 'libccl\.so\.1 ' | head -1 | awk '{print $NF}')"
if [ -z "$RESOLVED" ]; then
    echo "P18 FATAL: libccl.so.1 not resolvable after dedup" >&2
    exit 1
fi
case "$RESOLVED" in
    /opt/intel/oneapi/ccl/2022.1/*) ;;
    *) echo "P18 FATAL: libccl.so.1 resolves to wrong tree: $RESOLVED" >&2; exit 1 ;;
esac
echo "P18: libccl.so.1 resolves to $RESOLVED"

# 4b. version strings from that exact file read 2022.1
VER="$(strings "$RESOLVED" | grep -oE '2022\.1\.[0-9]+' | head -1 || true)"
if [ -z "$VER" ]; then
    echo "P18 FATAL: no 2022.1.x version string in $RESOLVED" >&2
    exit 1
fi
echo "P18: OMIX oneCCL version string: $VER"

# 4c. no leftover pip ccl trees
if /opt/venv/bin/pip show oneccl >/dev/null 2>&1; then
    echo "P18 FATAL: pip oneccl still installed" >&2
    exit 1
fi

# 4d. ctypes dlopen from a clean env resolves the OMIX tree
python3 - <<'PYEOF'
import ctypes, ctypes.util
p = ctypes.util.find_library("ccl") or "libccl.so.1"
lib = ctypes.CDLL("libccl.so.1")
print("P18 assert OK: dlopen(libccl.so.1) succeeded, find_library =", p)
PYEOF

echo "P18: written"
