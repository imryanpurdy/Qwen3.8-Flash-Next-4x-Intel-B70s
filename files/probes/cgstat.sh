#!/bin/sh
# cgroup memory.stat summary (GiB) — safe quoting, no nested awk escaping
C=/sys/fs/cgroup
cur=$(cat $C/memory.current)
echo "current_GiB=$((cur/1048576)).$(((cur%1048576)/104858))"
for f in anon file kernel slab sock pagetables shmem file_mapped file_writeback; do
  v=$(grep -E "^$f " $C/memory.stat | head -1 | cut -d' ' -f2)
  [ -n "$v" ] && echo "$f=$((v/1048576))GiB"
done
echo "peak_GiB=$(($(cat $C/memory.peak)/1048576))GiB"
