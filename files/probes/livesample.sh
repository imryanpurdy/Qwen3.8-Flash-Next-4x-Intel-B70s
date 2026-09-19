#!/bin/sh
# sample live cgroup + gtt + meminfo once, single line
C=/sys/fs/cgroup
cur=$(cat $C/memory.current)
anon=$(grep -E "^anon " $C/memory.stat | cut -d' ' -f2)
file=$(grep -E "^file " $C/memory.stat | cut -d' ' -f2)
slab=$(grep -E "^slab " $C/memory.stat | cut -d' ' -f2)
kern=$(grep -E "^kernel " $C/memory.stat | cut -d' ' -f2)
pt=$(grep -E "^pagetables " $C/memory.stat | cut -d' ' -f2)
g1=$(grep -a usage /sys/kernel/debug/dri/1/gtt_mm 2>/dev/null | tail -1 | grep -aoE '[0-9]+')
g2=$(grep -a usage /sys/kernel/debug/dri/2/gtt_mm 2>/dev/null | tail -1 | grep -aoE '[0-9]+')
g3=$(grep -a usage /sys/kernel/debug/dri/3/gtt_mm 2>/dev/null | tail -1 | grep -aoE '[0-9]+')
g4=$(grep -a usage /sys/kernel/debug/dri/4/gtt_mm 2>/dev/null | tail -1 | grep -aoE '[0-9]+')
mi=$(grep -E '^(Slab|AnonPages):' /proc/meminfo | tr '\n' ' ' | tr -s ' ')
echo "$(date +%H:%M:%S) cur=$((cur/1048576))M anon=$((anon/1048576))M file=$((file/1048576))M slab=$((slab/1048576))M kern=$((kern/1048576))M pt=$((pt/1048576))M gtt=$((g1/1048576))+$((g2/1048576))+$((g3/1048576))+$((g4/1048576))G | $mi"
