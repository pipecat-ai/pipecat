#!/bin/bash

PID=$1

while true; do
  # Clear the screen
  clear
  # Print the header + RSS in GB
  ps -p "$PID" -o pid,comm,rss | \
  awk 'NR==1 {print $0, "rss_GB"} NR>1 {printf "%s %s %s %.2f\n", $1,$2,$3,$3/1024/1024}'
  sleep 1
done