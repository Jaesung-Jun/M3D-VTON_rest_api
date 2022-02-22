#!bin/bash

# $1 : Video(image) path
# $2 : JSON output path

cd openpose
./build/examples/openpose/openpose.bin --video $1 --hand --face --write_json $2 --display 0 --render_pose 0;
