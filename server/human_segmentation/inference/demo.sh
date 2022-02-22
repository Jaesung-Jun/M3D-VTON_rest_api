CUDA_VISIBLE_DEVICES=0 \
python inference_single.py \
--loadmodel '../pretrained/deeplabv3plus-xception-vocNov14_20-51-38_epoch-89.pth' \
--img_list /home/lab/sda/jsjun/Image_Processing/3D-Try-On/m3d-vton/server/2D-Human-Parsing/demo_imgs/img_list.txt \
--output_dir /home/lab/sda/jsjun/Image_Processing/3D-Try-On/m3d-vton/server/2D-Human-Parsing/parsing_result \
--data_root /home/lab/sda/jsjun/Image_Processing/3D-Try-On/m3d-vton/server/2D-Human-Parsing/demo_imgs/
