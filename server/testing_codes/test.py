from cloth_mask import Preprocessing as cloth_preprocess
from osAdvanced import File_Control
import human_segmentation as hs
from PIL import Image
import numpy as np
import base64
import os
from distutils.dir_util import copy_tree
from shutil import copytree, copy
from util import data_preprocessing
import model_inference
import rgbd2pcd
import time

class ERROR:
    # File Control Errers.
    ERR00 = "00 : Error occured while Creating User Directory on Server."   # 디렉토리 만드는 동안 에러
    ERR01 = "01 : Error occured while Copy Sample Images to user directory." # 이미지 복사하는 동안 에러
    ERR02 = "02 : User Directory Not Exists. (Please request to [wtf/user_directory])"   # 유저 디렉토리가 존재하지 않음

    ########################################################################################
    # sample list loading api error
    ERR10 = "10 : Error occured while return sample image list (USER DIRECTORY NOT EXIST)" # 유저 디렉토리가 없을 때

    ########################################################################################
    ERR20 = "20 : Error occured while Openpose is excuting" # openpose 실행 중 에러


class User_Settings:
    @staticmethod
    def make_dir(uid):                
        try:
            os.mkdir("../user_requests/{}/".format(uid))
            os.mkdir("../user_requests/{}/{}".format(uid, "cloth"))
            os.mkdir("../user_requests/{}/{}".format(uid, "image"))
            os.mkdir("../user_requests/{}/{}".format(uid, "cloth-mask"))
            os.mkdir("../user_requests/{}/{}".format(uid, "image-parse"))
            os.mkdir("../user_requests/{}/{}".format(uid, "pose"))
            return True
        except:
            return ERROR.ERR00
    
    @staticmethod
    def copy_samples(uid):              # <- change to Link Files
        try:
            copy_tree("samples/cloth/", "../user_requests/{}/cloth".format(uid))
            copy_tree("samples/cloth-mask/", "../user_requests/{}/cloth-mask".format(uid))
            copy_tree("samples/image/", "../user_requests/{}/image".format(uid))
            copy_tree("samples/image-parse/", "../user_requests/{}/image-parse".format(uid))
            copy_tree("samples/pose/", "../user_requests/{}/pose".format(uid))
            copy_tree("samples/depth/", "../user_requests/{}/depth".format(uid))

            with open("../user_requests/{}/train_pairs.txt".format(uid), "w") as f:
                f.write("custom_model_whole_front.png    custom_cloth_front.jpg")
                f.close()

            with open("../user_requests/{}/test_pairs.txt".format(uid), "w") as f:
                f.write("custom_model_whole_front.png    custom_cloth_front.jpg")
                f.close()

            return True
        except:
            return ERROR.ERR01
    @staticmethod
    def copy_train_pairs(uid):
        try:
            copy_tree("../user_requests/{}/cloth".format(uid), "../user_requests/{}/aligned/test_pairs/cloth".format(uid))
            copy_tree("../user_requests/{}/cloth-mask".format(uid), "../user_requests/{}/aligned/test_pairs/cloth-mask".format(uid))
        except:
            return ERROR.ERR01

class OpenPose:
    @staticmethod
    def poses_3d(uid, image_path, json_path):
        try:
            if "successfully finished" in os.popen("bash ./openpose/openpose.sh {0} {1}".format("../" + image_path, "../" + json_path)).read():
                print("Openpose pose estimation successfully finished.")
                os.rename("../user_requests/{}/pose/custom_model_whole_front_000000000000_keypoints.json".format(uid), "../user_requests/{}/pose/custom_model_whole_front_keypoints.json".format(uid))
        except:
            return ERROR.ERR20

class Image_Preprocessing:
    @staticmethod
    def generate_path(uid, image_type, image_name):
        return "../user_requests/{}/{}/{}".format(uid, image_type, image_name)

    @staticmethod
    def save_decoded_image(encoded_image, image_path):
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(encoded_image))

    @staticmethod
    def resize_image(image_path):
        with Image.open(image_path) as im:
            im = im.resize((320, 512))
            im.save(image_path) 

uid = input("UID : ")
User_Settings.make_dir(uid)
User_Settings.copy_samples(uid)
f_img = open("test_img_cloth.txt", 'r')
cloth1 = f_img.readline()
f_img.close()
f_img = open("test_img_model.txt", 'r')
model1 = f_img.readline()
f_img.close()

cloth_image_path = Image_Preprocessing.generate_path(uid, "cloth", "custom_cloth_front.jpg")
model_image_path = Image_Preprocessing.generate_path(uid, "image", "custom_model_whole_front.png")

pose_json_path = Image_Preprocessing.generate_path(uid, "pose", "")

segmented_model_image_path = Image_Preprocessing.generate_path(uid, "image-parse", "custom_model_whole_front_label.png")
segmented_cloth_image_path = Image_Preprocessing.generate_path(uid, "cloth-mask", "custom_cloth_front_mask.jpg")
#0. Decode Images
Image_Preprocessing.save_decoded_image(cloth1, cloth_image_path)    # 디코딩 된 이미지가 ../user_requests/{uid}/cloth/custom_cloth.jpg 로 저장됨.
Image_Preprocessing.save_decoded_image(model1, model_image_path)    # 디코딩 된 이미지가 ../user_requests/{uid}/cloth/custom_model.png 로 저장됨.

#1. prepare an in-shop clothing image C (→ mpv3d_example/cloth) and a frontal person image P (→ mpv3d_example/image) with resolution of 320*512;
Image_Preprocessing.resize_image(cloth_image_path) # ../user_requests/{uid}/cloth/custom_cloth.jpg 가 320*512 사이즈로 변경됨.
Image_Preprocessing.resize_image(model_image_path) # ../user_requests/{uid}/cloth/custom_model.png 가 320*512 사이즈로 변경됨.

#2. obtain the mask of C (→ mpv3d_example/cloth-mask) by thresholding or using remove.bg;
cloth_preprocess.image_remove_bg(img=cloth_image_path, output_path=segmented_cloth_image_path) # ../user_requests/{uid}/cloth-mask/custom_cloth_mask.jpg 가 생성됨.
cloth_preprocess.image_masking(img=segmented_cloth_image_path, output_path=segmented_cloth_image_path)

#2.5 copy preprocessed cloth image to aligned folder
#User_Settings.copy_train_pairs(uid)
            
#3. obtain the human segmentation layout (→ mpv3d_example/image-parse) by applying 2D-Human-Paring on P;
hs.run(output_path=segmented_model_image_path.replace(os.path.basename(segmented_model_image_path), ""), output_name = os.path.basename(segmented_model_image_path),img=model_image_path) # ../user_requests/{uid}/image-parse/custom_model_segmented.png 가 생성됨.

#4. obtain the human joints (→ mpv3d_example/pose) by applying OpenPose (25 keypoints) on P;
OpenPose.poses_3d(uid, model_image_path, pose_json_path)  # ../user_requests/{uid}/image-parse/custom_model_segmented.png 가 생성됨.

#5. run the data processing script python util/data_preprocessing.py --MPV3D_root mpv3d_example to automatically obtain the remaining inputs (pre-aligned clothing, palm mask, and image gradients);
#data_preprocessing.preprocess_data("../user_requests/{}".format(uid))           # <- util 폴더 경로로 인한 에러 가능성 존재.

data_preprocessing.preprocess_data("../user_requests/{}".format(uid))           # <- util 폴더 경로로 인한 에러 가능성 존재.

#5.5 create depth map


#6. now the data preparation is finished and you should be able to run inference with the steps described in the next section "Running Inference".

#############################################################################################
##################################### Running Inference #####################################
#############################################################################################

# 1. Testing MTM Module
model_inference.run(model_type="MTM", uid=uid)

# 2. Testing DRM Module
model_inference.run(model_type="DRM", uid=uid)

# 3. Testing TFM Module
model_inference.run(model_type="TFM", uid=uid)

# 4. Getting colored point clound and Remeshing
results_dir = os.path.join("../user_requests", uid)
#results_dir = os.path.join("./results", "aligned")
rgbd2pcd.run(results_dir)