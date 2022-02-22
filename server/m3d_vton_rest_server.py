"""
버그 목록
1. 거의 동시에 2개의 요청을 보내면 1개의 요청이 씹힘.


Error Codes:

00 : Error occured when Creating User Directory on Server.
01 : Error occured when Copy Sample Images to user directory

10 : Error occured when return sample image list (USER DIRECTORY NOT EXIST)

11 : Error occured when Openpose is not correctly excuted.

"""
#import image_preprocessing
from cloth_mask import Preprocessing as cloth_preprocess
from osAdvanced import File_Control
import human_segmentation as hs

from PIL import Image
import base64
import os
from distutils.dir_util import copy_tree
from shutil import copytree
from util import data_preprocessing
import model_inference
import rgbd2pcd
import logging
import torch
import datetime

import flask
from flask import jsonify, request
from flask_cors import CORS

def create_log_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating log directory.")

create_log_dir("log")

app = flask.Flask(__name__)
CORS(app)

now = datetime.datetime.now()
#logging.basicConfig(filename='log/{}.log'.format(now.strftime('%Y-%m-%d %H:%M:%S')), level=logging.INFO, format=f'[%(levelname)s][%(asctime)s] : %(message)s')
logging.basicConfig(filename='log/{}.log'.format(now.strftime('%Y-%m-%d')), level=logging.INFO, format=f'[%(levelname)s][%(asctime)s] : %(message)s')
formatter = logging.Formatter('[%(levelname)s][%(asctime)s] : %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
app.logger.addHandler(stream_handler)

class ERROR:

    # File Control Errers.
    ERR00 = "00 : Error occured while Creating User Directory on Server."   # 디렉토리 만드는 동안 에러
    ERR01 = "01 : Error occured while Copy Sample Images to user directory." # 이미지 복사하는 동안 에러
    ERR02 = "02 : User Directory Not Exists. (Please request to [wtf/user_directory])"   # 유저 디렉토리가 존재하지 않음
    ########################################################################################
    # Model Error
    ERR10 = "10 : Error ocurred while preprocess data" # 전처리중 에러
    ERR11 = "11 : Error occured while execute OpenPose" # openpose 실행 중 에러
    ERR12 = "12 : Error occured while execute DRM/MTM/TFM models" # 모델 실행중 에러
    ERR13 = "13 : Error occured while execute Human Segmentation model" # Human Segmentation 중 에러
    ########################################################################################
    # User Requests Wrong
    ERR20 = "20 : UID is None" # 유저 아이디가 없음
    ########################################################################################
    # Image PreProcessing Error
    ERR30 = "30 : Error occured while resize image" # 이미지 전처리 중 에러
    ERR31 = "31 : Error occured while convert base64 to image; Maybe This file is not image :(" # base64를 이미지로 변환 중 에러
    ERR32 = "32 : Error occured while masking clothes" # 옷 마스킹 중 에러
    
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
            return False
    
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
            return False

class OpenPose:
    @staticmethod
    def poses_3d(uid, image_path, json_path):
        openpose_stdout = os.popen("bash ./openpose/openpose.sh {0} {1}".format("../" + image_path, "../" + json_path)).read()
        if "successfully finished." in openpose_stdout:
            os.rename("../user_requests/{}/pose/custom_model_whole_front_000000000000_keypoints.json".format(uid), "../user_requests/{}/pose/custom_model_whole_front_keypoints.json".format(uid))
            print("Openpose pose estimation successfully finished.")
        else:
            print(openpose_stdout)
            print("Openpose pose estimation failed.")
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


@app.route('/', methods=['GET', 'POST'])
def index():
    return "<h1>WTF</h1>"

@app.route('/wtf/user_directory', methods=['POST'])
def user_directory():
    user_info = request.json
    uid = user_info.get('uid')
    print(uid)
    app.logger.info("Create user directory request from : {}".format(uid))
    if uid != None:
        if not os.path.isdir("../user_requests/" + uid):
            # Directory Not Exists.
            # Custom Dataset Setting
            # Refer to https://github.com/fyviezhao/M3D-VTON#custom-data
            if User_Settings.make_dir(uid):
                if User_Settings.copy_samples(uid):
                    app.logger.info("User directory successfully created : {}".format(uid))
                    return jsonify({"status": "success"})
                else:
                    app.logger.error("Error : {}".format(ERROR.ERR01))
                    return jsonify({"status": ERROR.ERR01})
            else:
                app.logger.error("Error : {}".format(ERROR.ERR00))
                return jsonify({"status": ERROR.ERR00})
        else:
            # Directory Exists.
            app.logger.warning("User directory already exists : {}".format(uid))
            return jsonify({"status": "directory exists"})
            #return "Your directory already exist!"
    else:
        app.logger.warning("ERROR : {}".format(ERROR.ERR20))
        return jsonify({"status": ERROR.ERR20})

# Sample Clothes List Return (with images)
@app.route('/wtf/3dtryon/sample_clothes_list', methods=['GET', 'POST'])
def sample_clothes_request():
    requests = {
        'uid' : '',
    }
    responses = {
            'sample_clothes' : '', # <- sample cloth image list (encoded to base64)
            'sample_clothes_path' : '', # <- sample cloth image path list (encoded to base64)
            'status' : '',
    }
    
    user_info = request.args
    requests['uid'] = user_info.get('uid')

    base64_cloth_list = []
    if requests['uid'] != None:
        if os.path.isdir("../user_requests/" + requests['uid']):
            cloth_paths = File_Control.searchFilesInDirectory("../user_requests/{}/{}".format(requests['uid'], "cloth"))
            for cloth in cloth_paths:
                with open(cloth, "rb") as image:
                    base64_cloth_list.append(base64.b64encode(image.read()).decode("ascii"))
                    
            responses['sample_clothes'] = base64_cloth_list
            responses['sample_clothes_path'] = cloth_paths
        else:
            responses['status'] = ERROR.ERR01
    
    return jsonify(responses)

# Sample Models List Return (with images) 
@app.route('/wtf/3dtryon/sample_models_list', methods = ['GET', 'POST'])
def sample_models_request():
    requests = {
        'uid' : '',
    }
    responses = {
        'sample_models' : '', # <- sample cloth models list (encoded to base64) 
        'sample_models_path' : '', # <- sample model image paths list (encoded to base64)
        'status' : '',
    }

    user_info = request.args
    requests['uid'] = user_info.get('uid')

    base64_model_list = []
    if requests['uid'] != None:
        if os.path.isdir("../user_requests/" + requests['uid']):
            model_paths = File_Control.searchFilesInDirectory("../user_requests/{}/{}".format(requests['uid'], "image"))
            print("../user_requests/{}/{}".format(requests['uid'], "image"))
            for model in model_paths:
                with open(model, "rb") as image:
                    base64_model_list.append(base64.b64encode(image.read()).decode("ascii"))
                    
            responses['sample_models'] = base64_model_list
            responses['sample_models_path'] = model_paths        
        else:
            responses['status'] = ERROR.ERR01
    
    return jsonify(responses)

@app.route('/wtf/3dtryon/test', methods=['GET', 'POST'])
def tryon_test():
    responses = {
            '3d_model' : '',
            'status': 'success'
        }
    with open('./sample_ply/sample1.ply', "rb") as image:
        responses['3d_model'] = base64.b64encode(image.read()).decode("ascii")

    return jsonify(responses)

@app.route('/wtf/3dtryon', methods=['POST'])
def tryon():
    
    responses = {
                '3d_model' : '',
                'status': ''
            }

    requests = {
            'uid' : '',
            'uploaded_cloth' : '', # <- uploaded cloth image (encoded to base64)
            'uploaded_model' : '' # <- uploaded model image (encoded to base64)
    }
    
    user_info = request.json
    requests['uid'] = user_info.get('uid')
    requests['test'] = user_info.get('test')
    requests['sample_cloth'] = user_info.get('sample_cloth')
    requests['sample_model'] = user_info.get('sample_model')
    requests['uploaded_cloth'] = user_info.get('uploaded_cloth')
    requests['uploaded_model'] = user_info.get('uploaded_model')

    app.logger.info("Request From : {}".format(requests['uid']))

    if requests['uid'] != None:
        if not os.path.isdir("../user_requests/" + requests['uid']):
            responses['status'] = ERROR.ERR02           # Directory not exists.
            app.logger.warning("Error : {}".format(ERROR.ERR02))
            return jsonify(responses)

    cloth_image_path = Image_Preprocessing.generate_path(requests['uid'], "cloth", "custom_cloth_front.jpg")
    model_image_path = Image_Preprocessing.generate_path(requests['uid'], "image", "custom_model_whole_front.png")
    pose_json_path = Image_Preprocessing.generate_path(requests['uid'], "pose", "")

    segmented_model_image_path = Image_Preprocessing.generate_path(requests['uid'], "image-parse", "custom_model_whole_front_label.png")
    segmented_cloth_image_path = Image_Preprocessing.generate_path(requests['uid'], "cloth-mask", "custom_cloth_front_mask.jpg")
    #0. Decode Images
    try:
        Image_Preprocessing.save_decoded_image(requests['uploaded_cloth'], cloth_image_path)    # 디코딩 된 이미지가 ../user_requests/{uid}/cloth/custom_cloth.jpg 로 저장됨.
        Image_Preprocessing.save_decoded_image(requests['uploaded_model'], model_image_path)    # 디코딩 된 이미지가 ../user_requests/{uid}/cloth/custom_model.png 로 저장됨.
    except Exception as e:
        responses['status'] = ERROR.ERR31
        app.logger.warning("Error : {}".format(ERROR.ERR31))
        app.logger.error("Error message : {}".format(e))
        return jsonify(responses)
    #1. prepare an in-shop clothing image C (→ mpv3d_example/cloth) and a frontal person image P (→ mpv3d_example/image) with resolution of 320*512;
    try:
        Image_Preprocessing.resize_image(cloth_image_path) # ../user_requests/{uid}/cloth/custom_cloth.jpg 가 320*512 사이즈로 변경됨.
        Image_Preprocessing.resize_image(model_image_path) # ../user_requests/{uid}/cloth/custom_model.png 가 320*512 사이즈로 변경됨.
    except Exception as e:
        responses['status'] = ERROR.ERR30
        app.logger.warning("Error : {}".format(ERROR.ERR30))
        app.logger.error("Error message : {}".format(e))
        return jsonify(responses)
    #2. obtain the mask of C (→ mpv3d_example/cloth-mask) by thresholding or using remove.bg;
    try:
        cloth_preprocess.image_remove_bg(img=cloth_image_path, output_path=segmented_cloth_image_path) # ../user_requests/{uid}/cloth-mask/custom_cloth_mask.jpg 가 생성됨.
        cloth_preprocess.image_masking(img=segmented_cloth_image_path, output_path=segmented_cloth_image_path)
    except Exception as e:
        responses['status'] = ERROR.ERR32
        app.logger.warning("Error : {}".format(ERROR.ERR32))
        app.logger.error("Error message : {}".format(e))
        return jsonify(responses)
    #3. obtain the human segmentation layout (→ mpv3d_example/image-parse) by applying 2D-Human-Paring on P;
    try:
        hs.run(output_path=segmented_model_image_path.replace(os.path.basename(segmented_model_image_path), ""), output_name = os.path.basename(segmented_model_image_path),img=model_image_path) # ../user_requests/{uid}/image-parse/custom_model_segmented.png 가 생성됨.
    except Exception as e:
        responses['status'] = ERROR.ERR13
        app.logger.error("Error : {}".format(ERROR.ERR13))
        app.logger.error("Error message : {}".format(e))
        return jsonify(responses)
    #time.sleep(1)
    #4. obtain the human joints (→ mpv3d_example/pose) by applying OpenPose (25 keypoints) on P;
    try:
        OpenPose.poses_3d(requests['uid'], model_image_path, pose_json_path)  # ../user_requests/{uid}/image-parse/custom_model_segmented.png 가 생성됨.
    except Exception as e:
        responses['status'] = ERROR.ERR11
        app.logger.error("Error : {}".format(ERROR.ERR11))
        app.logger.error("Error message : {}".format(e))
        return jsonify(responses)
    #5. run the data processing script python util/data_preprocessing.py --MPV3D_root mpv3d_example to automatically obtain the remaining inputs (pre-aligned clothing, palm mask, and image gradients);
    try:
        data_preprocessing.preprocess_data("../user_requests/{}".format(requests['uid']))           # <- util 폴더 경로로 인한 에러 가능성 존재.
    except Exception as e:
            responses['status'] = ERROR.ERR10
            app.logger.error("Error : {}".format(ERROR.ERR10))
            app.logger.error("Error message : {}".format(e))
            print(e)
            return jsonify(responses)    
    #6. now the data preparation is finished and you should be able to run inference with the steps described in the next section "Running Inference".
    
    #############################################################################################
    ##################################### Running Inference #####################################
    #############################################################################################
    try:
        # 1. Testing MTM Module
        model_inference.run(model_type="MTM", uid=requests['uid'])

        # 2. Testing DRM Module
        model_inference.run(model_type="DRM", uid=requests['uid'])

        # 3. Testing TFM Module
        model_inference.run(model_type="TFM", uid=requests['uid'])
        # 4. Getting colored point clound and Remeshing
        results_dir = os.path.join("../user_requests", requests['uid'])
        rgbd2pcd.run(results_dir)
    except Exception as e:
       responses['status'] = ERROR.ERR12           # Directory not exists.
       app.logger.error("Error : {}".format(ERROR.ERR12))
       app.logger.error("Error message : {}".format(e))
       return jsonify(responses)

    
    with open(os.path.join(results_dir, "results", "custom_model.ply"), "rb") as ply:
        ply_base64 = base64.b64encode(ply.read()).decode("utf-8")
        responses['3d_model'] = ply_base64
        responses['status'] = "success"

    app.logger.info("Successfully finished work : {}".format(requests['uid']))
    
    # do something
    # and return json. (Can Convert Dictionary Type to json with jsonify)
    # Empty Caches in GPU Memory
    
    return jsonify(responses)

if __name__ == '__main__':
    # for Development server
    # app.run(debug=False, host='0.0.0.0', port=5000)
    pass
