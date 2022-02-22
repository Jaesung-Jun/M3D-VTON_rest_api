import json
import requests
import base64

url = 'http://127.0.0.1:5000/wtf/user_directory'
response = requests.post(url, json={"uid": "test33334"})
if response.ok:
    print(response.json())

f_img = open("test_img_cloth.txt", 'r')
cloth1 = f_img.readline()
f_img.close()
f_img = open("test_img_model.txt", 'r')
model1 = f_img.readline()
f_img.close()

url = 'http://127.0.0.1:5000/wtf/3dtryon'
response = requests.post(url, 
                         json={
                             "uid": "test33334", 
                             "uploaded_cloth": cloth1,
                             "uploaded_model": model1
                         })
if response.ok:
    #print(response.json())
    result = response.json()['3d_model']
    result = base64.b64decode(result).decode('utf-8')
    with open("result_test.ply", 'w') as f:
        f.write(result)