import cloth_mask
import base64
import cv2
import matplotlib.pyplot as plt

class Image_Preprocessing:
    @staticmethod
    def generate_path(uid, image_type, image_name):
        return "../user_requests/{}/{}/{}".format(uid, image_type, image_name)

    @staticmethod
    def save_decoded_image(encoded_image, image_path):
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(encoded_image))

img = cv2.imread('../user_requests/test_a/cloth/custom_cloth.jpg', 0)
if img is None:
    print("No image found")
else:
    img = cloth_mask.Preprocessing.image_remove_bg(img)
    cv2.imshow("test", img)
    cv2.waitKey(0) 
    cv2.DistroyAllWindows()