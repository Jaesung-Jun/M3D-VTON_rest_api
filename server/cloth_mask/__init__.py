import cv2
import numpy as np
class Preprocessing:
    @staticmethod
    def image_masking(img, output_path):
        img = cv2.imread(img, 0)
        _, masked_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(output_path, masked_img)

    def image_remove_bg(img, output_path):
        
        img = cv2.imread(img)
        gray = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
        mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
        # negate mask
        mask = 255 - mask
        # apply morphology to remove isolated extraneous noise
        # use borderconstant of black since foreground touches the edges
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # anti-alias the mask -- blur then stretch
        # blur alpha channel
       # mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)

        # linear stretch so that 127.5 goes to 0, but 255 stays 255
        mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)

        # put mask into alpha channel
        result = img.copy()
        result = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
        result[:, :] = mask

        cv2.imwrite(output_path, result)
        #print(output_path)
        #return result
        """
        # save resulting masked image
        cv2.imwrite('person_transp_bckgrnd.png', result)

        # display result, though it won't show transparency
        cv2.imshow("INPUT", img)
        cv2.imshow("GRAY", gray)
        cv2.imshow("MASK", mask)
        cv2.imshow("RESULT", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """