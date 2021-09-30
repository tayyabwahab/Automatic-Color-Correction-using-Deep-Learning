import cv2
import numpy as np
import os
import time
from glob import glob
import matplotlib.pyplot as plt



def main(fname):
    if fname != '':
        print('====================================')
        print(fname)
        stime = time.time()
        image_path = fname
        source_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        [h, w,ch]= source_img.shape
        plt.imshow(source_img)
        plt.show()
        if h>w :
            M = cv2.getRotationMatrix2D((w//2, h//2), -90, 1.0)
            source_img= cv2.warpAffine(source_img,M,(w,h))
            print('Rotated')
        img_checker = cv2.imread('ColorCard.jpg')
        print('Image details:',img_checker.shape)
        detector = cv2.mcc.CCheckerDetector_create()
        detector.process(img_checker, cv2.mcc.MCC24)

        checker = detector.getBestColorChecker()

        cdraw = cv2.mcc.CCheckerDraw_create(checker)
        img_draw = img_checker.copy()
        cdraw.draw(img_draw)

        chartsRGB = checker.getChartsRGB()

        src = chartsRGB[:, 1].copy().reshape(24, 1, 3)
        src /= 255.0

        model1 = cv2.ccm_ColorCorrectionModel(src, cv2.mcc.MCC24)

        model1.run()
        ccm = model1.getCCM()
        print("ccm ", ccm)
        loss = model1.getLoss()
        print("loss ", loss)

        img_ = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        img_ = img_.astype(np.float64)
        img_ = img_ / 255
        calibratedImage = model1.infer(img_)
        out_ = calibratedImage * 255
        out_[out_ < 0] = 0
        out_[out_ > 255] = 255
        out_ = out_.astype(np.uint8)

        out_img = cv2.cvtColor(out_, cv2.COLOR_RGB2BGR)
        a=np.max(out_img);
        print("****************88")
        print(a);
        refine = out_img.copy()/255
        lap = cv2.Laplacian(refine,cv2.CV_64F)
        diff = abs(lap-refine)
        diff = diff*255

        etime = time.time()
        print('Processing Time: ', etime - stime, 's')
        

if __name__ == "__main__":
    path = "/data/1.JPG"
    main(path)
    print('Done')
