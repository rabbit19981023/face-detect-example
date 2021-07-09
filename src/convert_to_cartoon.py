import cv2
import numpy

def convert_to_cartoon(src_img):
  origin_img = cv2.imread(src_img)
  gray_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)

  img = cv2.medianBlur(gray_img, 5) # 模糊
  edges = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5) # 調整最後兩個參數，黑色輪廓

  color = cv2.bilateralFilter(origin_img, 9, 130, 130) # 模糊
  cartoon = cv2.bitwise_and(color, color, mask=edges) # 濾鏡結果

  filename = src_img.split('/')[-1].split('.')[0]
  cv2.imwrite('./result/' + filename + '_cartoon.jpg', cartoon)

  # output
  cv2.imshow('cartoon', cartoon)
  cv2.waitKey(0)

convert_to_cartoon('./origin/dog.jpg')
