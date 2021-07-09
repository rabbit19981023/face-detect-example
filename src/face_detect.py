# 匯入OpenCV模組
import cv2

# 定義"人臉辨識"函式，接收一張圖片作為傳入參數
def detect_face(src_img):
  origin_img = cv2.imread(src_img) # 讀取原始圖片
  gray_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY) # 將原始圖片轉為灰階圖片，方便程式讀取

  face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # 導入OpenCV內建的人臉辨識工具
  faces = face_classifier.detectMultiScale(gray_img, minSize=(40, 40)) # 使用人臉辨識工具，對灰階圖片進行辨識，並設定最小尺寸為(40x40)像素
  frame_color = (0, 255, 0) # 定義方框顏色

  # 當faces的清單長度>0，代表有找到人臉
  if len(faces):
    # 用for迴圈，一一地把每個找到的人臉HighLight起來
    for face in faces:
      x, y, w, h = face # 取得每張人臉的(x, y)座標，與它佔的寬、高

      # 在找到的人臉上畫一個方框(原始圖片, 起始座標(左上), 對角座標(右下), 方框顏色, 方框粗細(像素))
      cv2.rectangle(origin_img, (x, y), (x + w, y + h), frame_color, 2)

  filename = src_img.split('/')[-1].split('.')[0] # 刪除任何路徑、任何附檔名(jpg, png等)
  cv2.imwrite('./result/' + filename + '_result.jpg', origin_img) # 輸出結果圖片(檔名：原始檔名_result.jpg)

  # 看輸出結果(僅供測試用)
  cv2.imshow('result', origin_img)
  cv2.waitKey(0)

# 執行我們剛編寫好的函式
detect_face('./origin/crowd.jpg')
