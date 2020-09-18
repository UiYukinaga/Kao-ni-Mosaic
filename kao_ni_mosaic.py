# ********************************************************
# 検出した顔にモザイクをかけるサンプルコード
# --------------------------------------------------------
# 入力画像から顔を検出して、検出したエリアをモザイク処理する
# haarcascade_frontalface_alt.xmlを同じディレクトリに入れてね
# --------------------------------------------------------
# 第1引数: 画像ファイルのパス
# ********************************************************
import cv2
import os
import sys

# モザイク処理
def mosaic(img, alpha):
    # 画像の高さと幅
    w = img.shape[1]
    h = img.shape[0]

    # 最近傍法で縮小→拡大することでモザイク加工
    img = cv2.resize(img, (int(w*alpha), int(h*alpha)))
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

    return img

# 特定のエリアのモザイク処理
def mosaic_area(src, x, y, width, height, ratio):
    dst = src.copy()
    dst[y:y + height, x:x + width] = mosaic(dst[y:y + height, x:x + width], ratio)
    return dst

def kaoni_mosaic(img_path):

    # 画像の読み込み by OpenCV
    img = cv2.imread(img_path)

    # カスケード分類器を生成する
    # 正面の顔検出
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    # 顔を検出する(ここでの戻り値がバウンディングボックスの座標)
    facerect = cascade.detectMultiScale(img)
    # ********************************************************
    # NOTE
    # --------------------------------------------------------
    # detectMultiScale()の返値は、バウンディングボックスの座標となる。
    # 座標は配列になっていて、[X座標, Y座標, 幅, 高さ]となっている。
    # さらに検出した顔の数が要素数となっていて、[要素No.][X, Y, W, H]
    # という2次元配列になっている。
    # ********************************************************
    #検出した顔を四角い枠線で囲む (検出した顔の数だけ for で繰り返す)
    i = 0
    for rect in facerect:
        # 顔のX,Y座標と、幅, 高さを参照する
        x = facerect[i][0]
        y = facerect[i][1]
        w = facerect[i][2]
        h = facerect[i][3]
        # 顔にモザイクをかける
        img = mosaic_area(img, x, y, w, h, 0.1)
        i += 1

    if i > 0:
        print(str(i)+"人の顔を見つけたから、モザイクをかけておいたよ。")
    else:
        print("人が検出されませんでした。")

    # 画像を表示する
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('mosaic_img.jpg', img)

if __name__ == '__main__':
    args = sys.argv
    img_path = args[1]
    kaoni_mosaic(img_path)