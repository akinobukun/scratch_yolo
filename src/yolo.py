import torch
import cv2
from typing import List, Tuple
import uuid


# YOLOはYOLOでClassを作って処理を隔離
class YOLO:

    # コンストラクタでモデルを読み込み
    def __init__(self, pt_path):
        self.__model = torch.hub.load('ultralytics/yolov5', 'custom', path=pt_path)
        self.__model.conf = 0.05
        self.classes = self.__model.classes

    # 物体のleft, top, right, bottomを出力
    def get_bounding_box(self, img_path: str, target_class=1, log=False) -> List:
        """バウディングボックスを取得

        Parameters
        ----------
        img_path : str
            画像のパス
        target_class : int, optional
            ターゲットクラス, by default 1
        log : bool, optional
            ログ, by default False

        Returns
        -------
        List
            バウディングボックスの情報
        """
        data = []
        self.__result = self.__model(img_path)
        self.__image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if log:
            print('----------')
            print(self.__result.pandas().xyxy[0])
        ndresult = self.__result.xyxy[0].numpy()
        for v in ndresult:
            if v[5] == target_class:  # target_classが複数の時はif文で繰り返し
                data.append(
                    [
                        int(v[0]),  # left
                        int(v[1]),  # top
                        int(v[2]),  # right
                        int(v[3]),  # bottom
                        float(v[4]),  # confidence
                    ]
                )
        return data

    def save_predicted_image(self, bb_results: List) -> str:
        for result in bb_results:
            # 信頼度が閾値を上回っていた場合は緑色の矩形を描画
            left, top, right, bottom = result[:4]
            cv2.rectangle(self.__image, (left, top), (right, bottom), (0, 255, 0), 3)
        self.__pred_img_name = str(uuid.uuid4()) + ".jpg"
        cv2.imwrite("static/cache/" + self.__pred_img_name, self.__image)
        return self.__pred_img_name

    def get_predicted_results(self, image_path: str, log: bool) -> Tuple[str, int]:
        bb_results = self.get_bounding_box(image_path, log=log)
        pred_image_name = self.save_predicted_image(bb_results)

        return pred_image_name, len(bb_results)
