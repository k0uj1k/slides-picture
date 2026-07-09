"""
スライド写真の歪み補正ツール
OpenCVを使用して、斜めから撮影したスライド写真を正面から撮影した状態に補正します。
文字レベルでも水平化します。
"""

import argparse
from pathlib import Path

import cv2
import numpy as np




class SlideCorrector:
    """スライド写真の歪み補正クラス"""

    def __init__(
        self,
        input_dir="input",
        output_dir="output",
    ):
        """
        Args:
            input_dir: 入力画像フォルダパス
            output_dir: 出力画像フォルダパス
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # FHD解像度
        self.fhd_width = 1920
        self.fhd_height = 1080

    def detect_slide_contour(self, image, factor=0.02):
        """
        スライド領域の輪郭検出

        Args:
            image: 入力画像
            factor: 輪郭近似の厳密度を表す係数

        Returns:
            検出された輪郭（最大の4角形）
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = factor * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) == 4:
            return approx
        return None

    def quantize_gray_image(self, image, levels: int):
        """指定した階調数でグレースケール画像を量子化する。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if levels <= 1:
            return gray
        quantized = np.round(gray / 255.0 * (levels - 1)).astype(np.uint8)
        quantized = (quantized * (255 // (levels - 1))).astype(np.uint8)
        return quantized

    def detect_slide_region(self, image, factor=0.02):
        """16〜1段階の再量子化と同一矩形判定を交えて探索し、妥当なスライド矩形を特定する。"""
        # 最初の通常検出（階調処理なし）を試みる
        contour = self.detect_slide_contour(image, factor=factor)
        
        remembered_contour = None
        remembered_area = 0.0

        if contour is not None:
            # 見つかった場合、これを最初の記憶（基準）とするが、即時リターンはせずにループで検証する
            pts = contour.reshape(4, 2).astype(np.float32)
            remembered_contour = self.order_points(pts)
            remembered_area = float(cv2.contourArea(remembered_contour))
            print(f"初期検出: 矩形を検出しました（基準面積: {remembered_area:.1f}）")
        else:
            print("初期検出: 矩形が検出されませんでした。")

        print("スライド領域の妥当性を再確認するため、階調を16→15→14...→1で再検出します。")

        for levels in range(16, 0, -1):
            quantized = self.quantize_gray_image(image, levels=levels)
            quantized_bgr = cv2.cvtColor(quantized, cv2.COLOR_GRAY2BGR)
            contour = self.detect_slide_contour(quantized_bgr, factor=factor)
            
            if contour is not None:
                # 検出された輪郭の面積を計算
                pts = contour.reshape(4, 2).astype(np.float32)
                ordered_pts = self.order_points(pts)
                area = float(cv2.contourArea(ordered_pts))

                if remembered_contour is None:
                    # まだ記憶されている矩形がない場合（初期検出なしだった場合）、記憶する
                    remembered_contour = ordered_pts
                    remembered_area = area
                    print(f"階調 {levels}: 最初の矩形を検出しました（面積: {area:.1f}）")
                else:
                    # すでに記憶されている矩形がある場合、同じ矩形か判定する (cv2.intersectConvexConvexを使用)
                    retval, intersection = cv2.intersectConvexConvex(remembered_contour, ordered_pts)
                    if retval > 0:
                        inter_area = float(retval)
                        union_area = remembered_area + area - inter_area
                        iou = inter_area / union_area if union_area > 0 else 0.0
                    else:
                        iou = 0.0

                    print(f"階調 {levels}: 矩形を検出しました（面積: {area:.1f}, IoU: {iou:.3f}）")

                    # IoU 閾値 0.90 を超えたら同一矩形と判定して採用
                    if iou >= 0.90:
                        print(f"-> 基準（または前の階調）とほぼ同じ矩形が検出されました（IoU: {iou:.3f} >= 0.90）。この矩形を採用します。")
                        return ordered_pts.reshape(4, 1, 2).astype(np.int32)
                    else:
                        # 異なる矩形の場合、面積の大きい方を記憶
                        if area > remembered_area:
                            print(f"-> 異なる矩形です。面積が大きいため記憶を更新します（{remembered_area:.1f} -> {area:.1f}）")
                            remembered_contour = ordered_pts
                            remembered_area = area
                        else:
                            print(f"-> 異なる矩形ですが、記憶している矩形の方が面積が大きいため、記憶を維持します。")
            else:
                if remembered_contour is not None:
                    print(f"階調 {levels}: 矩形が検出されませんでした。記憶している矩形を維持します。")

        # ループ終了後、記憶された矩形があればそれを適用
        if remembered_contour is not None:
            print(f"階調 1 まで探索完了。最大面積 of 矩形を採用します（面積: {remembered_area:.1f}）")
            return remembered_contour.reshape(4, 1, 2).astype(np.int32)

        return None

    def order_points(self, pts):
        """
        4つの点を左上、右上、右下、左下の順序に並べ替える

        Args:
            pts: 4つの点の配列

        Returns:
            順序が整えられた点の配列
        """
        pts = pts.reshape(4, 2)

        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def perspective_transform(self, image, pts):
        """
        透視変換を適用

        Args:
            image: 入力画像
            pts: 4つの点（左上、右上、右下、左下）

        Returns:
            変換後の画像
        """
        rect = self.order_points(pts)

        widthA = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
        widthB = np.sqrt(((rect[3][0] - rect[2][0]) ** 2) + ((rect[3][1] - rect[2][1]) ** 2))
        max_width = max(int(widthA), int(widthB))

        heightA = np.sqrt(((rect[3][0] - rect[0][0]) ** 2) + ((rect[3][1] - rect[0][1]) ** 2))
        heightB = np.sqrt(((rect[2][0] - rect[1][0]) ** 2) + ((rect[2][1] - rect[1][1]) ** 2))
        max_height = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        return warped

    def estimate_text_rotation_angle(self, image):
        """テキストの主な回転角を推定して、水平化用の角度を返す。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=10, maxLineGap=3)
        if lines is None or len(lines) < 3:
            return 0.0

        angles = []
        for line in lines:
            if line.ndim == 2 and line.shape[1] == 4:
                line_points = line
            elif len(line) == 4:
                line_points = [line]
            else:
                continue

            for x1, y1, x2, y2 in line_points:
                if x2 == x1:
                    continue
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) > 45:
                    angle = angle - 90 if angle > 0 else angle + 90
                angles.append(angle)

        if not angles:
            return 0.0

        median_angle = np.median(angles)
        return float(median_angle)

    def rotate_image(self, image, angle, border_value=(255, 255, 255)):
        """指定した角度（度）だけ画像を回転する。"""
        if abs(angle) < 0.5:
            return image.copy()

        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        matrix[0, 2] += (new_width / 2) - center[0]
        matrix[1, 2] += (new_height / 2) - center[1]
        rotated = cv2.warpAffine(image, matrix, (new_width, new_height), borderValue=border_value)
        return rotated

    def rotate_to_horizontal(self, image):
        """推定した角度だけ画像を回転して、文字を水平にする。"""
        angle = self.estimate_text_rotation_angle(image)
        return self.rotate_image(image, -angle)

    def resize_to_fhd(self, image):
        """
        画像をFHD（1920x1080）にリサイズ

        Args:
            image: 入力画像

        Returns:
            リサイズされた画像
        """
        height, width = image.shape[:2]

        aspect_ratio = width / height
        fhd_ratio = self.fhd_width / self.fhd_height

        if aspect_ratio > fhd_ratio:
            new_width = self.fhd_width
            new_height = int(self.fhd_width / aspect_ratio)
        else:
            new_height = self.fhd_height
            new_width = int(self.fhd_height * aspect_ratio)

        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        result = np.ones((self.fhd_height, self.fhd_width, 3), dtype=np.uint8) * 255
        y_offset = (self.fhd_height - new_height) // 2
        x_offset = (self.fhd_width - new_width) // 2
        result[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
        return result

    def correct_slide(self, image_path):
        """
        1枚のスライド画像を補正。最大3段階の厳密度で検出を試行し、検出に失敗した場合は
        明度反転画像を用いて再試行します。最終的にも検出できない場合は
        補正なしでFHDサイズにリサイズして保存します。

        Args:
            image_path: 画像ファイルパス

        Returns:
            補正または保存が成功したかどうか
        """
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"エラー: {image_path} を読み込めません")
            return False

        print(f"処理中: {image_path.name}")

        factors = [0.02, 0.03, 0.04, 0.06, 0.08]
        contour = None
        
        # 1. 元の画像で段階的緩和 (0.02 -> 0.03 -> 0.04 -> 0.06 -> 0.08) を試行
        for stage, factor in enumerate(factors, start=1):
            if stage > 1:
                print(f"-> 検出失敗のため厳密度を緩和して再試行します (第{stage}段階: factor={factor})")
            contour = self.detect_slide_region(image, factor=factor)
            if contour is not None:
                break

        # 2. 検出失敗した場合、明度反転画像で再試行
        if contour is None:
            print("-> 元の画像でスライド領域が検出されなかったため、明度反転画像を用いて再検出を試みます。")
            inverted_image = cv2.bitwise_not(image)
            for stage, factor in enumerate(factors, start=1):
                print(f"-> 明度反転画像で検出を試みます (第{stage}段階: factor={factor})")
                contour = self.detect_slide_region(inverted_image, factor=factor)
                if contour is not None:
                    print("-> 明度反転画像によりスライド領域を検出しました。")
                    break

        output_path = self.output_dir / image_path.name

        # 3. それでも検出できなかった場合は救済保存
        if contour is None:
            print(f"警告: {image_path.name} で明度反転を含む全探索でもスライド領域が検出できませんでした。")
            print(f"-> 補正処理をスキップし、元の画像をFHDリサイズして保存します。")
            resized = self.resize_to_fhd(image)
            cv2.imwrite(str(output_path), resized)
            print(f"完了 (補正なし保存): {output_path}")
            return True

        # 4. 検出できた場合は元の画像に適用して歪み補正実行
        transformed = self.perspective_transform(image, contour)
        horizontal = self.rotate_to_horizontal(transformed)
        resized = self.resize_to_fhd(horizontal)

        cv2.imwrite(str(output_path), resized)
        print(f"完了 (歪み補正完了): {output_path}")
        return True
    def process_all(self):
        """
        inputフォルダのすべての画像を処理。
        処理開始時に、出力先フォルダ (output) の内容を空にします。
        """
        if not self.input_dir.exists():
            print(f"エラー: {self.input_dir} が見つかりません")
            return

        # output フォルダ内の既存の画像・ファイルを削除
        if self.output_dir.exists():
            print(f"出力フォルダ {self.output_dir} を空にしています...")
            for f in self.output_dir.iterdir():
                if f.is_file():
                    try:
                        f.unlink()
                    except Exception as e:
                        print(f"警告: ファイル {f} の削除に失敗しました: {e}")
        else:
            self.output_dir.mkdir(exist_ok=True)

        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in self.input_dir.iterdir()
            if f.suffix.lower() in supported_formats
        ]

        if not image_files:
            print(f"警告: {self.input_dir} に画像ファイルが見つかりません")
            return

        print(f"合計 {len(image_files)} 枚の画像を処理します")
        print("=" * 50)

        success_count = 0
        for image_path in sorted(image_files):
            if self.correct_slide(image_path):
                success_count += 1

        print("=" * 50)
        print(f"処理完了: {success_count}/{len(image_files)} 枚成功")


def parse_args():
    parser = argparse.ArgumentParser(description="スライド写真の歪み補正")
    parser.add_argument("--input-dir", default="input")
    parser.add_argument("--output-dir", default="output")
    return parser.parse_args()


def main():
    """メイン処理"""
    args = parse_args()
    corrector = SlideCorrector(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    corrector.process_all()


if __name__ == "__main__":
    main()
