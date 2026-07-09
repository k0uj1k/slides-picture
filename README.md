# スライド写真補正ツール

会議で撮影したスライド写真の歪みを自動補正するツールです。OpenCVを使用して、斜めから撮影した写真を正面から撮影した状態に補正します。

## 機能

- **スライド領域の自動検出**: OpenCVの輪郭検出を使用してスライド領域を自動的に特定
- **透視変換**: 4隅を推定して Perspective Transform を適用
- **FHD対応**: 補正後の画像を 1920×1080 にリサイズ
- **バッチ処理**: input フォルダの複数の画像を一括処理
- **ファイル名保持**: 元のファイル名を保持してoutputフォルダに保存

## 処理フロー

```
画像読み込み
    ↓
グレースケール化・前処理
    ↓
輪郭検出
    ↓
スライド領域の検出（最大の4角形）
    ↓
4隅の推定
    ↓
Perspective Transform（透視変換）
    ↓
FHDへのリサイズ
    ↓
output フォルダに保存
```

## セットアップ

### 必要なライブラリ

```bash
pip install opencv-python numpy pillow
```

すでにインストール済みの場合は不要です。

## 使い方

### 1. 入力画像を準備

`input` フォルダに補正したい画像ファイルを配置してください。

サポートされている形式:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

```
slides-picture/
├── input/
│   ├── slide01.jpg
│   ├── slide02.jpg
│   └── slide03.jpg
├── output/
├── slide_corrector.py
└── README.md
```

### 2. スクリプトを実行

```bash
python slide_corrector.py
```

Windows PowerShellの場合:
```powershell
python .\slide_corrector.py
```

### 3. 結果を確認

`output` フォルダに補正された画像が保存されます。元のファイル名が保持されます。

## 処理結果の例

入力画像（歪みあり） → 出力画像（補正済み、FHD 1920×1080）

## カスタマイズ

### 出力解像度を変更する場合

`slide_corrector.py` の `SlideCorrector` クラスの `__init__` メソッドを修正してください：

```python
def __init__(self, input_dir="input", output_dir="output"):
    self.fhd_width = 1920   # 幅を変更
    self.fhd_height = 1080  # 高さを変更
```

### 検出精度を調整する場合

`detect_slide_contour` メソッドの以下のパラメータを調整してください：

```python
# 輪郭近似の精度（小さいほど厳密、大きいほど簡潔）
epsilon = 0.02 * cv2.arcLength(largest_contour, True)
```

## トラブルシューティング

### スライド領域が検出されない場合

1. **照明条件**: 画像の対比が不十分な可能性があります
   - 入力画像の品質を確認してください
   - コントラスト調整を試してください

2. **スライドのサイズ**: スライドが画像の小さな領域を占めている場合
   - より近くから撮影してください

3. **複雑な背景**: スライドの背後に複雑な背景がある場合
   - シンプルな背景で撮影してください

### 変換結果が不自然な場合

- `epsilon` の値を調整して、輪郭近似の精度を変更してください
- 画像の前処理パラメータ（ガウシアンフィルタのカーネルサイズなど）を調整してください

## 使用技術

- **OpenCV**: 画像処理
- **NumPy**: 数値計算
- **Pillow**: 画像形式の変換

## ライセンス

自由に使用・修正できます。
