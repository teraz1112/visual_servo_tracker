# visual_servo_tracker

既存の実験コードを、**元ディレクトリを壊さずに**新規ディレクトリ内へ再構成した実行パッケージです。
このディレクトリ単体でセットアップ・実行・テストが完結します。

## 1) これは何か（機能一覧）
- `dataset_prep`
  - 画像から目標ROI（`goal/0_0.jpg`）とずらし画像群（`gap/*.jpg`）を作成
  - 固定サイズROIモードと自由矩形モードを提供
- `jacobian_modeling`
  - ヤコビアン推定（`jacobian.pkl`）
  - 推定性能の可視化（`graph/*.png`）
  - 目標画像最適化（`optimized/result.jpg`）
- `tracking_runtime`
  - 動画追従
  - Baslerカメラ追従（`pypylon` オプション）
- `tools`
  - グレースケール変換
  - 画像差分表示

## 2) 前提（OS, GPU/CPU, 主要依存）
- OS: Windows / Linux（GUI利用機能はデスクトップ環境が必要）
- Python: 3.10+
- CPU実行可（GPU必須ではない）
- 主要依存:
  - `numpy`, `opencv-python`, `Pillow`, `matplotlib`, `PyYAML`
- Basler使用時のみ:
  - `pypylon`（`requirements-basler.txt`）

## 3) セットアップ（最短手順）
```powershell
cd visual_servo_tracker
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Baslerも使う場合:
```powershell
pip install -r requirements-basler.txt
```

## 4) 最小実行例（コピペ可能なコマンド）
### 4.1 オフライン一括実行（サンプルデータ使用）
```powershell
python -m visual_servo_tracker.cli offline-run --config configs/offline_sample.yaml
```

出力先:
- `outputs/jacobian/circle_red_green/jacobian.pkl`
- `outputs/graph/circle_red_green_normal_graph/*.png`
- `outputs/optimized/circle_red_green/result.jpg`
- `outputs/graph/circle_red_green_optimized_graph/*.png`

### 4.2 サンプル動画トラッキングまで一括実行
```powershell
python -m visual_servo_tracker.cli offline-track-video --config configs/offline_sample.yaml
```
または:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_sample_video_tracking.ps1
```

実行フロー:
1. ヤコビアン生成
2. 評価グラフ生成（normal）
3. 最適化
4. 評価グラフ生成（optimized）
5. サンプル動画 `data/samples/videos/circle_red_green_sample.mp4` で追従開始

操作:
- 追従ウィンドウで対象の左上をクリック
- `s` キーで追従開始
- `q` キーで終了

### 4.3 トラッキングだけ再実行（生成済みヤコビアンを使用）
```powershell
python -m visual_servo_tracker.cli track-video --config configs/offline_sample.yaml
```

### 4.4 ステップ実行
```powershell
python -m visual_servo_tracker.cli build-jacobian --config configs/offline_sample.yaml
python -m visual_servo_tracker.cli evaluate --config configs/offline_sample.yaml --type normal
python -m visual_servo_tracker.cli optimize --config configs/offline_sample.yaml
python -m visual_servo_tracker.cli evaluate --config configs/offline_sample.yaml --type optimized
```

### 4.5 スモークテスト
```powershell
pytest -q
```

## 5) データの置き場所（サンプル、取得方法）
- 最小サンプル入力は同梱済み:
  - `data/samples/circle_red_green/goal/0_0.jpg`
  - `data/samples/circle_red_green/gap/*.jpg`（`max_gap=2` 相当の最小セット）
  - `data/samples/videos/circle_red_green_sample.mp4`（動画追従サンプル）
- 自分のデータを使う場合:
  1. `data/samples/<version>/goal/0_0.jpg` を配置
  2. `data/samples/<version>/gap/<dx>_<dy>.jpg` を配置
  3. （任意）`data/samples/videos/<任意名>.mp4` を配置
  4. `configs/default.yaml` か `--version` で対象を指定

## 6) よくあるエラーと対処
- `FileNotFoundError: ... gap/...jpg`
  - `data/samples/<version>/gap/` の命名が `<dx>_<dy>.jpg` になっているか確認
- `Jacobian shape mismatch`
  - 目標画像サイズとヤコビアン作成時のサイズが一致していない
  - `build-jacobian` を同じ `goal` で再実行
- OpenCVウィンドウが出ない
  - GUI無し環境で実行している可能性
  - `offline-run` はGUI不要、`prep-*`/`track-*` はGUI必須
- `FileNotFoundError: ... circle_red_green_sample.mp4`
  - `data/samples/videos/circle_red_green_sample.mp4` が存在するか確認
  - 別動画を使う場合は `track-video --video <path>` を指定
- Basler追従で `pypylon` エラー
  - `pip install -r requirements-basler.txt`

## 7) プロジェクト構造（各ディレクトリの役割）
```text
visual_servo_tracker/
├─ src/visual_servo_tracker/
│  ├─ cli.py
│  ├─ dataset_prep/
│  ├─ jacobian_modeling/
│  ├─ tracking_runtime/
│  └─ tools/
├─ configs/               # 実行設定
├─ data/samples/          # 最小入力サンプル（画像+動画）
├─ outputs/               # 実行生成物（.gitignore対象）
├─ tests/smoke/           # スモークテスト
├─ third_party/           # 外部コード配置場所（必要時）
└─ scripts/               # 実行補助スクリプト
```

## 8) ライセンス/引用（third_party含む）
- この再構成コード: `LICENSE`（MIT）
- 注意事項: `NOTICE`
- 外部コード運用: `third_party/README.md`
- third_party一覧: `THIRD_PARTY_LICENSES.md`

---

## 変更点（互換性に影響しうる点）
以下は意図的な改善で、動作に影響する可能性があります。

1. パス管理の変更
- 変更: 絶対パス固定を廃止し、`configs/*.yaml` + 相対パス化
- 影響:
  - 入出力: 出力先が `outputs/` 配下に統一
  - 精度: 影響なし
  - 速度: 影響なし
  - 再現性: 環境依存が減り向上

2. ヤコビアン推定時の特異行列フォールバック
- 変更: `inv(B^T B)` が失敗した場合に疑似逆行列へフォールバック
- 影響:
  - 入出力: 生成ファイル形式は同じ
  - 精度: 退化ケースで推定値が変わる可能性あり（ただし処理は継続）
  - 速度: 退化時のみわずかに増加
  - 再現性: エラー停止が減り、運用上の再現性は向上

3. オフライン実行の標準設定
- 変更: `offline_sample.yaml` は最小サンプル向けに `learning_rate=0.01`, `iterations=50`
- 影響:
  - 入出力: 小規模サンプルで短時間完走
  - 精度: 元コードの長時間探索より最適化品質が下がる可能性
  - 速度: 大幅改善
  - 再現性: 同じ設定なら再現可

## 検証手順（推奨）
1. `pytest -q` を実行し、スモークテスト通過を確認
2. `offline-run` で `outputs/` に成果物生成を確認
3. 必要に応じて `optimization.iterations` を増やし精度差を比較
