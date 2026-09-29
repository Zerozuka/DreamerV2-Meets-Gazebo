# tellus モデルの出典

このディレクトリの中身は自作ではなく、以下のリポジトリから取り込んだものである。

- 取得元: <https://github.com/ICONgroupCWC/Gazebo-Sionna-RT-Integration>
- 取得元のパス: `environments/tellus_unioulu/gazebo/models`
- ライセンス: MIT
- 著作権表示: `Copyright (c) 2026 Intelligent Connectivity and Networks Group`
- 取り込み日: 2026-09-30

MIT ライセンスは著作権表示とライセンス条文の保持を求めるため、全文を下記に転載する。

## 取り込んだファイル

`model.sdf` が参照するものだけを取り込んだ。

| 内容 | 件数 | サイズ |
|------|------|--------|
| `meshes/tellus.dae` | 1 | 14 MB |
| `meshes/EnvSamplerTex.*` (jpg / png) | 35 | 3.9 MB |
| `model.sdf`, `model.config` | 2 | 微小 |

`tellus.dae` はテクスチャ 35 枚すべてを `<init_from>` で参照しているため、全数が必要である。

## 取り込まなかったファイル

上流には以下も含まれるが、`model.sdf` から参照されていないため除外した (計 21 MB)。

| ファイル | サイズ | 備考 |
|----------|--------|------|
| `TELLUS.dae` | 8.9 MB | `TELLUS_arena.dae` とバイト数が同一 |
| `TELLUS_arena.dae` | 8.9 MB | |
| `TELLUS_arena_.stl` | 3.5 MB | 衝突形状の簡略版として使える可能性がある |

必要になれば上流から取得できる。上流にはこの他に Sionna RT 用のシーン (`sionna/tellus.xml` とメッシュ 133 件)、Blender 原本 (`blender/tellus.blend`)、ロボットモデル (`robots/sionna/TIAGo_Base`, `waveshare_jetbot`) がある。

## 変更点

上流からの変更は 2 点のみで、メッシュは無改変である。

- `model.config` の `<name>` を `my_model` から `tellus` に変更した。`<author>` と `<description>` も Blender エクスポートのテンプレート (`Your Name` / `you@example.com` / `Your model from Blender`) が残っていたので実態に合わせた
- `model.sdf` の `<model name="...">` を `my_model` から `tellus` に変更した。あわせて衝突形状についてのコメントを追加した

## 引用

上流の `CITATION.cff` より。研究で使う場合はこちらを引用する。

```yaml
title: "Gazebo-Sionna RT Integration"
authors:
  - family-names: "H.P."
    given-names: "Madushanka"
    orcid: "https://orcid.org/0009-0008-0530-5731"
  - family-names: "Samarakoon"
    given-names: "Sumudu"
    orcid: "https://orcid.org/0000-0002-2382-1982"
  - family-names: "Bennis"
    given-names: "Mehdi"
    orcid: "https://orcid.org/0000-0003-0261-0171"
repository-code: "https://github.com/ICONgroupCWC/Gazebo-Sionna-RT-Integration"
license: MIT
```

## ライセンス全文

```text
MIT License

Copyright (c) 2026 Intelligent Connectivity and Networks Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 注意

この MIT ライセンスは `Gazebo-Sionna-RT-Integration` のものである。**このリポジトリ本体 (`DreamerV2-Meets-Gazebo`) は依然としてライセンス未定**で、全 `package.xml` の `<license>` が `TODO` のままである。著者への確認が必要。

## 追加で受領したモデル (2026-09-30)

同じ著者から個別にメールで受領した 3 モデルも `models/` 配下に置いた。
`Gazebo-Sionna-RT-Integration` には含まれていないものである。

| モデル | 内容 |
|--------|------|
| `race_end` | レーストラックのゴールライン (`short_path.sdf`) |
| `road_model` | レーストラック本体。`road_section_0` 以下 350 個の入れ子モデル |
| `road_material` | 路面テクスチャ `road_texture.jpg` (894x894) |

受領物からの変更点:

- `road_model/model.config` の `<name>` が `my_model` のままだったので
  `road_model` に修正した。`race_end` も含め `<author>` と `<description>` が
  Blender エクスポートのテンプレート (`Your Name` 等) だったので実態に合わせた
- **Ogre material script を PBR 指定に置き換えた** (`road_model.sdf` で 174 箇所、
  `race_end/short_path.sdf` で 1 箇所)。Harmonic は Ogre material script に
  非対応で、そのままだと 174 件の警告が出て路面が無地になる。`MyRoad/Road` の
  定義 (ambient 0.8 / diffuse 1.0 / `road_texture.jpg`) を `<albedo_map>` に移した
- `race_end` が参照していた `model://end_line/materials/...` は、その `end_line`
  モデルが受領物に含まれていないため解決できない。単色 (ほぼ白) に置き換えた
- `road_model/model.sdf` (22 行) は取り込んでいない。`model.config` が
  `road_model.sdf` を指しており未使用で、中身は `model://tellus/meshes/tellus.dae`
  を参照する tellus モデルの残骸だった
- 同梱の `road_material/materials/scripts/road.material` は Harmonic では
  使われないが、元定義の参照用に残している

著者が同時に送ってきた `tellus3_with_road.world` は取り込んでいない。差分を
取った結果、こちらの版に対して新しい内容はなく、違いは (1) Ogre material
script が残っている、(2) include の多くがコメントアウトされている、
(3) gz-sim システムプラグインが無い、の 3 点のみで、いずれもこちらの版の方が
Harmonic に適合していた。

### まだ入手できていないモデル

`cross_line`, `receiver_1`, `receiver_2`, `receiver_3` の 4 件。
`tellus3_with_road.world` では該当の `<include>` をコメントアウトしている
(Harmonic は解決できない `<include>` があると world 全体を読み込めない)。
`pose` は残してあるので、受領したらコメントを外すだけでよい。
