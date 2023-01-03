# Stable Diffusion Jp

Stable Diffusion Jp は Stable Diffusion のテキスト部分(README.md 等)を翻訳し、テキストの入力部分を日本語で可能にしたものです。
使用する場合や、使用ルール形態は下記を参照ください。
※[Stable Diffusion](https://github.com/CompVis/stable-diffusion)

[**High-Resolution Image Synthesis with Latent Diffusion Models**](https://ommer-lab.com/research/latent-diffusion-models/)<br/>
[Robin Rombach](https://github.com/rromb)\*,
[Andreas Blattmann](https://github.com/ablattmann)\*,
[Dominik Lorenz](https://github.com/qp-qp)\,
[Patrick Esser](https://github.com/pesser),
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>
_[CVPR '22 Oral](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html) |
[GitHub](https://github.com/CompVis/latent-diffusion) | [arXiv](https://arxiv.org/abs/2112.10752) | [Project page](https://ommer-lab.com/research/latent-diffusion-models/)_

![txt2img-stable2](assets/stable-samples/txt2img/merged-0006.png)
[Stable Diffusion](#stable-diffusion-v1)は、潜在的なテキストから画像への拡散モデルです。[Stability AI](https://stability.ai/)からの寛大な計算機の寄付と[LAION](https://laion.ai/)からのサポートのおかげで、[LAION-5B](https://laion.ai/blog/laion-5b/)データベースのサブセットからの 512x512 画像で Latent Diffusion Model を学習させることができました。Google の[Imagen](https://arxiv.org/abs/2205.11487)と同様に、このモデルは凍結された CLIP ViT-L/14 テキストエンコーダーを使用して、テキストプロンプトでモデルを条件付けしています。860M の UNet と 123M のテキストエンコーダを持つこのモデルは、比較的軽量で、少なくとも 10GB の VRAM を持つ GPU で動作します。以下の[この項](#stable-diffusion-v1)と[モデルカード](https://huggingface.co/CompVis/stable-diffusion)をご覧ください。

## Requirements

ldm という名前の適切な[conda 環境](https://conda.io/)を作成し、起動することができます。

```
conda env create -f environment.yaml
conda activate ldm
```

また、既存の環境を更新するには、以下のように実行します。
[最近環境](https://github.com/CompVis/latent-diffusion)

```
conda install pytorch torchvision -c pytorch
pip install transformers==4.19.2 diffusers invisible-watermark
pip install -e .
```

## Stable Diffusion v1

Stable Diffusion v1 とは、拡散モデルに 860M UNet と CLIP ViT-L/14 テキストエンコーダを用いたダウンサンプリングファクター 8 のオートエンコーダを用いたモデルアーキテクチャの特定の構成を指します。このモデルは 256x256 画像で事前学習を行い、その後 512x512 画像で微調整を行いました。

注：Stable Diffusion v1 は一般的なテキストから画像への拡散モデルであるため、その学習データに存在するバイアスや（誤った）概念を反映します。学習手順やデータ、モデルの使用目的などの詳細は、対応する[モデルカード](Stable_Diffusion_v1_Model_Card.md)に記載されています。

重みは、[Hugging Face の CompVis 組織](https://huggingface.co/CompVis)を通じて、モデルカードで知らされた誤用や害を防ぐための特定の使用ベースの制限を含むが、それ以外は寛容なライセンスで利用可能です。ライセンスの条件下で商用利用は許可されていますが、ウェイトには既知の限界と偏りがあり、一般的なテキスト画像モデルの安全かつ倫理的な展開に関する研究は現在進行中なので、追加の安全機構と配慮なしに提供されたウェイトをサービスや製品に使用することはお勧めしません。重みは研究成果物であり、そのように扱われるべきである。

CreativeML OpenRAIL M ライセンスは、[BigScience](https://bigscience.huggingface.co/)と[RAIL Initiative](https://www.licenses.ai/)が責任ある AI ライセンスの領域で共同で行っている作業から適応された、[Open RAIL M ライセンス](https://www.licenses.ai/blog/2022/8/18/naming-convention-of-responsible-ai-licenses)である。ライセンスのベースとなった[BLOOM Open RAIL ライセンス](https://bigscience.huggingface.co/blog/the-bigscience-rail-license)についての記事もご覧ください。

### 重み

現在、以下のチェックポイントを提供しています。:

- `sd-v1-1.ckpt`: [laion2B-en](https://huggingface.co/datasets/laion/laion2B-en)で解像度 256x256 で 237k ステップ。[laion-high-resolution](https://huggingface.co/datasets/laion/laion-high-resolution)で解像度 512x512 で 194k ステップ（解像度>= 1024x1024 の LAION-5B で 170M の例）。
- `sd-v1-2.ckpt`: sd-v1-1.ckpt から作成。 [laion-aesthetics v2 5+](https://laion.ai/blog/laion-aesthetics/) で解像度 512x512 で 515k ステップ（laion2B-en のサブセットで推定美学スコア>5.0、さらにオリジナルサイズ>= 512x512、推定透かし確率<0.5 の画像にフィルタリングしています。透かしの推定値は[LAION-5B](https://laion.ai/blog/laion-5b/) のメタデータから、美観の推定値は[LAION-Aesthetics Predictor V2](https://github.com/christophschuhmann/improved-aesthetic-predictor)から得ています）。
- `sd-v1-3.ckpt`: sd-v1-2.ckpt から作成。laion-aesthetics v2 5+ で解像度 512x512 で 195k ステップ、[分類器不要のガイダンスサンプリング](https://arxiv.org/abs/2207.12598)を改善するためテキストコンディショニングを 10%ドロップした。
- `sd-v1-4.ckpt`: sd-v1-2.ckpt から作成。laion-aesthetics v2 5+ で解像度 512x512 で 225k ステップ、[分類器不要のガイダンスサンプリング](https://arxiv.org/abs/2207.12598)を改善するためテキストコンディショニングを 10%ドロップした。

クラシファイアフリーのガイダンススケール（1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0） と PLMS サンプリングステップ 50 を用いた評価では、チェックポイントの相対的な向上が確認された。

![sd evaluation results](assets/v1-variants-scores.jpg)

### テキストから画像へ(Text-to-Image) with Stable Diffusion

![txt2img-stable2](assets/stable-samples/txt2img/merged-0005.png)
![txt2img-stable2](assets/stable-samples/txt2img/merged-0007.png)

Stable Diffusion は、CLIP ViT-L/14 テキストエンコーダの（非プール化）テキスト埋め込みを条件とした潜在的拡散モデルです。
[サンプリングのためのリファレンススクリプト](#reference-sampling-script)を提供していますが、[diffusers の統合](#diffusers-integration)も存在し、より活発なコミュニティによる開発が期待されます。

#### リファレンスサンプリングスクリプト

下記各項目を組み込んだ参考サンプリングスクリプトを提供します。

- [Safety Checker モジュール](https://github.com/CompVis/stable-diffusion/pull/36),
  明示的な出力が発生する確率を低減する,
- 出力に[不可視の電子透かし](https://github.com/ShieldMnt/invisible-watermark)を入れることで、[視聴者が機械で作成された画像](scripts/tests/test_watermark.py)であることを識別しやすくする。

[`stable-diffusion-v1-*-original`の重みを設定した](#weights)後, 紐づける

```
mkdir -p models/ldm/stable-diffusion-v1/
ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt
```

そして下記コードでサンプルを実行させる

```
python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms
```

デフォルトでは、[PLMS](https://arxiv.org/abs/2202.09778)サンプラーの[Katherine Crowson の実装](https://github.com/CompVis/latent-diffusion/pull/51)である --scale 7.5 のガイダンススケールを使用し、サイズ 512x512 の画像をレンダリングし 50 段階で学習させた者です。
支持されたすべての論点を以下に示します。(`python scripts/txt2img.py --help`)

```commandline
usage: txt2img.py [-h] [--prompt [PROMPT]] [--outdir [OUTDIR]] [--skip_grid] [--skip_save] [--ddim_steps DDIM_STEPS] [--plms] [--laion400m] [--fixed_code] [--ddim_eta DDIM_ETA]
                  [--n_iter N_ITER] [--H H] [--W W] [--C C] [--f F] [--n_samples N_SAMPLES] [--n_rows N_ROWS] [--scale SCALE] [--from-file FROM_FILE] [--config CONFIG] [--ckpt CKPT]
                  [--seed SEED] [--precision {full,autocast}]

optional arguments:
  -h, --help            show this help message and exit
  --prompt [PROMPT]     the prompt to render
  --outdir [OUTDIR]     dir to write results to
  --skip_grid           do not save a grid, only individual samples. Helpful when evaluating lots of samples
  --skip_save           do not save individual samples. For speed measurements.
  --ddim_steps DDIM_STEPS
                        number of ddim sampling steps
  --plms                use plms sampling
  --laion400m           uses the LAION400M model
  --fixed_code          if enabled, uses the same starting code across samples
  --ddim_eta DDIM_ETA   ddim eta (eta=0.0 corresponds to deterministic sampling
  --n_iter N_ITER       sample this often
  --H H                 image height, in pixel space
  --W W                 image width, in pixel space
  --C C                 latent channels
  --f F                 downsampling factor
  --n_samples N_SAMPLES
                        how many samples to produce for each given prompt. A.k.a. batch size
  --n_rows N_ROWS       rows in the grid (default: n_samples)
  --scale SCALE         unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
  --from-file FROM_FILE
                        if specified, load prompts from this file
  --config CONFIG       path to config which constructs model
  --ckpt CKPT           path to checkpoint of model
  --seed SEED           the seed (for reproducible sampling)
  --precision {full,autocast}
                        evaluate at this precision
```

注意：すべての v1 バージョンの推論設定は、EMA のみのチェックポイントで使用するように設計されています。
このため、`use_ema=False`を設定すると、EMA 以外の重みから EMA の重みへの切り替えが行われます。
そうでない場合、コードは非 EMA ウェイトから EMA ウェイトに切り替えようとします。EMA と EMA なしの効果を比較したい場合は、両方の重みが含まれる完全なチェックポイントを提供します。
を用意しています。この場合、`use_ema=False`を指定すると、EMA でない重みが読み込まれ、使用されます。

#### ディフューザー一体型(Diffusers Integration)

Stable Diffusion をダウンロードし、サンプリングする簡単な方法は、[diffusers ライブラリ](https://github.com/huggingface/diffusers/tree/main#new--stable-diffusion-is-now-fully-compatible-with-diffusers)を使用することです。

```py
# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4",
	use_auth_token=True
).to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
    image = pipe(prompt)["sample"][0]

image.save("astronaut_rides_horse.png")
```

### 安定した拡散を用いた画像改質

[SDEdit](https://arxiv.org/abs/2108.01073)によって初めて提案された拡散-非拡散機構を用いることで、このモデルはテキストガイド付きの画像間変換やアップスケーリングなどの異なるタスクに利用することができます。txt2img サンプリングスクリプトと同様に、Stable Diffusion を用いた画像修正を行うスクリプトを提供する。

以下では、[Pinta](https://www.pinta-project.com/)で作成したラフスケッチを、詳細なアートワークに変換する例について説明します。

```
python scripts/img2img.py --prompt "A fantasy landscape, trending on artstation" --init-img <path-to-img.jpg> --strength 0.8
```

ここで、strength は 0.0 から 1.0 の間の値で、入力画像に加えられるノイズの量を制御します。
1.0 に近い値では、多くのバリエーションが可能になりますが、入力と意味的に一致しない画像も生成されます。次の例をご覧ください。

**Input**

![sketch-in](assets/stable-samples/img2img/sketch-mountains-input.jpg)

**Outputs**

![out3](assets/stable-samples/img2img/mountains-3.png)
![out2](assets/stable-samples/img2img/mountains-2.png)

この手順は、例えば、ベースモデルからサンプルをアップスケールする場合にも利用できます。

## コメント

- 拡散モデル用のコードベースは、[OpenAI の ADM コードベース](https://github.com/openai/guided-diffusion)と[https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) を大いに利用しています。オープンソースに感謝します
- トランスフォーマーエンコーダーの実装は、lucidrains さんの[x-transformers](https://github.com/lucidrains/x-transformers) by [lucidrains](https://github.com/lucidrains?tab=repositories)を使用しています。

## BibTeX

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models},
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
