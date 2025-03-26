# Unified Models for Vision Language Understanding and Generation


## Benchmark Results

### Understanding


| Model         | Params | POPE | MME-P  | MMB_dev | SEED | VQAv2 | GQA   | MMMU | MM-Vet | TextVQA | MMStar |
| ------------- | ------ | ---- | ------ | ------- | ---- | ----- | ----- | ---- | ------ | ------- | ------ |
| Gemini-Nano-1 | 1.8B   | -    | -      | -       | -    | 62.7  | -     | 26.3 | -      | -       | -      |
| VILA-U        | 7B     | 85.8 | 1401.8 | -       | 59.0 | 79.4  | 60.8  | -    | 33.5   | -       | -      |
| Chameleon     | 7B     | -    | 170.0  | 31.1    | 30.6 | -     | -     | 25.4 | 8.3    | -       | 31.1   |
| Chameleon     | 30B    | -    | 575.3  | 32.5    | 48.5 | -     | -     | 38.8 | -      | -       | 31.8   |
| DreamLLM      | 7B     | -    | -      | 58.2    | -    | 72.9  | -     | -    | 36.6   | -       | -      |
| LaVIT         | 7B     | -    | -      | -       | -    | 66.0  | 46.8  | -    | -      | -       | -      |
| Video-LaVIT   | 7B     | -    | 1551.8 | 67.3    | 64.0 | -     | -     | -    | -      | -       | -      |
| Emu           | 13B    | -    | -      | -       | -    | 52.0  | -     | -    | -      | -       | -      |
| Emu3          | 8B     | -    | -      | 58.5    | 68.2 | -     | -     | 31.6 | -      | -       | -      |
| NExT-GPT      | 13B    | -    | -      | -       | -    | 66.7  | -     | -    | -      | -       | -      |
| Show-o        | 1.3B   | 73.8 | 948.4  | -       | -    | 59.3  | 48.7  | 25.1 | -      | -       | -      |
| Janus         | 1.3B   | 87.0 | 1338.0 | 69.4    | 63.7 | 77.3  | 59.1  | 30.5 | 34.3   | -       | -      |
| JanusFlow     | 1.3B   | 88.0 | 1333.1 | 74.9    | 70.5 | 79.8  | 60.3  | 29.3 | 30.9   | -       | -      |
| Orthus        | 7B     | 79.6 | 1265.8 | -       | -    | 63.2  | 52.8  | 28.2 | -      | -       | -      |
| Liquid†       | 7B     | 81.1 | 1119.3 | -       | -    | 71.3* | 58.4* | -    | -      | 42.4    | -      |
| Unified-IO2   | 6.8B   | -    | -      | 71.5    | -    | -     | -     | 86.2 | -      | -       | 61.8   |
| SEEDLLaMA     | 7B     | -    | -      | 45.8    | 51.5 | -     | -     | -    | -      | -       | 31.7   |
| MUSE-VL       | 7B     | -    | 1480.9 | 72.1    | 70.0 | -     | -     | 42.3 | -      | -       | 48.3   |
| MUSE-VL       | 32B    | -    | 1581.6 | 81.8    | 71.0 | -     | -     | 50.1 | -      | -       | 56.7   |


### MJHQ-30K


| Method          | Resolution | Params | #Images | FID   |
| --------------- | ---------- | ------ | ------- | ----- |
| SD-XL           |            | -      | 2000M   | 9.55  |
| PixArt          |            | -      | 25M     | 6.14  |
| Playground v2.5 |            | -      | -       | 4.48  |
| LWM             |            | 7B     | -       | 17.77 |
| VILA-U          | 256        | 7B     | 15M     | 12.81 |
| VILA-U          | 384        | 7B     | 15M     | 7.69  |
| Show-o          |            | 1.3B   | 36M     | 15.18 |
| Janus           |            | 1.3B   | -       | 10.10 |
| JanusFlow       |            | 1.3B   | -       | 9.51  |
| Liquid          | 512        | 7B     | 30M     | 5.47  |

### GenEval Bench

| Model                | Params | Res. | Single Obj. | Two Obj. | Count. | Colors | Position | Color Attri. | Overall↑ |
| -------------------- | ------ | ---- | ----------- | -------- | ------ | ------ | -------- | ------------ | -------- |
| **Generation Model** |
| LlamaGen             | 0.8B   | -    | 0.71        | 0.34     | 0.21   | 0.58   | 0.07     | 0.04         | 0.32     |
| LDM                  | 1.4B   | -    | 0.92        | 0.29     | 0.23   | 0.70   | 0.02     | 0.05         | 0.37     |
| SDv1.5               | 0.9B   | -    | 0.97        | 0.38     | 0.35   | 0.76   | 0.04     | 0.06         | 0.43     |
| PixArt-α             | 0.6B   | -    | 0.98        | 0.50     | 0.44   | 0.80   | 0.08     | 0.07         | 0.48     |
| SDv2.1               | 0.9B   | -    | 0.98        | 0.51     | 0.44   | 0.85   | 0.07     | 0.17         | 0.50     |
| DALL-E 2             | 6.5B   | -    | 0.94        | 0.66     | 0.49   | 0.77   | 0.10     | 0.19         | 0.52     |
| Emu3-Gen             | 8B     | -    | 0.98        | 0.71     | 0.34   | 0.81   | 0.17     | 0.21         | 0.54     |
| SDXL                 | 2.6B   | -    | 0.98        | 0.74     | 0.39   | 0.85   | 0.15     | 0.23         | 0.55     |
| IF-XL                | 4.3B   | -    | 0.97        | 0.74     | 0.66   | 0.81   | 0.13     | 0.35         | 0.61     |
| DALL-E 3             | -      | -    | 0.96        | 0.87     | 0.47   | 0.83   | 0.43     | 0.45         | 0.67     |
| **Unified Model**    |
| Chameleon            | 34B    | -    | -           | -        | -      | -      | -        | -            | 0.39     |
| LWM                  | 7B     | -    | 0.93        | 0.41     | 0.46   | 0.79   | 0.09     | 0.15         | 0.47     |
| SEED-X               | 17B    | -    | 0.97        | 0.58     | 0.26   | 0.80   | 0.19     | 0.14         | 0.49     |
| Show-o               | 1.3B   | -    | 0.95        | 0.52     | 0.49   | 0.82   | 0.11     | 0.28         | 0.53     |
| Janus                | 1.3B   | -    | 0.97        | 0.68     | 0.30   | 0.84   | 0.46     | 0.42         | 0.61     |
| JanusFlow            | 1.3B   | -    | 0.97        | 0.59     | 0.45   | 0.83   | 0.53     | 0.42         | 0.63     |
| Orthus               | 7B     | 512  | 0.99        | 0.75     | 0.26   | 0.84   | 0.28     | 0.38         | 0.58     |


## External Image Generator

### Discrete Condition

| Publication Date | Method Abbreviation | Full Title                                                                           | arXiv Link                                | Code Repository                                    |
| ---------------- | ------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------- | -------------------------------------------------- |
| 23/10            | SEED                | Making Llama See and Draw With Seed Tokenizer                                        | [arXiv](https://arxiv.org/abs/2310.01218) | [GitHub](https://github.com/AILab-CVC/SEED/)       |
| 24/06            | LAViT               | Unified Language-Vision Pretraining in Llm With Dynamic Discrete Visual Tokenization | [arXiv](https://arxiv.org/abs/2406.09399) | [GitHub](https://github.com/jy0205/LaVIT)          |
| 24/02            | AnyGPT              | Anygpt: Unified Multimodal Llm With Discrete Sequence Modeling                       | [arXiv](https://arxiv.org/abs/2402.12226) | [GitHub](https://github.com/OpenMOSS/AnyGPT)       |
| 24/09            | MIO                 | Mio: A Foundation Model on Multimodal Tokens                                         | [arXiv](https://arxiv.org/abs/2409.17692) | [GitHub](https://github.com/dvlab-research/MGM)    |
| 24/12            | illume              | Illume: Illuminating Your Llms to See, Draw, and Self-Enhance                        | [arXiv](https://arxiv.org/abs/2412.06673) | [GitHub](https://github.com/illume-project/illume) |

### Continuous Condition

| Publication Date | Method Abbreviation | Full Title                                                                                        | arXiv Link                                            | Code Repository                                         |
| ---------------- | ------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------- |
| 23/06            | Emu                 | Emu: Generative Pretraining in Multimodality                                                      | [arXiv](https://arxiv.org/abs/2306.11838)             | [GitHub](https://github.com/baaivision/Emu)             |
| 23/09            | NExT-GPT            | Next-Gpt                                                                                          | [arXiv](https://arxiv.org/abs/2309.05519)             | [GitHub](https://github.com/NExT-GPT/NExT-GPT)          |
| 24/02            | EasyGen             | Easygen: Easing Multimodal Generation With Bidiffuser and Llms                                    | [arXiv](https://aclanthology.org/2024.findings-1.12/) |                                                         |
| 23/12            | VL-GPT              | Vl-Gpt: A Generative Pre-Trained Transformer for Vision and Language Understanding and Generation | [arXiv](https://arxiv.org/abs/2312.09251)             | [GitHub](https://github.com/AILab-CVC/VL-GPT)           |
| 24/03            | CoDi-2              | Codi-2: In-Context Interleaved and Interactive Any-to-Any Generation                              | [arXiv](https://arxiv.org/abs/2403.06764)             |                                                         |
| 24/06            | Emu2                | Generative Multimodal Models Are In-Context Learners                                              | [arXiv](https://arxiv.org/abs/2406.10797)             | [GitHub](https://github.com/baaivision/Emu)             |
| 24/01            | MM-Interleaved      | Mm-Interleaved: Interleaved Image-Text Generative Modeling Via Multi-Modal Feature Synchronizer   | [arXiv](https://arxiv.org/abs/2401.10208)             | [GitHub](https://github.com/OpenGVLab/MM-Interleaved)   |
| 24/05            | DEEM                | Deem: Diffusion Models Serve as the Eyes of Large Language Models for Image Perception            | [arXiv](https://arxiv.org/abs/2405.15232)             |                                                         |
| 24/05            | X-VILA              | X-Vila: Cross-Modality Alignment for Large Language Model                                         | [arXiv](https://arxiv.org/abs/2405.19335)             | [GitHub](https://github.com/facebookresearch/chameleon) |
| 24/11            | Spider              | Spider: Any-to-Many Multimodal Llm                                                                | [arXiv](https://arxiv.org/abs/2411.09439)             | [GitHub](https://github.com/deepseek-ai/Janus)          |
| 24/12            | MetaMorph           | Metamorph: Multimodal Understanding and Generation Via Instruction Tuning                         | [arXiv](https://arxiv.org/abs/2412.14164)             | [GitHub](https://github.com/deepseek-ai/Janus)          |
| 24/04            | DreamLLM            | Dreamllm: Synergistic Multimodal Comprehension and Creation                                       | [arXiv](https://arxiv.org/abs/2404.18202)             | [GitHub](https://github.com/RunpeiDong/DreamLLM)        |
| 23/10            | MiniGPT-5           | Minigpt-5: Interleaved Vision-and-Language Generation Via Generative Vokens                       | [arXiv](https://arxiv.org/abs/2310.02239)             |                                                         |
| 24/04            | SEED-X              | Seed-X: Multimodal Models With Unified Multi-Granularity Comprehension and Generation             | [arXiv](https://arxiv.org/abs/2404.14396)             | [GitHub](https://github.com/AILab-CVC/SEED-X)           |

## Discrete Image Modelling

### VQGAN Encoder

| Publication Date | Method Abbreviation | Full Title                                                                                                | arXiv Link                                   | Code Repository                                                |
| ---------------- | ------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------- | -------------------------------------------------------------- |
| 21/06            | OFA                 | Ofa: Unifying Multimodal Pretrained Models                                                                | [arXiv](https://arxiv.org/abs/2106.08254)    | [GitHub](https://github.com/facebookresearch/segment_anything) |
| 22/04            | Unified-IO          | Unified-Io: A Unified Model for Vision, Language, and Multi-Modal Tasks                                   | [arXiv](https://arxiv.org/abs/2204.05772)    | [GitHub](https://github.com/allenai/unified-io-2)              |
| 23/11            | Teal                | Teal: Tokenize and Embed All for Multi-Modal Large Language Models                                        | [arXiv](https://arxiv.org/abs/2311.04589)    |                                                                |
| 24/02            | LWM                 | World Model on Million-Length Video and Language With Ringattention                                       | [arXiv](https://arxiv.org/abs/2402.08268)    | [GitHub](https://github.com/LargeWorldModel/LWM)               |
| 24/06            | 4M-21               | 4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities                                        | [arXiv](https://arxiv.org/abs/2406.09406)    |                                                                |
| 24/05            | Chameleon           | Chameleon: Mixed-Modal Early-Fusion Foundation Models                                                     | [arXiv](https://arxiv.org/abs/2405.09818)    | [GitHub](https://github.com/facebookresearch/chameleon)        |
| 24/08            | ANOLE               | Anole: An Open, Autoregressive, Native Large Multimodal Models for Interleaved Image-Text Generation      | [arXiv](https://huggingface.co/blog/idefics) | [GitHub](https://github.com/GAIR-NLP/anole)                    |
| 24/08            | Show-o              | Show-o: One single transformer to unify multimodal understanding and generation                           | [arXiv](https://arxiv.org/abs/2408.12528)    |                                                                |
| 24/09            | Emu3                | Emu3: Next-Token Prediction is All You Need                                                               | [arXiv](https://arxiv.org/abs/2409.18869)    |                                                                |
| 24/12            | Liquid              | Liquid: Language Models are Scalable Multi-modal Generators                                               | [arXiv](https://arxiv.org/abs/2412.04332)    |                                                                |
| 24/12            | SynerGen-VL         | SynerGen-VL: Towards Synergistic Image Understanding and Generation with Vision Experts and Token Folding | [arXiv](https://arxiv.org/abs/2412.09604)    |                                                                |

### Semantic Encoder

| Publication Date | Method Abbreviation | Full Title                                                                         | arXiv Link                                | Code Repository |
| ---------------- | ------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------- | --------------- |
| 24/04            | LIBRA               | Libra: Building Decoupled Vision System on Large Language Models                   | [arXiv](https://arxiv.org/abs/2404.16821) |                 |
| 24/09            | VILA-U              | VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation | [arXiv](https://arxiv.org/abs/2409.04429) |                 |
| 24/11            | MUSE-VL             | MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding                   | [arXiv](https://arxiv.org/abs/2411.17762) |                 |
| 24/12            | TokenFlow           | TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation     | [arXiv](https://arxiv.org/abs/2412.03069) |                 |

### Decoupled Encoder

| Publication Date | Method Abbreviation | Full Title                                                                                   | arXiv Link                                | Code Repository                                   |
| ---------------- | ------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------- |
| 24/06            | Unified-IO 2        | Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision Language Audio and Action | [arXiv](https://arxiv.org/abs/2406.08394) | [GitHub](https://github.com/allenai/unified-io-2) |
| 24/05            | Morph-Tokens        | Auto-Encoding Morph-Tokens for Multimodal LLM                                                | [arXiv](https://arxiv.org/abs/2405.01926) |                                                   |
| 24/10            | Janus               | Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation        | [arxiv](https://arxiv.org/abs/2410.13848) | [GitHub](https://github.com/deepseek-ai/Janus)    |

## Continuous Image Modelling

### Feature Prediction

| Publication Date | Method Abbreviation | Full Title                                                                                                   | arXiv Link                                | Code Repository                                    |
| ---------------- | ------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------- | -------------------------------------------------- |
| 24/08            | Transfusion         | Transfusion: Predict The Next Token and Diffuse Images With One Multi-Modal Model                            | [arXiv](https://arxiv.org/abs/2408.11039) |                                                    |
| 24/09            | MonoFormer          | MonoFormer: One transformer for both diffusion and autoregression                                            | [arXiv](https://arxiv.org/abs/2409.16280) | [GitHub](https://github.com/MonoFormer/MonoFormer) |
| 24/11            | JanusFlow           | JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation | [arXiv](https://arxiv.org/abs/2411.07975) | [GitHub](https://github.com/deepseek-ai/Janus)     |
| 24/11            | JetFormer           | JetFormer: An Autoregressive Generative Model of Raw Images and Text                                         | [arXiv](https://arxiv.org/abs/2411.19722) |                                                    |
| 24/12            | CausalFusion        | Causal Diffusion Transformers for Generative Modeling                                                        | [arXiv](https://arxiv.org/abs/2412.12095) |                                                    |
| 24/12            | LLaMAFusion         | LlamaFusion: Adapting Pretrained Language Models for Multimodal Generation                                   | [arXiv](https://arxiv.org/abs/2412.15188) |                                                    |

### Condition Prediction

| Publication Date | Method Abbreviation | Full Title                                                                            | arXiv Link                                | Code Repository |
| ---------------- | ------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------- | --------------- |
| 24/10            | MMAR                | Towards Lossless Multi-Modal Auto-Regressive Prababilistic Modeling                   | [arXiv](https://arxiv.org/abs/2410.10798) |                 |
| 24/12            | Orthus              | Orthus: Autoregressive Interleaved Image-Text Generation with Modality-Specific Heads | [arXiv](https://arxiv.org/abs/2412.00127) |                 |
| 24/12            | LatentLM            | Multimodal Latent Language Modeling with Next-Token Diffusion                         | [arxiv](https://arxiv.org/abs/2412.08635) |                 |


## Diffusion Models

| Publication Date | Method Abbreviation | Full Title                                                                            | arXiv Link                                | Code Repository                                     |
| ---------------- | ------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------- | --------------------------------------------------- |
| 2023-03-13       | UniDiffuser         | One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale              | [arXiv](https://arxiv.org/abs/2303.06555) |                                                     |
| 2023-05-20       | CoDi                | Any-to-Any Generation via Composable Diffusion                                        | [arXiv](https://arxiv.org/abs/2305.11846) | [Project Page](https://codi-gen.github.io/)         |
| 2023-06-01       | UniDiff             | UniDiff: Advancing Vision-Language Models with Generative and Discriminative Learning | [arXiv](https://arxiv.org/abs/2306.00813) |                                                     |
| 2024-12-02       | OmniFlow            | OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows                      | [arXiv](https://arxiv.org/abs/2412.01169) | [GitHub](https://github.com/jacklishufan/OmniFlows) |
| 2024-12-31       | Dual Diffusion      | Dual Diffusion for Unified Image Generation and Understanding                         | [arXiv](https://arxiv.org/abs/2501.00289) |                                                     |

