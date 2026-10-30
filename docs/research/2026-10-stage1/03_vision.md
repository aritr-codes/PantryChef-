# 03 Vision: ingredient recognition for PantryChef

Evidence caveat: web tools returned search snippets, not primary pages (Roboflow Universe 403, Kaggle docs empty). Anything not quoted from a source is ESTIMATE or HYPOTHESIS. No measured fridge-clutter benchmark for any approach was found; that gap is itself a finding.

## 0. Vocabulary reality check
MEASURED (repo `data/processed/vocab.json`): 30,481 entries, a list of strings, and they are recipe-parser phrases, not clean ingredients ("cream of chicken soup or cream of mushroom soup", "campbell s condensed cream of broccoli soup", "extra virgin extra virgin olive oil plu additional for drizzling"). No detector or CLIP can target this directly. A curated visual vocabulary (~100-300 items) with a map to canonical vocab is required whatever model is chosen. This is the biggest hidden cost, bigger than model choice.

## 1. Datasets
| Dataset | Classes / images | Annotation | Realism | Licence | Notes |
|---|---|---|---|---|---|
| Roboflow "AICook" (fridge) | 30 of top-50 Recipe1M ingredients; 516 imgs (3,050 augmented) | boxes | real fridge, tiny | MIT (per search snippet; page 403) | [link](https://universe.roboflow.com/objectdetectionlearning-4empo/aicook-lcv4d-zzbgs); best fridge-specific, far too small to generalise |
| Roboflow FridgeFoodDetection, "fridge" | unverified | boxes | mixed | per-dataset, verify | [1](https://universe.roboflow.com/objectdetectionlearning-4empo/fridgefooddetection-cxnl3), [2](https://universe.roboflow.com/michael-ringer/fridge-vzg6p/dataset/); label noise typical |
| LVIS v1 / COCO | LVIS 1,203 cats (many food); COCO ~10 food classes | boxes+masks | everyday scenes | annotations CC BY 4.0, COCO images Flickr mixed (ESTIMATE, verify) | LVIS long-tail, few samples per food |
| Open Images V7 | ~600 box classes incl. many foods | boxes | real web photos | annotations CC BY 4.0, images CC BY 2.0 (ESTIMATE, verify) | best large source of diverse real food boxes |
| FoodSeg103 | 103 ingredient classes, ~7k imgs | masks | cooked-dish photos | Apache-2.0 (MEASURED, [HF card](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103/raw/main/README.md)) | ingredients on a plate, not raw/packaged |
| Food-101 | 101 dishes, 101k | image label | dish photos | research-only terms (ESTIMATE) | dishes, not ingredients; irrelevant to fridge. Recipes5k/Ingredients101 add ingredient lists per dish ([paper](https://arxiv.org/pdf/1707.08816)) |
| Fruits-360 | ~130 fruit/veg classes, ~90k | classification, white background | studio only | MIT (MEASURED, [repo](https://github.com/Horea94/Fruit-Images-Dataset)) | huge domain gap; poor for clutter |
| Freiburg Groceries | 25 classes, 5,000 imgs | classification + 74 cluttered-shelf imgs | real shelves | unclear/"Other" ([paper](https://www.arxiv.org/pdf/1611.05799)) | check before use |
| KTH Grocery Store | ~81 fine classes, 5k imgs | classification | store shots | unverified | |
| SKU-110K | 1 class | dense boxes | retail shelves | academic/non-commercial (ESTIMATE) | localisation only, no semantics |
| Food Recognition Benchmark (AIcrowd) | 498 classes | instance seg | dishes | CC BY-NC (ESTIMATE) | dishes |

Plain answer: no public dataset covers real fridge clutter with enough classes. Closest: Open Images + LVIS (diversity) plus AICook (domain). A closed-set detector trained on these covers ~50-100 common items at best, weak on packaged goods. Weights trained on CC BY / MIT / Apache data are fine to ship; anything CC BY-NC taints a public MIT release. Train only on permissive sources and record provenance.

## 2. Model licensing
- MEASURED ([Ultralytics licensing](https://www.ultralytics.com/license), [docs](https://docs.ultralytics.com/)): YOLOv8/11/26 code and weights are AGPL-3.0, or a paid Enterprise licence. Latest PyPI seen: ultralytics 8.4.x.
- Implication (ESTIMATE, not legal advice): Ultralytics treats weights trained with its package as AGPL. Shipping them in a public MIT repo makes that component AGPL, contradicting "MIT". A hosted server demo triggers AGPL section 13 (offer source to network users). Browser-side ONNX of such weights is distribution to every visitor. Workable only if the whole project goes AGPL or the part is an isolated optional AGPL module; otherwise avoid.

| Model | Code / weights licence | Note |
|---|---|---|
| RF-DETR Nano-Large | Apache-2.0 / Apache-2.0 | XL/2XL under PML 1.0 (MEASURED, [repo](https://github.com/roboflow/rf-detr)); T4 TensorRT fp16: Nano 2.3 ms, Small 3.5 ms; DINOv2 backbone so probably 100+ MB fp32 (ESTIMATE) |
| RT-DETR / D-FINE | Apache-2.0 | heavier for browser |
| YOLOX | Apache-2.0 | YOLOX-Nano under 1M params; older accuracy; best licence-clean browser-size detector (ESTIMATE) |
| YOLO-NAS | Apache code, weights under a restrictive Deci licence (ESTIMATE, verify) | avoid |
| YOLO-World | GPL-3.0 (MEASURED, [README](https://huggingface.co/spaces/stevengrove/YOLO-World/blame/a8dc1acccb3b3677f13a447d2e466dcba37417cb/README.md)) | avoid |
| Grounding DINO | Apache-2.0 (MEASURED, [Roboflow](https://playground.roboflow.com/models/idea-research/grounding-dino)) | large, server only |
| OWLv2 | Apache-2.0 (ESTIMATE, unconfirmed) | server/CPU slow |
| MobileCLIP2 | apple-amlr (MEASURED, [card](https://huggingface.co/apple/MobileCLIP2-B/blob/main/README.md)) | research-style licence; do not ship in MIT product without reading it |
| SigLIP2 | Apache-2.0 (HF ONNX repo tagged apache-2.0, [upshift](https://huggingface.co/upshift/siglip2-base-patch16-384-onnx/discussions)) | OK |
| OpenCLIP ViT-B (LAION) | MIT | OK |

## 3. Approach comparison
| Approach | Real fridge accuracy | Coverage | Size | Latency | Train cost | Vocab mapping |
|---|---|---|---|---|---|---|
| Fine-tuned closed-set detector | No measured fridge benchmark. HYPOTHESIS (E1): 0.5-0.7 per-photo F1 on ~50 classes, poor on packaged/occluded | 30-100 classes | 3-100 MB | 50-300 ms CPU (ESTIMATE) | 2-6 h Kaggle plus weeks of data cleaning | per-class table, easy but small |
| Open-vocab detector (OWLv2, GDINO) | Zero-shot LVIS AP modest (YOLO-World-L 35.4, S 17.6; MEASURED, same README); packaged goods unmeasured | any text | 150-700 MB | seconds; server only | none | prompt = our names |
| Crop classification | needs a localiser; two-stage error compounds | classifier classes | small | adds | cheap | |
| Whole-image multi-label classifier | Hard even in-domain (Recipes5k paper); no fridge data | trained classes | 10-30 MB | under 100 ms | 1-3 h | easy |
| CLIP/SigLIP matching on tiles | No measured fridge multi-label result found. MEASURED ImageNet zero-shot: MobileCLIP2-S0 71.5%, B 79.4% (card). HYPOTHESIS (E2): 3x3 tiles plus full image, precomputed text embeddings | whole curated vocab, extendable with no training | image tower 90-370 MB fp32; q8 about 25-100 MB (ESTIMATE); text tower not needed at runtime | 100-500 ms/tile WASM (ESTIMATE) | none (optional linear-probe calibration) | direct: name to text embedding |

Insight check: the product (user-edited checklist) fits "set of candidates with scores" better than boxes; boxes add nothing the checklist uses. CLIP-style matching gives vocabulary coverage no dataset gives. But no published number for CLIP on cluttered fridge multi-label exists in what I found, so accuracy is unproven. Expected weaknesses (ESTIMATE): whole-image embeddings dilute small items; similar items confused (cream vs yogurt, herbs); scores are relative, so per-class thresholds need tuning.

## 4. Runtime
- Server CPU: simplest, any model, ~1-3 s for tiled ViT-B (ESTIMATE); GitHub Pages is static, so needs a separate host.
- Browser: ONNX Runtime Web (WASM/WebGPU) and Transformers.js run community SigLIP/CLIP ONNX exports (MEASURED existence, [SigLIP2 ONNX](https://huggingface.co/SkyStar-tech/siglip2-large-patch16-384-ONNX)). I found no published low-end-phone latency; all browser latency is ESTIMATE. Keep model plus activations under ~200 MB for mobile browsers (ESTIMATE).
- Quantization: int8 dynamic typically costs 1-3 points on classification, ViT attention can be worse; fp16 safer on WebGPU (ESTIMATE; measure in E2).
- Budget: git blocks files over 100 MB, Pages sites ~1 GB. Host weights on Hugging Face (CDN, CORS), lazy-load on "scan photo" click, cache via Cache API/IndexedDB. Target first load at most 40-90 MB, with explicit size warning. Small Apache/MIT towers under 40 MB barely exist; OpenCLIP B/32 q8 is ~90 MB (ESTIMATE).

## 5. Privacy and UX
On-device inference: photos never leave the device, removing upload storage, GDPR exposure (faces, labels, mail visible in kitchen photos), moderation burden and server cost. Claim it only if CSP `connect-src` is restricted to the model host and verified. Show output as editable checklist: high-confidence pre-ticked, medium shown as unticked "maybe" with the triggering tile thumbnail, nothing auto-submitted, always a type-ahead to add missing items.

## 6. Evaluation
100-150 photos, self-taken plus CC0/CC BY (Wikimedia Commons), mixed lighting, 5-40 visible items. Annotation: per-photo set of visible ingredients in the curated vocabulary (no boxes), double-annotate 30 for agreement. Metrics: per-photo ingredient precision, recall, F1 after vocab mapping; "edits needed" = FP+FN count; recall@K of candidates shown; report produce vs packaged separately. Do not report mAP.

## 7. Recommendation
Primary (HYPOTHESIS, test first): scan-to-checklist without boxes. An Apache/MIT CLIP-family encoder (SigLIP2-B or OpenCLIP B/32, int8/fp16) over tiled crops, in-browser, matched to precomputed text embeddings of a curated ~150-200-item visual vocabulary mapped to canonical names. ESTIMATE per-photo F1 0.4-0.65, better on produce and staples, worse on packaged goods; recall above precision. No training needed; optional Kaggle linear-probe calibration on Open Images crops (~1 h; Kaggle free quota reported ~30 GPU h/week, sessions 9-12 h, unconfirmed by official docs: [summary](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)). Artifact ~60-90 MB on HF CDN plus ~1 MB embeddings.

Fallback: same pipeline server-side with Apache OWLv2/Grounding DINO verification, or a YOLOX/RF-DETR-Nano fine-tune on Open Images+AICook for ~60 classes (Kaggle 3-6 h, 10-50 MB, ESTIMATE F1 0.5-0.7 on those 60 only). Avoid Ultralytics unless the project relicenses to AGPL.

Scope: do not put CV in the v1 critical path. Ship manual/typeahead ingredient entry first; CV as v1.1 "beta: suggestions only" for ~150-200 items, gated on eval set F1 >= 0.5 and <= 5 edits/photo. If E2 fails, do not ship CV.
Experiments: E1 closed-set detector baseline; E2 tiled SigLIP2/OpenCLIP vs eval set (fp32/int8/fp16); E3 per-class threshold calibration.
