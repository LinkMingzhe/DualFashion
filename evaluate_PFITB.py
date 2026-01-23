import os
import ast
import json
import pathlib
import re
import glob
from collections import defaultdict
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import open_clip

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x, *args, **kwargs):
        return x

import eval_utils

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', type=int, default=5,
                    help='Batch size to use')
parser.add_argument('--num_workers', type=int, default=4,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--gpu', type=int, default=None,
                    help='gpu id to use')
parser.add_argument('--device', type=str, default="cuda:0",
                    help='gpu id to use')
parser.add_argument('--dims4fid', type=int, default=2048,
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('--num_samples', type=int, default=5,
                    help='Number of samples per text-image pair.')
parser.add_argument('--data_path', type=str, default='../dataset/ifashion/semantic_category')
parser.add_argument('--pretrained_evaluator_ckpt', type=str, default='eval/compatibility_evaluator/ifashion-ckpt/fashion_evaluator.pth')
parser.add_argument('--dataset', type=str, default="ifashion")
parser.add_argument('--output_dir', type=str, default="output/QualitativeResults")
parser.add_argument('--eval_version', type=str, default="stage2_imgloss_inference_PFITB_lrem_878")
parser.add_argument('--task', type=str, default="FITB")
parser.add_argument('--num_classes', type=int, default=50)
parser.add_argument('--sim_func', type=str, default="cosine")
parser.add_argument('--lpips_net', type=str, default="vgg")
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--log_name', type=str, default="log") 
parser.add_argument('--mode', type=str, default="test")
parser.add_argument('--API_KEY', type=str, default='AIzaSyAt67OjtwLl2U4f3ySZyca0bYKmEA_bBjw')

SPECIAL_CATES = ["shoes", "pants", "sneakers", "boots", "earrings", "slippers", "sandals"]
TEXT_ATTRIBUTE_KEYS = ["Color", "Material", "Design features", "Clothing Fashion Style"]
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

GEMINI_TEXT_COMP_MODEL_ENV = "GEMINI_TEXT_COMP_MODEL"
GEMINI_TEXT_COMP_DEFAULT_MODEL = "gemini-2.5-flash"
GEMINI_TEXT_COMP_RUBRIC = """
Use the following 1-10 rubric (higher is better):
1 - Extremely conflicting: breaks the outfit or has no relevance.
2 - Very poor fit: major color or theme clashes.
3 - Strong mismatch: key materials, silhouette, or style disagree.
4 - Noticeable mismatch: difficult to integrate with the outfit.
5 - Barely acceptable: conflicts remain obvious.
6 - Mostly acceptable: wearable but with one or two clear issues.
7 - Good fit: generally coherent with minor differences.
8 - Very good fit: most details complement the outfit.
9 - Excellent fit: highly coherent with only tiny imperfections.
10 - Perfect fit: completely aligned in color, material, design, and style.
""".strip()

class FashionEvalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class FashionRetrievalDataset(Dataset):
    def __init__(self, gen_images, candidates):
        self.gen_images = gen_images
        self.candidates = candidates

    def __len__(self):
        return len(self.gen_images)
    
    def __getitem__(self, index):
        return self.gen_images[index], self.candidates[index]

class FashionPersonalSimDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data["gen"])
    
    def __getitem__(self, index):
        gen = self.data["gen"][index]
        hist = self.data["hist"][index]

        return gen, hist

def cate_trans(cid, id_cate_dict):
    category = id_cate_dict[cid]
    # prompt = "A photo of a " + category + ", on white background"
    prompt = "A photo of a " + category + "."
    return prompt

def parse_attribute_fields(text):
    cleaned = text.strip()
    if not cleaned:
        return {}
    if cleaned.startswith("'") and cleaned.endswith("'"):
        cleaned = cleaned[1:-1]
    if not cleaned.startswith("{"):
        cleaned = "{" + cleaned
    if not cleaned.endswith("}"):
        cleaned = cleaned + "}"
    cleaned = cleaned.replace("'", '"')
    cleaned = re.sub(r",\s*}", "}", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        matches = re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', cleaned)
        data = {key: value for key, value in matches}
    return {key: value for key, value in data.items() if key in TEXT_ATTRIBUTE_KEYS and isinstance(value, str) and value.strip()}

def detect_mime_type(path):
    lower = path.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    if lower.endswith(".webp"):
        return "image/webp"
    return "application/octet-stream"

def load_image_for_gemini(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as handle:
            data = handle.read()
        return {"mime_type": detect_mime_type(path), "data": data}
    except Exception:
        return None

def build_text_compatibility_prompt(attribute, value):
    return (
        "You are a professional stylist. Judge whether the proposed item works with the reference outfit images.\n"
        f"{GEMINI_TEXT_COMP_RUBRIC}\n"
        "Task: Given the three incomplete-outfit reference images, rate how well this attribute fits with them.\n"
        f"Attribute: {attribute}\n"
        f"Description: {value}\n"
        "Respond with a single integer score between 1 and 10, with no extra text."
    )

def initialize_gemini_model(args):
    if genai is None:
        print("google.generativeai is not installed; skip Gemini-based Text_Compatibility evaluation.")
        return None
    api_key = args.API_KEY
    if not api_key:
        print("Environment variable GOOGLE_API_KEY is not set; skip Gemini-based Text_Compatibility evaluation.")
        return None
    model_name = os.environ.get(GEMINI_TEXT_COMP_MODEL_ENV, GEMINI_TEXT_COMP_DEFAULT_MODEL)
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name)
    except Exception as exc:
        print(f"Failed to initialize Gemini model '{model_name}': {exc}")
        return None

def score_attribute_with_gemini(model, images, attribute, value):
    if model is None or not images or not value:
        return None
    prompt = build_text_compatibility_prompt(attribute, value)
    parts = [{"text": prompt}]
    parts.extend(images)
    try:
        response = model.generate_content(parts)
        candidate = getattr(response, "text", "") or ""
        candidate = candidate.strip()
        match = re.search(r"\b(10|[1-9])\b", candidate)
        if match:
            score = int(match.group(1))
            return float(score)
        print(f"Unexpected Gemini response format for attribute {attribute}: '{candidate}'")
    except Exception as exc:
        print(f"Gemini evaluation failed for attribute {attribute}: {exc}")
    return None

def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    if args.dataset == "ifashion":
        args.data_path = '../dataset/ifashion'
        args.pretrained_evaluator_ckpt = 'eval/compatibility_evaluator/ifashion-ckpt/ifashion_evaluator.pth'
        # args.output_dir = '/output/path/ifashion/xxx'
    elif args.dataset == "polyvore":
        args.data_path = '../dataset/polyvore'
        args.pretrained_evaluator_ckpt = 'eval/compatibility_evaluator/polyvore-ckpt/polyvore_evaluator.pth'
        # args.output_dir = '/output/path/polyvore/xxx'
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}.")

    eval_path = os.path.join(args.output_dir, args.eval_version)

    ckpts = []
    if os.path.isdir(eval_path):
        for entry in os.listdir(eval_path):
            if entry.endswith(".npy"):
                continue
            ckpts.append(entry)
    ckpts.sort()

    if args.gpu is None:
        device = torch.device(args.device if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(f"cuda:{args.gpu}")
    print(f"Evaluate on device {device}")

    num_workers = args.num_workers

    id_cate_dict = np.load(os.path.join(args.data_path, "id_cate_dict.npy"), allow_pickle=True).item()
    cid_to_label = np.load('eval/finetuned-inception/cid_to_label.npy', allow_pickle=True).item()  # map cid to inception predicted label
    cnn_features_clip = np.load(os.path.join(args.data_path, "cnn_features_clip.npy"), allow_pickle=True)
    cnn_features_clip = torch.tensor(cnn_features_clip)

    if args.mode == "valid":
        history = np.load(os.path.join(args.data_path, "processed", "valid_history_clipembs.npy"), allow_pickle=True).item()
        fitb_retrieval_candidates = np.load(os.path.join(args.data_path, "fitb_valid_retrieval_candidates.npy"), allow_pickle=True).item()
        fitb_dict = np.load(os.path.join(args.data_path, "fitb_valid_dict.npy"), allow_pickle=True).item()
    else:
        history = np.load(os.path.join(args.data_path, "processed", "test_history_clipembs.npy"), allow_pickle=True).item()
        fitb_retrieval_candidates = np.load(os.path.join(args.data_path, "fitb_test_retrieval_candidates.npy"), allow_pickle=True).item()
        fitb_dict = np.load(os.path.join(args.data_path, "fitb_test_dict.npy"), allow_pickle=True).item()

    eval_save_path = os.path.join(eval_path, f"eval_results.npy")
    print(f"save_path:{eval_save_path}")
    if not os.path.exists(eval_save_path):
        all_eval_metrics = {}
    else:
        all_eval_metrics = np.load(eval_save_path, allow_pickle=True).item()

    gemini_model = initialize_gemini_model(args)

    for ckpt in ckpts:
        # if ckpt not in all_eval_metrics:
        #     all_eval_metrics[ckpt] = {}
        # else:
        #     print(f"checkpoint-{ckpt} has already been evaluated. Skip.")
        #     continue
        all_eval_metrics[ckpt] = {}

        gen_ckpt_dir = os.path.join(eval_path, ckpt)
        gen_filename = "gen.npy"
        grd_filename = "grd.npy"

        gen_candidate_paths = []
        grd_candidate_paths = []
        gen_candidate_paths.append(os.path.join(gen_ckpt_dir, gen_filename))
        grd_candidate_paths.append(os.path.join(gen_ckpt_dir, grd_filename))

        gen_path = next((p for p in gen_candidate_paths if os.path.exists(p)), None)
        if gen_path is None:
            raise FileNotFoundError(f"Cannot find generated data file for ckpt {ckpt}: tried {gen_candidate_paths}")

        grd_path = next((p for p in grd_candidate_paths if os.path.exists(p)), None)
        if grd_path is None:
            raise FileNotFoundError(f"Cannot find ground truth data file for ckpt {ckpt}: tried {grd_candidate_paths}")

        gen_data = np.load(gen_path, allow_pickle=True).item()
        grd_data = np.load(grd_path, allow_pickle=True).item()

        trans = transforms.ToTensor()
        resize = transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR)
        _, _, img_trans = open_clip.create_model_and_transforms('ViT-H-14', pretrained="laion2b-s32b-b79K")
        gen4eval = []
        gen4clip = []
        gen4lpips = []
        gen_image_paths = []

        grd4eval = []
        grd4clip = []
        grd4lpips = []

        gen_lpips_groups = defaultdict(list)
        txt4eval = []
        gen_cates = []
        for uid in tqdm(gen_data):
            for oid in gen_data[uid]:
                for img_path in gen_data[uid][oid]["image_paths"]:
                    im = Image.open(img_path)
                    gen4eval.append(trans(im))
                    gen4clip.append(img_trans(im))
                    gen4lpips.append(eval_utils.im2tensor_lpips(im))
                    gen_image_paths.append(img_path)

                for cate in gen_data[uid][oid]["cates"]:
                    gen_cates.append(cid_to_label[cate.item()])
                    txt4eval.append(cate_trans(cate.item(), id_cate_dict))
        
                for img_path in grd_data[uid][oid]["image_paths"]:
                    im = Image.open(img_path)
                    if args.dataset == "polyvore":
                        im = resize(im)  # 291 --> 512
                    grd4eval.append(trans(im))
                    grd4clip.append(img_trans(im))
                    grd4lpips.append(eval_utils.im2tensor_lpips(im))

        # -------------------------------------------------------------- #
        #                    Calculating IS-acc & IS                     #
        # -------------------------------------------------------------- #
        gen_dataset = FashionEvalDataset(gen4eval)
        grd_dataset = FashionEvalDataset(grd4eval)
        txt_dataset = FashionEvalDataset(txt4eval)
        cate_dataset = FashionEvalDataset(gen_cates)
        
        # print("Calculating FID Value...")
        # fid_value = eval_utils.calculate_fid_given_data(
        #     gen_dataset,
        #     grd_dataset,
        #     batch_size=args.batch_size,
        #     dims=args.dims4fid,
        #     device=device,
        #     num_workers=num_workers
        # )
        # torch.cuda.empty_cache()

        # all_eval_metrics[ckpt]["FID"] = fid_value
        # np.save(eval_save_path, np.array(all_eval_metrics))

        print("Calculating Inception Score...")
        is_acc, is_entropy, _, is_score, _ = eval_utils.calculate_inception_score_given_data(
            gen_dataset,
            cate_dataset,
            model_path='./eval/finetuned-inception/Inception-finetune-epoch300',
            num_classes=args.num_classes,
            batch_size=args.batch_size, 
            device=device,
            num_workers=num_workers
        )
        torch.cuda.empty_cache()

        all_eval_metrics[ckpt]["IS-Acc"] = is_acc
        all_eval_metrics[ckpt]["IS"] = is_score
        all_eval_metrics[ckpt]["IS-Entopy"] = is_entropy
        np.save(eval_save_path, np.array(all_eval_metrics))

        del gen4eval
        del grd4eval
        del gen_cates

        # -------------------------------------------------------------- #
        #          Calculating CLIP score and CLIP image score           #
        # -------------------------------------------------------------- #
        # gen_dataset_clip = FashionEvalDataset(gen4clip)
        # grd_dataset_clip = FashionEvalDataset(grd4clip)

        # print("Calculating CLIP Score...")
        # clip_score = eval_utils.calculate_clip_score_given_data(
        #     gen_dataset_clip,
        #     txt_dataset,
        #     batch_size=args.batch_size,
        #     device=device,
        #     num_workers=num_workers
        # )
        # torch.cuda.empty_cache()

        # all_eval_metrics[ckpt]["CLIP score"] = clip_score
        # np.save(eval_save_path, np.array(all_eval_metrics))

        # print("Calculating Grd CLIP Score...")
        # grd_clip_score = eval_utils.calculate_clip_score_given_data(
        #     grd_dataset_clip,
        #     txt_dataset,
        #     batch_size=args.batch_size,
        #     device=device,
        #     num_workers=num_workers
        # )
        # torch.cuda.empty_cache()

        # all_eval_metrics[ckpt]["Grd CLIP score"] = grd_clip_score
        # np.save(eval_save_path, np.array(all_eval_metrics))

        # del txt4eval

        # -------------------------------------------------------------- #
        #               Calculating CLIP Retrieval Acc                   #
        # -------------------------------------------------------------- #
        # print("Calculating CLIP Retrieval Accuracy...")
        # all_candidates = []
        # for uid in gen_data:
        #     for oid in gen_data[uid]:
        #         all_candidates.append(torch.tensor(fitb_retrieval_candidates[uid][oid]))
        # assert len(gen4clip) == len(all_candidates)

        # gen_retrieval_dataset_clip = FashionRetrievalDataset(gen4clip, all_candidates)
        # clip_acc = eval_utils.calculate_clip_retrieval_acc_given_data(
        #     gen_retrieval_dataset_clip,
        #     cnn_features_clip,
        #     batch_size=args.batch_size,
        #     device=device,
        #     num_workers=num_workers,
        #     similarity_func=args.sim_func
        # )
        # torch.cuda.empty_cache()

        # all_eval_metrics[ckpt]["CLIP accuracy"] = clip_acc
        # np.save(eval_save_path, np.array(all_eval_metrics))

        # print("Calculating CLIP Image Score...")
        # clip_img_score = eval_utils.calculate_clip_img_score_given_data(
        #     gen_dataset_clip,
        #     grd_dataset_clip,
        #     batch_size=args.batch_size,
        #     device=device,
        #     num_workers=num_workers,
        #     similarity_func=args.sim_func
        # )
        # torch.cuda.empty_cache()

        # all_eval_metrics[ckpt]["CLIP Image score"] = clip_img_score
        # np.save(eval_save_path, np.array(all_eval_metrics))
        
        # del gen4clip
        # del grd4clip

        # -------------------------------------------------------------- #
        #                       Calculating LPIPS                        #
        # -------------------------------------------------------------- #
        print("Calculating LPIP Score...")
        pairwise_groups = []
        for uid in tqdm(gen_data):
            for oid in gen_data[uid]:
                tensors = []
                for img_path in gen_data[uid][oid]["image_paths"]:
                    with Image.open(img_path) as im:
                        tensors.append(eval_utils.im2tensor_lpips(im))
                if len(tensors) >= 2:
                    pairwise_groups.append(tensors)

        lpip_score = eval_utils.calculate_pairwise_lpips_given_groups(
            pairwise_groups,
            device=device,
            use_net=args.lpips_net
        )
        torch.cuda.empty_cache()

        all_eval_metrics[ckpt]["LPIP score"] = lpip_score
        np.save(eval_save_path, np.array(all_eval_metrics))
        gen_lpips_groups.clear()

        # -------------------------------------------------------------- #
        #                  Evaluating Personalization                    #
        # -------------------------------------------------------------- #
        
        print("Evaluating Personalization of similarity...")
        # The similarity between history and generated images
        gen4personal_sim = {}
        gen4personal_sim["gen"] = []
        gen4personal_sim["hist"] = []
        for uid in tqdm(gen_data):
            for oid in gen_data[uid]:
                for i,img_path in enumerate(gen_data[uid][oid]["image_paths"]):
                    cate = gen_data[uid][oid]["cates"][i].item()
                    try:
                        gen4personal_sim["hist"].append(history[uid][cate])
                        im = Image.open(img_path)
                        gen4personal_sim["gen"].append(img_trans(im))
                    except:
                        continue
                        gen4personal_sim["hist"].append(history['null'])
        
        gen_dataset_personal_sim = FashionPersonalSimDataset(gen4personal_sim)
        personal_sim_score = eval_utils.evaluate_personalization_given_data_sim(
            gen4eval=gen_dataset_personal_sim,
            batch_size=args.batch_size,
            device=device,
            num_workers=num_workers,
            similarity_func=args.sim_func
        )
        torch.cuda.empty_cache()

        all_eval_metrics[ckpt]["Personal Sim"] = personal_sim_score
        np.save(eval_save_path, np.array(all_eval_metrics))

        del gen4personal_sim

        # -------------------------------------------------------------- #
        #                   Evaluating Compatibility                     #
        # -------------------------------------------------------------- #
        print("Evaluating Compatibility...")
        outfits = []
        gen_imgs = []
        gen_idx = 0
        for uid in tqdm(gen_data):
            for oid in gen_data[uid]:
                outfit = fitb_dict[uid][oid]  # 0 as blank to be filled
                img_paths = gen_data[uid][oid]["image_paths"]
                if len(img_paths) == 0:
                    continue

                # Create one evaluation outfit per generated sample so that the
                # negative id maps exactly to the corresponding entry in gen_imgs.
                for img_path in img_paths:
                    new_outfit = []
                    for iid in outfit:
                        if iid == 0:
                            new_outfit.append(-gen_idx)
                        else:
                            new_outfit.append(iid)
                    outfits.append(new_outfit)

                    with Image.open(img_path) as im:
                        gen_imgs.append(img_trans(im))

                    gen_idx += 1

        outfits = torch.tensor(outfits)
        outfit_dataset = FashionEvalDataset(outfits)

        grd_outfits = []
        for uid in grd_data:
            for oid in grd_data[uid]:
                outfit = grd_data[uid][oid]["outfits"]
                grd_outfits.append(outfit)
        grd_outfits = torch.tensor(grd_outfits)
        grd_outfit_dataset = FashionEvalDataset(grd_outfits)

        cnn_feat_path = os.path.join(args.data_path, "cnn_features_clip.npy")
        cnn_feat_gen_path = os.path.join(eval_path, f"{args.task}-checkpoint-{ckpt}-cnnfeat.npy")
        compatibility_score, grd_compatibility_score = eval_utils.evaluate_compatibility_given_data(
            outfit_dataset,
            grd_outfit_dataset,
            gen_imgs,
            cnn_feat_path,
            cnn_feat_gen_path,
            args.pretrained_evaluator_ckpt,
            batch_size=args.batch_size,
            device=device,
            num_workers=num_workers
        )
        torch.cuda.empty_cache()

        all_eval_metrics[ckpt]["Compatibility"] = compatibility_score
        all_eval_metrics[ckpt]["Grd Compatibility"] = grd_compatibility_score
        np.save(eval_save_path, np.array(all_eval_metrics))

        # -------------------------------------------------------------- #
        #                   Evaluating Text_Align                        #
        # -------------------------------------------------------------- #
        print("Evaluating Text_Align...")
        text_align_scores = {key: float("nan") for key in TEXT_ATTRIBUTE_KEYS}
        if len(gen4clip) > 0:
            path_to_index = {path: idx for idx, path in enumerate(gen_image_paths)}
            attribute_texts = {key: [] for key in TEXT_ATTRIBUTE_KEYS}
            attribute_indices = {key: [] for key in TEXT_ATTRIBUTE_KEYS}

            for uid in tqdm(gen_data):
                for oid in gen_data[uid]:
                    sample_dir = os.path.join(gen_ckpt_dir, f"{uid}_{oid}")
                    # caption_file = os.path.join(sample_dir, "generated_caption.json")
                    caption_file = os.path.join(sample_dir, "extracted_caption.json")

                    if not os.path.exists(caption_file):
                        continue
                    try:
                        with open(caption_file, "r", encoding="utf-8") as handle:
                            caption_info = json.load(handle)
                    except Exception:
                        continue
                    for entry in caption_info.get("generated_captions", []):
                        img_path = entry.get("image_path")
                        raw_caption = entry.get("generated_caption", "")
                        idx = path_to_index.get(img_path)
                        if idx is None:
                            continue
                        attr_dict = parse_attribute_fields(raw_caption)
                        if not attr_dict:
                            continue
                        for key in TEXT_ATTRIBUTE_KEYS:
                            value = attr_dict.get(key, "").strip()
                            if value:
                                attribute_texts[key].append(f"{key}: {value}")
                                attribute_indices[key].append(idx)

            has_text_alignment_samples = any(attribute_texts[key] for key in TEXT_ATTRIBUTE_KEYS)
            if has_text_alignment_samples:
                clip_model, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained="laion2b-s32b-b79K")
                clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
                clip_model = clip_model.to(device)
                clip_model.eval()

                clip_dataset = FashionEvalDataset(gen4clip)
                clip_loader = DataLoader(
                    clip_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=num_workers
                )
                image_feature_batches = []
                with torch.no_grad():
                    for batch in clip_loader:
                        batch = batch.to(device)
                        feats = clip_model.encode_image(batch)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                        image_feature_batches.append(feats.cpu())
                if image_feature_batches:
                    image_features = torch.cat(image_feature_batches, dim=0)
                    for key in TEXT_ATTRIBUTE_KEYS:
                        texts = attribute_texts[key]
                        if not texts:
                            continue
                        indices = attribute_indices[key]
                        scores = []
                        start = 0
                        with torch.no_grad():
                            while start < len(texts):
                                end = min(start + args.batch_size, len(texts))
                                text_batch = texts[start:end]
                                idx_batch = indices[start:end]
                                tokenized = clip_tokenizer(text_batch).to(device)
                                text_feats = clip_model.encode_text(tokenized)
                                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                                img_batch = image_features[torch.tensor(idx_batch, dtype=torch.long)]
                                img_batch = img_batch.to(device)
                                # batch_scores = 100 * torch.sum(img_batch * text_feats, dim=-1)
                                batch_scores = torch.sum(img_batch * text_feats, dim=-1)
                                scores.append(batch_scores.cpu())
                                start = end
                        if scores:
                            merged = torch.cat(scores, dim=0)
                            text_align_scores[key] = merged.mean().item()
                    del image_features
                del clip_model
                torch.cuda.empty_cache()

        all_eval_metrics[ckpt]["Text Align"] = text_align_scores
        np.save(eval_save_path, np.array(all_eval_metrics))


        # -------------------------------------------------------------- #
        #                   Evaluating Text_Personalization              #
        # -------------------------------------------------------------- #
        print("Evaluating Text_Personalization...")
        text_personal_pairs = {key: {"gen": [], "pref": []} for key in TEXT_ATTRIBUTE_KEYS}
        for uid in tqdm(gen_data):
            for oid in gen_data[uid]:
                sample_dir = os.path.join(gen_ckpt_dir, f"{uid}_{oid}")
                caption_file = os.path.join(sample_dir, "generated_caption.json")
                if not os.path.exists(caption_file):
                    continue
                try:
                    with open(caption_file, "r", encoding="utf-8") as handle:
                        caption_info = json.load(handle)
                except Exception:
                    continue

                preference_text = caption_info.get("preference_text", "")
                preference_attrs = parse_attribute_fields(preference_text)

                for entry in caption_info.get("generated_captions", []):
                    gen_caption = entry.get("generated_caption", "")
                    gen_attrs = parse_attribute_fields(gen_caption)
                    for key in TEXT_ATTRIBUTE_KEYS:
                        gen_val = gen_attrs.get(key, "").strip()
                        pref_val = preference_attrs.get(key, "").strip()
                        if gen_val and pref_val:
                            text_personal_pairs[key]["gen"].append(f"{key}: {gen_val}")
                            text_personal_pairs[key]["pref"].append(f"{key}: {pref_val}")

        text_personal_scores = {key: float("nan") for key in TEXT_ATTRIBUTE_KEYS}
        has_pairs = any(len(text_personal_pairs[key]["gen"]) > 0 for key in TEXT_ATTRIBUTE_KEYS)
        if has_pairs:
            clip_model, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained="laion2b-s32b-b79K")
            clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
            clip_model = clip_model.to(device)
            clip_model.eval()

            def encode_texts(texts):
                encoded = []
                with torch.no_grad():
                    for start in range(0, len(texts), args.batch_size):
                        end = min(start + args.batch_size, len(texts))
                        tokens = clip_tokenizer(texts[start:end]).to(device)
                        feats = clip_model.encode_text(tokens)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                        encoded.append(feats.cpu())
                return torch.cat(encoded, dim=0) if encoded else None

            for key in TEXT_ATTRIBUTE_KEYS:
                gen_texts = text_personal_pairs[key]["gen"]
                pref_texts = text_personal_pairs[key]["pref"]
                if not gen_texts:
                    continue
                gen_emb = encode_texts(gen_texts)
                pref_emb = encode_texts(pref_texts)
                if gen_emb is None or pref_emb is None or gen_emb.size(0) == 0:
                    continue
                # sims = 100 * torch.sum(gen_emb * pref_emb, dim=-1)
                sims = torch.sum(gen_emb * pref_emb, dim=-1)

                text_personal_scores[key] = sims.mean().item()

            del clip_model
            torch.cuda.empty_cache()

        all_eval_metrics[ckpt]["Text Personalization"] = text_personal_scores
        np.save(eval_save_path, np.array(all_eval_metrics))
        



        # -------------------------------------------------------------- #
        #                   Evaluating Text_Diversity                    #
        # -------------------------------------------------------------- #
        print("Evaluating Text_Diversity...")
        semantic_group_ranges = []
        caption_texts_all = []
        attribute_value_counts = {key: defaultdict(int) for key in TEXT_ATTRIBUTE_KEYS}

        for uid in tqdm(gen_data):
            for oid in gen_data[uid]:
                sample_dir = os.path.join(gen_ckpt_dir, f"{uid}_{oid}")
                caption_file = os.path.join(sample_dir, "generated_caption.json")
                if not os.path.exists(caption_file):
                    continue
                try:
                    with open(caption_file, "r", encoding="utf-8") as handle:
                        caption_info = json.load(handle)
                except Exception:
                    continue

                start_idx = len(caption_texts_all)
                for entry in caption_info.get("generated_captions", []):
                    caption_text = entry.get("generated_caption", "").strip()
                    if not caption_text:
                        continue
                    caption_texts_all.append(caption_text)
                    attr_dict = parse_attribute_fields(caption_text)
                    for key in TEXT_ATTRIBUTE_KEYS:
                        value = attr_dict.get(key, "").strip()
                        if value:
                            attribute_value_counts[key][value] += 1
                end_idx = len(caption_texts_all)
                if end_idx - start_idx >= 2:
                    semantic_group_ranges.append((start_idx, end_idx))

        semantic_diversity = float("nan")
        if semantic_group_ranges and caption_texts_all:
            clip_model, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained="laion2b-s32b-b79K")
            clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
            clip_model = clip_model.to(device)
            clip_model.eval()

            text_features = []
            with torch.no_grad():
                for start in range(0, len(caption_texts_all), args.batch_size):
                    end = min(start + args.batch_size, len(caption_texts_all))
                    tokens = clip_tokenizer(caption_texts_all[start:end]).to(device)
                    feats = clip_model.encode_text(tokens)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    text_features.append(feats.cpu())
            if text_features:
                text_features = torch.cat(text_features, dim=0)
                semantic_scores = []
                for start_idx, end_idx in semantic_group_ranges:
                    group_feats = text_features[start_idx:end_idx]
                    num_caps = group_feats.size(0)
                    if num_caps < 2:
                        continue
                    sim_matrix = torch.matmul(group_feats, group_feats.T)
                    idx = torch.triu_indices(num_caps, num_caps, offset=1)
                    if idx.numel() == 0:
                        continue
                    cos_vals = sim_matrix[idx[0], idx[1]]
                    distances = 1 - cos_vals
                    if distances.numel() > 0:
                        semantic_scores.append(distances.mean().item())
                if semantic_scores:
                    semantic_diversity = float(np.mean(semantic_scores))
            del clip_model
            torch.cuda.empty_cache()

        attribute_entropy_values = {}
        attribute_counts_total = {}
        for key in TEXT_ATTRIBUTE_KEYS:
            counts = attribute_value_counts[key]
            total = sum(counts.values())
            attribute_counts_total[key] = total
            if total == 0:
                continue
            probs = np.array(list(counts.values()), dtype=np.float64) / float(total)
            entropy = float(-(probs * np.log(probs + 1e-12)).sum())
            attribute_entropy_values[key] = entropy
        attribute_entropy_mean = (
            float(np.mean(list(attribute_entropy_values.values())))
            if attribute_entropy_values else float("nan")
        )
        positive_totals = [attribute_counts_total[k] for k in TEXT_ATTRIBUTE_KEYS if attribute_counts_total[k] > 0]
        total_counts_sum = sum(positive_totals)
        attribute_entropy_weighted = (
            float(
                sum(attribute_entropy_values.get(k, 0.0) * attribute_counts_total[k] for k in TEXT_ATTRIBUTE_KEYS)
                / total_counts_sum
            )
            if total_counts_sum > 0 else float("nan")
        )

        # sem_str = f"{semantic_diversity:.4f}" if semantic_diversity == semantic_diversity else "N/A"
        # attr_mean_str = f"{attribute_entropy_mean:.4f}" if attribute_entropy_mean == attribute_entropy_mean else "N/A"
        # attr_weighted_str = (
        #     f"{attribute_entropy_weighted:.4f}"
        #     if attribute_entropy_weighted == attribute_entropy_weighted else "N/A"
        # )
        # print(rf" $D_s = \frac{{1}}{{K}}\sum_{{k=1}}^K \frac{{2}}{{n_k(n_k-1)}} \sum_{{i<j}}\left(1 - \cos(f_{{k,i}}, f_{{k,j}})\right) = {sem_str}")
        # print(rf" $D_a = \frac{{1}}{{|A|}}\sum_{{a \in A}} \left(-\sum_{{v}} p_{{a,v}} \log p_{{a,v}}\right) = {attr_mean_str}")
        # print(rf" $D_a^w = \frac{{\sum_a n_a H_a}}{{\sum_a n_a}} = {attr_weighted_str}")
        for key in TEXT_ATTRIBUTE_KEYS:
            entropy = attribute_entropy_values.get(key, float("nan"))
            ent_str = f"{entropy:.4f}" if entropy == entropy else "N/A"
            count = attribute_counts_total.get(key, 0)
            print(f"    - {key}: H = {ent_str}, count = {count}")

        all_eval_metrics[ckpt]["Text Diversity Semantic"] = semantic_diversity
        all_eval_metrics[ckpt]["Text Diversity Attribute Mean"] = attribute_entropy_mean
        all_eval_metrics[ckpt]["Text Diversity Attribute Weighted"] = attribute_entropy_weighted
        all_eval_metrics[ckpt]["Text Diversity Attribute Detail"] = attribute_entropy_values
        np.save(eval_save_path, np.array(all_eval_metrics))


        # -------------------------------------------------------------- #
        #                   Evaluating Text_Compatibility                #
        # -------------------------------------------------------------- #
        # print("Evaluating Text_Compatibility...")
        # num = 0
        # text_compat_attr_values = {key: [] for key in TEXT_ATTRIBUTE_KEYS}
        # text_compat_detail = []
        # if gemini_model is None:
        #     print("Gemini model unavailable. Skipping Text_Compatibility scoring.")
        # else:
        #     for uid in tqdm(gen_data):
        #         for oid in gen_data[uid]:
        #             num += 1
        #             if num > 2:
        #                 break
        #             sample_dir = os.path.join(gen_ckpt_dir, f"{uid}_{oid}")
        #             incomplete_image_paths = sorted(glob.glob(os.path.join(sample_dir, "incomplete_outfit_*.png")))
        #             gemini_images = [load_image_for_gemini(p) for p in incomplete_image_paths]
        #             gemini_images = [img for img in gemini_images if img is not None]
        #             if not gemini_images:
        #                 continue
        #             caption_file = os.path.join(sample_dir, "generated_caption.json")
        #             if not os.path.exists(caption_file):
        #                 continue
        #             try:
        #                 with open(caption_file, "r", encoding="utf-8") as handle:
        #                     caption_info = json.load(handle)
        #             except Exception:
        #                 continue
        #             for entry in caption_info.get("generated_captions", []):
        #                 attr_dict = parse_attribute_fields(entry.get("generated_caption", ""))
        #                 if not attr_dict:
        #                     continue
        #                 for key in TEXT_ATTRIBUTE_KEYS:
        #                     attr_value = attr_dict.get(key, "").strip()
        #                     if not attr_value:
        #                         continue
        #                     score = score_attribute_with_gemini(gemini_model, gemini_images, key, attr_value)
        #                     if score is not None:
        #                         text_compat_attr_values[key].append(score)
        #                         text_compat_detail.append({
        #                             "uid": int(uid),
        #                             "oid": int(oid),
        #                             "attribute": key,
        #                             "attribute_value": attr_value,
        #                             "score": score,
        #                             "incomplete_outfit_images": incomplete_image_paths,
        #                             "caption_sample_idx": entry.get("caption_sample_idx")
        #                         })

        # text_compat_attr_scores = {}
        # valid_attribute_means = []
        # for key in TEXT_ATTRIBUTE_KEYS:
        #     values = text_compat_attr_values[key]
        #     if values:
        #         mean_score = float(np.mean(values))
        #         text_compat_attr_scores[key] = mean_score
        #         valid_attribute_means.append(mean_score)
        #     else:
        #         text_compat_attr_scores[key] = float("nan")
        # text_compat_overall = float(np.mean(valid_attribute_means)) if valid_attribute_means else float("nan")

        # all_eval_metrics[ckpt]["Text Compatibility"] = {
        #     "per_attribute": text_compat_attr_scores,
        #     "overall": text_compat_overall
        # }
        # all_eval_metrics[ckpt]["Text Compatibility Detail"] = text_compat_detail
        # np.save(eval_save_path, np.array(all_eval_metrics))
        

        del outfits
        del grd_outfits

        print("-" * 10 + f"{args.eval_version}-checkpoint-{str(ckpt)}" + "-" * 10)
        print("##### Image Branch #####")
        print("## Quality ##")
        print(" " * 2 + f"[IS Accuracy]: {is_acc:.2f}")
        print(" " * 2 + f"[Inception Score]: {is_score:.2f}")
        print()
        print("## Personalization ##")
        print(" " * 2 + f"[Personal Sim]: {personal_sim_score:.2f}")
        print()
        print("## Compatibility ##")
        print(" " * 2 + f"[Compatibility score]: {compatibility_score:.2f}")
        print(" " * 2 + f"[Grd Compatibility score]: {grd_compatibility_score:.2f}")
        print()
        print("## Diversity ##")
        print(" " * 2 + f"[Diversity Score]: {lpip_score:.2f}")
        print()
        print("##### Text Branch #####")
        print("## Text_Align ##")
        for key in TEXT_ATTRIBUTE_KEYS:
            value = text_align_scores.get(key, float("nan"))
            score_str = f"{value:.2f}" if value == value else "N/A"
            print(" " * 2 + f"[{key}]: {score_str}")
        # print("## Text_Compatibility ##")
        # compat_overall_str = f"{text_compat_overall:.2f}" if text_compat_overall == text_compat_overall else "N/A"
        # print(" " * 2 + f"[Overall]: {compat_overall_str}")
        # for key in TEXT_ATTRIBUTE_KEYS:
        #     value = text_compat_attr_scores.get(key, float("nan"))
        #     score_str = f"{value:.2f}" if value == value else "N/A"
        #     print(" " * 2 + f"[{key}]: {score_str}")
        print("## Text_Diversity ##")
        sem_disp = f"{semantic_diversity:.2f}" if semantic_diversity == semantic_diversity else "N/A"
        attr_mean_disp = f"{attribute_entropy_mean:.2f}" if attribute_entropy_mean == attribute_entropy_mean else "N/A"
        attr_weighted_disp = (
            f"{attribute_entropy_weighted:.2f}"
            if attribute_entropy_weighted == attribute_entropy_weighted else "N/A"
        )
        print(" " * 2 + f"[Semantic Diversity D_s]: {sem_disp}")
        print(" " * 2 + f"[Attribute Entropy Mean D_a]: {attr_mean_disp}")
        print(" " * 2 + f"[Attribute Entropy Weighted D_a^w]: {attr_weighted_disp}")
        for key in TEXT_ATTRIBUTE_KEYS:
            entropy = attribute_entropy_values.get(key, float("nan"))
            ent_disp = f"{entropy:.2f}" if entropy == entropy else "N/A"
            count_disp = attribute_counts_total.get(key, 0)
            print(" " * 4 + f"- {key}: H = {ent_disp}, count = {count_disp}")
        print("## Text_Personalization ##")
        for key in TEXT_ATTRIBUTE_KEYS:
            value = text_personal_scores.get(key, float("nan"))
            score_str = f"{value:.2f}" if value == value else "N/A"
            print(" " * 2 + f"[{key}]: {score_str}")

        # print("## Text_Compatibility ##")
        # compat_metrics_summary = all_eval_metrics[ckpt].get("Text Compatibility", {})
        # print(" " * 2 + f"[Metrics Dict]: {compat_metrics_summary}")

        # detail_count = len(text_compat_detail) if isinstance(text_compat_detail, list) else 0
        # print(" " * 2 + f"[Detail Pairs Logged]: {detail_count}")






    print(f"All the ckpts of {args.eval_version} have been evaluated: {ckpts}")
    print(all_eval_metrics)
    print(f"Successfully saved evaluation results of {args.eval_version} checkpoint-{ckpt} to {eval_save_path}.")

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    main()
