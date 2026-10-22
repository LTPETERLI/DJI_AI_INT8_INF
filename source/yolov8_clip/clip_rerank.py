# clip_rerank.py
import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict

try:
    import clip  # pip install git+https://github.com/openai/CLIP.git
except Exception as e:
    raise RuntimeError("请先安装 openai/CLIP：pip install git+https://github.com/openai/CLIP.git") from e

# ====== 词表 ======
DRONE = [
    "drone", "uav", "quadcopter", "micro drone", "small drone",
    "nano drone", "multirotor drone", "fixed-wing drone",
    "remote-controlled aircraft", "flying device"
]
DISTRACTOR = [
    "bird", "kite", "helicopter", "airplane", "balloon", "flying toy",
    "street light", "lamp post", "antenna", "satellite dish",
    "air conditioning unit", "ventilation unit", "crane",
    "power pole", "power line tower", "chimney", "tower"
]
BACKGROUND = [
    "sky", "cloud", "sea", "ocean", "water", "building", "tree", "road", "grass", "roof"
]
ALL_CLASSES = DRONE + DISTRACTOR + BACKGROUND

PROMPTS = [
    "an overhead photo of a small {}",
    "a tiny {} from a drone view",
    "an aerial view of a {}",
    "a {} in the sky (overhead)",
    "a {} near power lines (aerial)",
    "a distant {} in the sky (aerial)"
]

# ====== 文本特征 ======
def build_text_features(model, device, classes, templates, agg="max"):
    prompts, spans = [], []
    for c in classes:
        s = len(prompts)
        for t in templates:
            prompts.append(t.format(c))
        e = len(prompts)
        spans.append((s, e))
    with torch.no_grad():
        toks = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        feats = model.encode_text(toks).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats, spans

def _group_score(logits: torch.Tensor, span_list: List[Tuple[int,int]], how="max") -> torch.Tensor:
    scores = []
    for (s, e) in span_list:
        v = logits[:, s:e]
        scores.append(v.mean(dim=1, keepdim=True) if how=="mean" else v.max(dim=1, keepdim=True).values)
    return torch.concat(scores, dim=1).max(dim=1, keepdim=True).values

# ====== 裁剪 ======
def crop_with_context(img: np.ndarray, box: Tuple[float,float,float,float],
                      expand: float = 1.8, min_side: int = 64) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
    bw, bh = (x2-x1), (y2-y1)
    side = max(bw, bh)
    side = max(side*expand, min_side)
    nx1 = int(round(cx - side/2)); ny1 = int(round(cy - side/2))
    nx2 = int(round(cx + side/2)); ny2 = int(round(cy + side/2))
    nx1 = max(0, nx1); ny1 = max(0, ny1); nx2 = min(w-1, nx2); ny2 = min(h-1, ny2)
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return img[ny1:ny2, nx1:nx2, :]

def draw_boxes(img_bgr, boxes, scores=None, color=(0, 255, 255), thickness=2, put_score=True, prefix=""):
    out = img_bgr.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if put_score and scores is not None:
            txt = f"{prefix}{scores[i]:.2f}"
            cv2.putText(out, txt, (x1, max(20, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out

# ====== 主类 ======
class CLIPReranker:
    def __init__(self,
                 model_name: str = "ViT-L/14@336px",
                 device: str = None,
                 prompt_agg: str = "max",
                 margin_tau: float = 0.20,     # 判定门槛：drone - max(dist,bg)
                 fuse_alpha: float = 0.6,      # 融合权重（0=只用YOLO，1=只用CLIP）
                 temperature: float = 1.0):    # 温度缩放
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.text_feats, self.spans = build_text_features(self.model, self.device, ALL_CLASSES, PROMPTS, agg=prompt_agg)
        self.prompt_agg = prompt_agg
        self.margin_tau = margin_tau
        self.fuse_alpha  = fuse_alpha
        self.temperature = temperature

        self.span_dr = self.spans[:len(DRONE)]
        self.span_di = self.spans[len(DRONE):len(DRONE)+len(DISTRACTOR)]
        self.span_bg = self.spans[len(DRONE)+len(DISTRACTOR):]

    @torch.no_grad()
    def score_crops(self, crops: List[Image.Image]) -> Dict[str, np.ndarray]:
        if len(crops) == 0:
            return {"drone": np.array([]), "dist": np.array([]), "bg": np.array([]), "margin": np.array([])}
        ims = torch.stack([self.preprocess(im) for im in crops], dim=0).to(self.device)  # [B,3,336,336]
        imf = self.model.encode_image(ims).float()
        imf = imf / imf.norm(dim=-1, keepdim=True)
        logits = (imf @ self.text_feats.T) / self.temperature  # [B, num_prompts]

        s_dr = _group_score(logits, self.span_dr, self.prompt_agg)  # [B,1]
        s_di = _group_score(logits, self.span_di, self.prompt_agg)
        s_bg = _group_score(logits, self.span_bg, self.prompt_agg)
        margin = s_dr - torch.max(s_di, s_bg)                       # [B,1]

        return {
            "drone": s_dr.squeeze(1).cpu().numpy(),
            "dist":  s_di.squeeze(1).cpu().numpy(),
            "bg":    s_bg.squeeze(1).cpu().numpy(),
            "margin":margin.squeeze(1).cpu().numpy()
        }

    def rerank(self,
               image_bgr: np.ndarray,
               dets_xyxy: List[Tuple[float,float,float,float]],
               det_scores: List[float],
               topN: int = 50,
               expand: float = 1.8,
               min_side: int = 64,
               return_topk: int = 1):
        idx = np.argsort(det_scores)[::-1][:topN]
        dets = [dets_xyxy[i] for i in idx]
        scrs = [det_scores[i] for i in idx]

        crops, valid = [], []
        for i, box in enumerate(dets):
            crop = crop_with_context(image_bgr, box, expand=expand, min_side=min_side)
            if crop is None:
                continue
            crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            valid.append(i)

        clip_scores = self.score_crops(crops)
        margins = clip_scores["margin"]  # len(valid)

        sig = lambda x: 1.0/(1.0+np.exp(-x))
        yolo_sel = np.array([scrs[i] for i in valid], dtype=np.float32)
        fused = (1.0 - self.fuse_alpha)*sig(yolo_sel) + self.fuse_alpha*sig(margins / max(1e-6, self.temperature))

        keep_mask = margins >= self.margin_tau
        fused_filtered = fused.copy()
        fused_filtered[~keep_mask] = -1.0

        order = np.argsort(fused_filtered)[::-1]
        out = []
        for j in order[:return_topk]:
            if fused_filtered[j] < 0:
                continue
            i = valid[j]
            out.append({
                "xyxy": dets[i],
                "yolo_score": float(scrs[i]),
                "clip_margin": float(margins[j]),
                "fused_score": float(fused_filtered[j]),
                "clip_drone": float(clip_scores["drone"][j]),
                "clip_dist":  float(clip_scores["dist"][j]),
                "clip_bg":    float(clip_scores["bg"][j]),
                "idx_in_topN": int(i),
                "valid_index": int(j)
            })
        return out

    # ====== 新增：一站式导出（画YOLO、保存裁剪、画最终结果、CSV） ======
    def export_all(self,
                   image_bgr: np.ndarray,
                   dets_xyxy: List[Tuple[float,float,float,float]],
                   det_scores: List[float],
                   output_dir: str,
                   img_basename: str = "image",
                   topN: int = 50,
                   expand: float = 1.8,
                   min_side: int = 64,
                   return_topk: int = 1):
        os.makedirs(output_dir, exist_ok=True)
        crops_dir = os.path.join(output_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

        H, W = image_bgr.shape[:2]

        # 1) YOLO可视化
        idx_sort = np.argsort(det_scores)[::-1]
        yolo_vis = draw_boxes(image_bgr, [dets_xyxy[i] for i in idx_sort],
                              scores=[det_scores[i] for i in idx_sort],
                              color=(0,255,255), thickness=2, prefix="y:")
        yolo_vis_path = os.path.join(output_dir, f"{img_basename}_yolo_dets.jpg")
        cv2.imwrite(yolo_vis_path, yolo_vis)

        # 2) Top-N裁剪保存
        idx = idx_sort[:topN]
        dets = [dets_xyxy[i] for i in idx]
        scrs = [det_scores[i] for i in idx]

        crops_pil, valid, crop_paths = [], [], []
        for rank, i in enumerate(idx):
            crop = crop_with_context(image_bgr, dets_xyxy[i], expand=expand, min_side=min_side)
            if crop is None:
                continue
            crop_name = f"{img_basename}_crop_{rank:02d}_idx{i}.jpg"
            crop_path = os.path.join(crops_dir, crop_name)
            cv2.imwrite(crop_path, crop)
            crop_paths.append(crop_path)
            crops_pil.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            valid.append(i)

        # 3) CLIP打分 + 融合
        clip_scores = self.score_crops(crops_pil)
        margins = clip_scores["margin"] if len(crops_pil) else np.array([])
        sig = lambda x: 1.0/(1.0+np.exp(-x))
        fused_list = []
        rows = []

        for j, i in enumerate(valid):
            yolo_conf = float(scrs[valid.index(i)]) if i in idx else 0.0
            margin = float(margins[j])
            fused = (1.0 - self.fuse_alpha)*sig(yolo_conf) + self.fuse_alpha*sig(margin/max(1e-6, self.temperature))
            keep = bool(margin >= self.margin_tau)
            fused_list.append((j, i, fused, keep))
            rows.append({
                "rank": j,
                "idx_in_all": int(idx[j]),
                "x1": dets_xyxy[idx[j]][0], "y1": dets_xyxy[idx[j]][1],
                "x2": dets_xyxy[idx[j]][2], "y2": dets_xyxy[idx[j]][3],
                "yolo_score": yolo_conf,
                "clip_drone": float(clip_scores["drone"][j]),
                "clip_dist":  float(clip_scores["dist"][j]),
                "clip_bg":    float(clip_scores["bg"][j]),
                "clip_margin": margin,
                "fused_score": fused,
                "keep_by_margin": int(keep),
                "crop_path": crop_paths[j]
            })

        # CSV
        import csv
        csv_path = os.path.join(output_dir, f"{img_basename}_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                                    ["rank","idx_in_all","x1","y1","x2","y2","yolo_score",
                                     "clip_drone","clip_dist","clip_bg","clip_margin",
                                     "fused_score","keep_by_margin","crop_path"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        # 4) 复核后总图
        final_vis = image_bgr.copy()
        red_boxes, red_scores, green_boxes, green_scores = [], [], [], []
        for (j, i, fused, keep) in fused_list:
            box = dets_xyxy[idx[j]]
            if keep:
                green_boxes.append(box); green_scores.append(fused)
            else:
                red_boxes.append(box); red_scores.append(fused)
        if red_boxes:
            final_vis = draw_boxes(final_vis, red_boxes, red_scores, color=(0,0,255), thickness=2, prefix="f:")
        if green_boxes:
            order = np.argsort(green_scores)[::-1][:return_topk]
            sel_boxes = [green_boxes[k] for k in order]
            sel_scores = [green_scores[k] for k in order]
            final_vis = draw_boxes(final_vis, sel_boxes, sel_scores, color=(0,255,0), thickness=3, prefix="f*:")

        final_vis_path = os.path.join(output_dir, f"{img_basename}_clip_rerank_final.jpg")
        cv2.imwrite(final_vis_path, final_vis)

        return {
            "yolo_vis": yolo_vis_path,
            "crops_dir": crops_dir,
            "csv": csv_path,
            "final_vis": final_vis_path
        }
