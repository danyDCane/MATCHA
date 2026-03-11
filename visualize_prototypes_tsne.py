import argparse
import os

import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import util
from pacs_dataset import PACSDataset
from models.resnet import StandardResNetWrapper, ResNet, CosineClassifier


def load_model_from_checkpoint(ckpt_path: str, device: str = "cuda"):
    # weights_only=False: checkpoint 含 args (argparse.Namespace)，PyTorch 2.6+ 預設 weights_only=True 會拒絕
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    num_classes = 7 if args.dataset == "pacs" else 10
    model = util.select_model(num_classes, args).to(device)

    # backbone_state 會連同 diffusion_model.* 一起存進來，
    # 這裡只載入分類 backbone，過濾掉所有 diffusion_model.* 的權重。
    full_state = ckpt["backbone_state"]
    filtered_state = {
        k: v for k, v in full_state.items()
        if not k.startswith("diffusion_model.")
    }
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if unexpected:
        print(f"[visualize_prototypes_tsne] Ignored unexpected keys (showing first 10): {list(unexpected)[:10]}")
    if missing:
        print(f"[visualize_prototypes_tsne] Missing keys when loading backbone (first 10): {list(missing)[:10]}")

    model.eval()
    return model, args, ckpt


def build_test_loader_pacs(args, domain_name: str):
    """
    建立 PACS 某個 domain 的 test loader。
    """
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    dataset = PACSDataset(
        root=args.datasetRoot,
        dataset_name=domain_name,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=True,
    )
    return loader


def collect_features(model, loader, device: str = "cuda"):
    """
    用 intermediate_forward 抽取特徵 + label。
    """
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if hasattr(model, "intermediate_forward"):
                feats = model.intermediate_forward(x)
            else:
                out = model(x, return_feature=True)
                if isinstance(out, tuple) and len(out) == 2:
                    _, feats = out
                else:
                    raise RuntimeError(
                        "Model has no intermediate_forward and forward(...) did not return (logits, feat)."
                    )

            feats = feats.view(feats.size(0), -1)
            all_feats.append(feats.cpu())
            all_labels.append(y.cpu())

    feats = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return feats.numpy(), labels.numpy()


def get_classifier_weight(model):
    """
    取得最後分類頭的權重矩陣 W，形狀 [C, D]。
    """
    head = None
    if isinstance(model, StandardResNetWrapper):
        head = model.backbone.fc
    elif isinstance(model, ResNet):
        head = model.linear
    else:
        for name in ["fc", "classifier", "linear", "head"]:
            if hasattr(model, name):
                head = getattr(model, name)
                break

    if head is None:
        raise RuntimeError("Cannot find classifier head on the model.")

    if isinstance(head, CosineClassifier):
        W = head.weight.detach()
    elif isinstance(head, torch.nn.Linear):
        W = head.weight.detach()
    else:
        if hasattr(head, "weight"):
            W = head.weight.detach()
        else:
            raise RuntimeError("Classifier head has no .weight attribute.")

    return W.cpu().numpy()


def run_tsne_with_prototypes(features, labels, W, save_path, title=None, random_state=42):
    """
    將樣本特徵 + C 個 prototype 一起跑 t-SNE。
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).ravel()
    W = np.asarray(W, dtype=np.float64)

    # L2 normalize，模擬 cosine 幾何
    def l2norm(x, axis=1, eps=1e-8):
        norm = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
        return x / np.maximum(norm, eps)

    features_norm = l2norm(features, axis=1)
    W_norm = l2norm(W, axis=1)

    N = features_norm.shape[0]
    C = W_norm.shape[0]

    X_all = np.concatenate([features_norm, W_norm], axis=0)
    tsne = TSNE(n_components=2, random_state=random_state)
    X_2d = tsne.fit_transform(X_all)

    feat_2d = X_2d[:N]
    proto_2d = X_2d[N:]

    plt.figure(figsize=(10, 8))

    # 類別 id 假設為 0..C-1，顏色對齊 util.draw_tsne_feature_distribution（直接用 labels 當顏色索引）
    classes = np.arange(C)
    cmap = plt.get_cmap("viridis")

    # 樣本點：c=labels，與 util.tsne 畫法一致，確保 7 類顏色固定
    scatter = plt.scatter(
        feat_2d[:, 0],
        feat_2d[:, 1],
        c=labels,
        cmap=cmap,
        s=15,
        alpha=0.4,
        linewidths=0.0,
    )

    # prototype：使用對應類別 id 的顏色（與該類樣本同色）
    for c_idx in classes:
        color = scatter.cmap(scatter.norm(c_idx))
        plt.scatter(
            proto_2d[c_idx, 0],
            proto_2d[c_idx, 1],
            c=[color],
            marker="*",
            s=200,
            edgecolors="k",
            linewidths=1.0,
        )
        plt.text(
            proto_2d[c_idx, 0],
            proto_2d[c_idx, 1],
            str(c_idx),
            fontsize=10,
            ha="center",
            va="center",
            color="k",
        )

    plt.xticks([])
    plt.yticks([])
    if title:
        plt.title(title)

    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"[t-SNE] Saved prototype visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize features (optionally with class prototypes) via t-SNE."
    )
    # 單一模型、單一 domain（原本的用法）
    parser.add_argument("--checkpoint", type=str,
                        help="Path to checkpoint (.pth) with backbone_state and args.")
    parser.add_argument("--domain", type=str,
                        help="PACS domain name whose features you want to visualize "
                             "(art_painting, photo, sketch, cartoon).")
    parser.add_argument("--save_path", type=str,
                        help="Path to save the t-SNE figure.")

    # 多 checkpoint × 多 domain 批次模式：會自動產生多張 t-SNE 圖
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Enable batch mode: loop over multiple checkpoints (source domains) "
             "and multiple target domains to generate many t-SNE figures.",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        help="List of checkpoint paths when using --multi. "
             "Each checkpoint will be evaluated on all target_domains.",
    )
    parser.add_argument(
        "--target_domains",
        type=str,
        nargs="+",
        default=["art_painting", "cartoon", "photo", "sketch"],
        help="Target domains to visualize on when using --multi. "
             "Default: art_painting cartoon photo sketch.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to save multiple t-SNE figures when using --multi.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 批次模式：多個 source-domain 模型 × 多個 target domains，一次產生多張圖。
    if args.multi:
        if not args.checkpoints or not args.save_dir:
            raise ValueError(
                "--multi requires --checkpoints (list of ckpt paths) and --save_dir."
            )

        os.makedirs(args.save_dir, exist_ok=True)

        for ckpt_path in args.checkpoints:
            # 從檔名猜一個簡單的模型名稱，例如 checkpoint_art_painting_epoch150.pth -> checkpoint_art_painting_epoch150
            src_name = os.path.splitext(os.path.basename(ckpt_path))[0]
            print(f"[multi] Loading model '{src_name}' from: {ckpt_path}")
            model, train_args, ckpt = load_model_from_checkpoint(ckpt_path, device=device)
            if train_args.dataset != "pacs":
                raise ValueError("This script is currently written for PACS (dataset == 'pacs').")

            W = get_classifier_weight(model)

            for tgt_dom in args.target_domains:
                print(f"[multi] Model='{src_name}'  TargetDomain='{tgt_dom}'")
                loader = build_test_loader_pacs(train_args, tgt_dom)
                feats, labels = collect_features(model, loader, device=device)

                title = f"Prototypes vs features (model={src_name}, domain={tgt_dom})"
                filename = f"prototypes_{src_name}_test_{tgt_dom}.png"
                save_path = os.path.join(args.save_dir, filename)

                run_tsne_with_prototypes(
                    feats,
                    labels,
                    W,
                    save_path=save_path,
                    title=title,
                    random_state=train_args.randomSeed,
                )
        return

    # 單一模型、單一 domain：維持原本用法
    if not args.checkpoint or not args.domain or not args.save_path:
        raise ValueError("Single-mode requires --checkpoint, --domain and --save_path.")

    model, train_args, ckpt = load_model_from_checkpoint(args.checkpoint, device=device)

    if train_args.dataset != "pacs":
        raise ValueError("This script is currently written for PACS (dataset == 'pacs').")

    loader = build_test_loader_pacs(train_args, args.domain)
    feats, labels = collect_features(model, loader, device=device)

    # 抽完整個 domain 的特徵，並把 7 個 prototype 一起丟進同一個 t-SNE。
    W = get_classifier_weight(model)
    title = f"Prototypes vs features (domain={args.domain})"
    run_tsne_with_prototypes(
        feats,
        labels,
        W,
        save_path=args.save_path,
        title=title,
        random_state=train_args.randomSeed,
    )


if __name__ == "__main__":
    main()

