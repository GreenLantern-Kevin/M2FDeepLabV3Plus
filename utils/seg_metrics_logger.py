import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class SegMetricHistory:
    """按 epoch 记录的训练/验证指标历史，用于写 CSV + 画图。"""

    epochs: List[int] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)

    train_loss: List[float] = field(default_factory=list)
    train_pacc: List[float] = field(default_factory=list)

    val_loss: List[float] = field(default_factory=list)
    val_pacc: List[float] = field(default_factory=list)
    val_macc: List[float] = field(default_factory=list)
    val_miou: List[float] = field(default_factory=list)
    val_fwiou: List[float] = field(default_factory=list)
    
    # 新增：保存单类的指标历史用于画图
    val_ls_iou: List[float] = field(default_factory=list)
    val_ls_prec: List[float] = field(default_factory=list)
    val_ls_rec: List[float] = field(default_factory=list)
    val_ls_f1: List[float] = field(default_factory=list)


class SegMetricsLogger:
    """将每个 epoch 的指标：
    1) 追加写入 exp_dir/metrics.csv
    2) 维护内存中的 history，训练结束后画曲线保存到 exp_dir

    你只需要在 train.py 中：
        logger = SegMetricsLogger(exp_dir)
        ... 每个 epoch 结束后 logger.log_epoch(...)
        ... 训练结束后 logger.save_plots()

    注意：这里假设你的 val_score 是 StreamSegMetrics.get_results() 的输出：
        {
          'Overall Acc': float,
          'Mean Acc': float,
          'FreqW Acc': float,
          'Mean IoU': float,
          ...
        }
    """

    def __init__(self, exp_dir: str, csv_name: str = "metrics.csv"):
        self.exp_dir = exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)

        self.csv_path = os.path.join(self.exp_dir, csv_name)
        self.history = SegMetricHistory()

        # CSV 表头：加入了背景(bg)和滑坡(ls)的详细指标
        self.fieldnames = [
            "epoch", "lr", "train_loss", "train_pacc", "val_loss",
            "val_pacc", "val_macc", "val_miou", "val_fwiou",
            "val_bg_iou", "val_bg_prec", "val_bg_rec", "val_bg_f1",
            "val_ls_iou", "val_ls_prec", "val_ls_rec", "val_ls_f1"
        ]

        # 若 CSV 不存在就创建并写入表头
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    @staticmethod
    def _safe_float(x, default: float = 0.0) -> float:
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    def log_epoch(self, epoch: int, lr: float, train_loss: float, train_pacc: float, val_loss: float, val_score: Dict) -> None:
        """记录一个 epoch 的指标。

        参数：
            epoch:      从 1 开始更直观（你可以传 epoch+1）
            lr:         当前学习率（一般记录 head 的 lr 或 base lr）
            train_loss: 当前 epoch 的平均训练 loss
            train_pacc: 当前 epoch 的像素准确率（忽略 255）
            val_loss:   当前 epoch 的平均验证 loss
            val_score:  StreamSegMetrics.get_results() 返回的 dict
        """

        # 解析公共指标
        val_pacc = self._safe_float(val_score.get("Overall Acc"))
        val_macc = self._safe_float(val_score.get("Mean Acc"))
        val_fwiou = self._safe_float(val_score.get("FreqW Acc"))
        val_miou = self._safe_float(val_score.get("Mean IoU"))

        # 解析单类指标 (0: Background, 1: Landslide)
        cls_iou = val_score.get("Class IoU", {0: 0.0, 1: 0.0})
        cls_prec = val_score.get("Class Precision", {0: 0.0, 1: 0.0})
        cls_rec = val_score.get("Class Recall", {0: 0.0, 1: 0.0})
        cls_f1 = val_score.get("Class F1", {0: 0.0, 1: 0.0})

        row = {
            "epoch": int(epoch),
            "lr": self._safe_float(lr),
            "train_loss": self._safe_float(train_loss),
            "train_pacc": self._safe_float(train_pacc),
            "val_loss": self._safe_float(val_loss),
            "val_pacc": val_pacc,
            "val_macc": val_macc,
            "val_miou": val_miou,
            "val_fwiou": val_fwiou,
            "val_bg_iou": cls_iou.get(0, 0.0),
            "val_bg_prec": cls_prec.get(0, 0.0),
            "val_bg_rec": cls_rec.get(0, 0.0),
            "val_bg_f1": cls_f1.get(0, 0.0),
            "val_ls_iou": cls_iou.get(1, 0.0),
            "val_ls_prec": cls_prec.get(1, 0.0),
            "val_ls_rec": cls_rec.get(1, 0.0),
            "val_ls_f1": cls_f1.get(1, 0.0),
        }

        # 1) 追加写 CSV（每个 epoch 落盘一次，防止中途崩溃丢数据）
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

        # 2) 记录到 history（用于画图）
        self.history.epochs.append(int(epoch))
        self.history.lr.append(row["lr"])
        self.history.train_loss.append(row["train_loss"])
        self.history.train_pacc.append(row["train_pacc"])
        self.history.val_loss.append(row["val_loss"])
        self.history.val_pacc.append(row["val_pacc"])
        self.history.val_macc.append(row["val_macc"])
        self.history.val_miou.append(row["val_miou"])
        self.history.val_fwiou.append(row["val_fwiou"])
        
        # 记录滑坡类的具体表现，用于最终画图
        self.history.val_ls_iou.append(row["val_ls_iou"])
        self.history.val_ls_prec.append(row["val_ls_prec"])
        self.history.val_ls_rec.append(row["val_ls_rec"])
        self.history.val_ls_f1.append(row["val_ls_f1"])

        # 生成并覆盖保存最新 Epoch 的混淆矩阵热力图
        cm = val_score.get("Confusion Matrix", None)
        if cm is not None:
            self._save_confusion_matrix(cm, epoch)

    def _save_confusion_matrix(self, cm, epoch):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(6, 5))
        # 按行归一化 (Recall视角)，看真实的各类有多少预测对/错
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues",
                    xticklabels=["Background", "Landslide"],
                    yticklabels=["Background", "Landslide"])
        plt.title(f"Confusion Matrix (Normalized) - Epoch {epoch}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, "confusion_matrix_latest.png"), dpi=200)
        plt.close()
            
    def save_plots(self) -> None:
        """训练结束后画 3 张图并保存到 exp_dir。

        1) loss 曲线：train_loss vs val_loss
        2) PAcc 曲线：train_pacc vs val_pacc
        3) val 指标曲线：val_macc / val_miou / val_fwiou
        """
        if len(self.history.epochs) == 0:
            return

        # 这里不强依赖项目里其它画图代码，直接用 matplotlib
        import matplotlib

        # 如果你在服务器/无 GUI 环境跑，强制用 Agg
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = self.history.epochs

        # -------- 1) Loss --------
        plt.figure()
        plt.plot(epochs, self.history.train_loss, label="train_loss")
        plt.plot(epochs, self.history.val_loss, label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, "loss_curve.png"), dpi=200)
        plt.close()

        # -------- 2) PAcc --------
        plt.figure()
        plt.plot(epochs, self.history.train_pacc, label="train_PAcc")
        plt.plot(epochs, self.history.val_pacc, label="val_PAcc")
        plt.xlabel("epoch")
        plt.ylabel("PAcc")
        plt.title("Pixel Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, "pacc_curve.png"), dpi=200)
        plt.close()

        # -------- 3) mAcc / mIoU  --------
        plt.figure()
        plt.plot(epochs, self.history.val_macc, label="val_mAcc")
        plt.plot(epochs, self.history.val_miou, label="val_mIoU")
        plt.xlabel("epoch")
        plt.ylabel("score")
        plt.title("Validation Metrics (Mean)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, "val_metrics_curve.png"), dpi=200)
        plt.close()

        # 4) [新增] 专门为“滑坡”单独画的综合性能雷达曲线
        plt.figure()
        plt.plot(epochs, self.history.val_ls_iou, label="Landslide IoU", color='orange')
        plt.plot(epochs, self.history.val_ls_f1, label="Landslide F1-Score", color='red')
        plt.plot(epochs, self.history.val_ls_prec, label="Landslide Precision", color='green', linestyle='--')
        plt.plot(epochs, self.history.val_ls_rec, label="Landslide Recall", color='blue', linestyle='--')
        plt.xlabel("epoch")
        plt.ylabel("score")
        plt.title("Landslide Class Specific Metrics")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, "landslide_specific_metrics.png"), dpi=200)
        plt.close()
