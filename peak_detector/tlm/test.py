import torch
import matplotlib.pyplot as plt
from peak_detector.tlm.utils.predictor import TLMPredictor
from peak_detector.tlm.utils.evaluator import TLMEvaluator
from peak_detector.tlm.data.dataloader import get_dataloaders

def predict_from_nxs(folder: str, scan_no: str, threshold: float, use_nms=False):
    pred = TLMPredictor("peak_detector/tlm/state_dicts/474.torch")
    img_no = 36
    outs = pred.predict(folder, scan_no, img_no, 0.4)
    pred.show_output(folder, scan_no, img_no, outs[0], outs[1])

def precision_recall():
    ious = []
    p = []
    r = []
    train, test = get_dataloaders(32, 0.75, "peak_detector/clean_full_dataset.csv", 'mm24570-1')
    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for i in thresholds:
        print(f"threshold: {i}")
        eval = TLMEvaluator(test, 'peak_detector/tlm/state_dicts/474.torch', torch.device('cpu'), i)
        #mean_iou = eval.compute_mean_iou()
        precision, recall = eval.precision_recall()
        #ious.append(mean_iou)
        p.append(precision)
        r.append(recall)

    print(ious)
    print(p)
    print(r)
    plt.scatter(p, r)
    plt.title("Precision-Recall Curve")
    plt.show()
    
    

if __name__ == "__main__":
    # folder = "mm33190-1"
    # scan_no = "1007323"
    # predict_from_nxs(folder, scan_no, 0.35)

    precision_recall()
