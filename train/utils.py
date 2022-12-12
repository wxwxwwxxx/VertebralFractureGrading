import numpy as np
np.seterr(all="ignore")
import torch
import matplotlib.pyplot as plt
from sklearn import metrics as skm
import logging
import os
import shutil
import glob
import csv
from matplotlib import cm
from collections import OrderedDict
class CustomLogger():
    def __init__(self, test_file_fn, test_csv_fn):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(message)s')
        file_handler = logging.FileHandler(test_file_fn)
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        csv_handler = logging.FileHandler(test_csv_fn)
        csv_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(message)s')
        csv_handler.setFormatter(formatter)
        self.logger_train = logging.getLogger("train_log")
        self.logger_test = logging.getLogger("test_log")
        self.logger_test.addHandler(file_handler)
        self.csv_logger = logging.getLogger("test_csv")
        self.csv_logger.addHandler(csv_handler)
        self.csv_logger.propagate = False
        self.csv_header = []
        self.csv_fn = test_csv_fn
        self.best_met = 0
        self.last_ckpt_fn = ""
        self.best_ckpt_fn = ""

    def train_log(self, msg):
        self.logger_train.info(msg)

    def test_log(self, msg):
        self.logger_test.info(msg)

    def csv_log(self, msg_dict):

        if len(self.csv_header) == 0:
            for i in msg_dict:
                self.csv_header.append(i)
            self.csv_header.sort()

            self.csv_header.remove('epoch')
            self.csv_header.insert(0,'epoch')

            if os.path.getsize(self.csv_fn) == 0:
                header_str = ""
                for i in self.csv_header:
                    header_str += i
                    header_str += ','
                header_str = header_str[:-1]
                self.csv_logger.info(header_str)
            else:
                csv_reader = csv.reader(open(self.csv_fn))
                pre_csv_header = next(csv_reader)
                assert pre_csv_header == self.csv_header, f"Csv Header not consistent\npre:{pre_csv_header}\nnow:{self.csv_header}"

        log_str = ""
        for i in self.csv_header:
            log_str += str(msg_dict[i])
            log_str += ','
        log_str = log_str[:-1]
        self.csv_logger.info(log_str)

    def result_log_train(self, confusion_matrix, loss, epoch, g_step, batch_step, tboard_writter):
        confusion_matrix = confusion_matrix.cpu().numpy()
        acc = cal_mean_acc(confusion_matrix)
        rec = cal_rec_2(confusion_matrix)
        diag = cal_diag(confusion_matrix)
        self.train_log('[%d, %5d] loss: %.6f acc: %.6f\n rec0: %.6f rec1: %.6f rec2: %.6f rec3: %.6f'
                       % (epoch, batch_step, loss, acc, rec[0], rec[1], rec[2], rec[3]))
        log_dict = {"loss": loss, "acc": acc, "rec": rec, "diag": diag}
        self.tboard_write(tboard_writter, log_dict, g_step, 10)


    def result_log_test(self, y_true, y_pred_score, epoch, g_step, tboard_writter, tboard_only=False,
                        additional_dict=None):
        if additional_dict is None:
            additional_dict = dict()
        y_pred = np.argmax(y_pred_score, axis=1)
        confusion_matrix = skm.confusion_matrix(y_true, y_pred)
        pre, rec, fs, _ = skm.precision_recall_fscore_support(y_true, y_pred,zero_division=0)
        acc = skm.accuracy_score(y_true, y_pred)
        diag = cal_diag(confusion_matrix)
        met_dict = self.metrics(y_true, y_pred_score)
        if not tboard_only:
            self.test_log(f'[{epoch}]')
            self.test_log(f'\n{skm.classification_report(y_true, y_pred,zero_division = 0)}')
            self.test_log(f'Confusion Matrix:\n{confusion_matrix}')

        log_dict = {"acc": acc, "rec": rec, "pre": pre, "f1": fs , "diag": diag}

        csv_dict = self.tboard_write(tboard_writter, {**log_dict, **met_dict,**additional_dict}, g_step, 1)
        if not tboard_only:
            csv_dict["epoch"] = epoch
            self.csv_log(csv_dict)
        return met_dict["macro_f1"]





    def save_checkpoint(self,save_dict,ckpt_path, met,g_step):
        temp_last_ckpt_fn = f"last_ckpt_{g_step}.tar"
        torch.save(save_dict,
                   os.path.join(ckpt_path, temp_last_ckpt_fn))
        if self.last_ckpt_fn != "":
            if os.path.exists(os.path.join(ckpt_path, self.last_ckpt_fn)):
                os.remove(os.path.join(ckpt_path, self.last_ckpt_fn))
            else:
                print(f"Warning: last_ckpt_fn do not exist, {self.last_ckpt_fn}")
        self.last_ckpt_fn = temp_last_ckpt_fn
        if met > self.best_met:
            self.best_met = met
            self.train_log(f"New Best Metric: {self.best_met}")
            temp_best_ckpt_fn = f"best_ckpt_{g_step}_{self.best_met}.tar"

            shutil.copy2(os.path.join(ckpt_path, self.last_ckpt_fn),
                         os.path.join(ckpt_path, temp_best_ckpt_fn))
            if self.best_ckpt_fn != "":
                if os.path.exists(os.path.join(ckpt_path, self.best_ckpt_fn)):
                    os.remove(os.path.join(ckpt_path, self.best_ckpt_fn))
                else:
                    print(f"Warning: best_ckpt_fn do not exist, {self.best_ckpt_fn}")
            self.best_ckpt_fn = temp_best_ckpt_fn
            return True
        return False

    def load_checkpoint(self,ckpt_path,local_rank):
        f_l = glob.glob(os.path.join(ckpt_path, "last_ckpt_*"))
        checkpoint = None
        if len(f_l) != 0:
            checkpoint = torch.load(os.path.join(ckpt_path, f_l[0]),
                                    map_location=torch.device('cuda', local_rank))

            print(f"Load ckpt, g_step={checkpoint['g_step']}, local_rank={local_rank}")
            self.last_ckpt_fn = os.path.split(f_l[0])[1]
            b_l = glob.glob(os.path.join(ckpt_path, "best_ckpt_*"))
            if len(b_l) > 0:
                self.best_ckpt_fn = os.path.split(b_l[0])[1]
                best_met = float(os.path.splitext(self.best_ckpt_fn)[0].split('_')[-1])
                print(f"Best ckpt detected, best_met={best_met}")
                self.best_met = best_met
        return checkpoint

    def tboard_write(self, writer, datadict, g_step, interval):
        if g_step % interval != 0:
            return
        csv_dict = {}
        for k in datadict:
            if isinstance(datadict[k], torch.Tensor):
                datadict[k] = datadict[k].tolist()
            if isinstance(datadict[k], np.ndarray):
                datadict[k] = datadict[k].tolist()
            if isinstance(datadict[k], list):
                for i, v in enumerate(datadict[k]):
                    if v != 0.0:
                        writer.add_scalar(f'{k}{i}', v, global_step=g_step)
                    csv_dict[f'{k}{i}'] = v
            else:
                if datadict[k] != 0.0:
                    writer.add_scalar(k, datadict[k], global_step=g_step)
                csv_dict[k] = datadict[k]
        return csv_dict
    def multiclass_to_binary(self,y_true, y_scores):
        y_true = np.where(y_true >= 1, 1, 0)
        y_scores = 1 - y_scores[:, 0:1]
        return y_true,y_scores
    def draw_roc_auc(self, y_true, y_scores,g_step, save_fn):
        #To binary
        y_true,y_scores = self.multiclass_to_binary(y_true,y_scores)
        fpr, tpr, threshold = skm.roc_curve(y_true, y_scores)
        roc_auc = skm.auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.title(f'Validation ROC,step = {g_step}')
        plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(save_fn)
        plt.close()
        return roc_auc
    def metrics(self,y_true, y_pred_score):
        y_pred = np.argmax(y_pred_score, axis=1)
        ret_dict = dict()
        ret_dict["weighted_pre"] = skm.precision_score(y_true, y_pred, average='weighted', zero_division=0)
        ret_dict["weighted_rec"] = skm.recall_score(y_true, y_pred, average='weighted', zero_division=0)
        ret_dict["weighted_f1"] = skm.f1_score(y_true, y_pred, average='weighted', zero_division=0)
        ret_dict["macro_pre"] = skm.precision_score(y_true, y_pred, average='macro', zero_division=0)
        ret_dict["macro_rec"] = skm.recall_score(y_true, y_pred, average='macro', zero_division=0)
        ret_dict["macro_f1"] = skm.f1_score(y_true, y_pred, average='macro', zero_division=0)
        y_true, y_scores = self.multiclass_to_binary(y_true, y_pred_score)
        ret_dict["binary_aucroc"] = skm.roc_auc_score(y_true, y_scores)
        return ret_dict


def calculate_confusion_matrix(gt, pred, confusion):
    gt = gt.tolist()
    pred = pred.tolist()
    # print(len(gt))
    # print(len(pred))
    for i in range(len(gt)):
        # print(i)
        confusion[int(gt[i])][int(pred[i])] += 1


def calculate_result(confusion):
    confusion_matrix = confusion
    gt_list = []
    pred_list = []
    for i_gt in range(4):
        for i_pred in range(4):
            l = int(confusion_matrix[i_gt][i_pred])
            gt_sub = np.ones([l], dtype=np.int32) * i_gt
            pred_sub = np.ones([l], dtype=np.int32) * i_pred
            gt_list.append(gt_sub)
            pred_list.append(pred_sub)
    gt_re = np.concatenate(gt_list)
    pred_re = np.concatenate(pred_list)
    return gt_re, pred_re


def cal_mean_acc(confusion_matrix):
    return np.sum(np.diagonal(confusion_matrix)) / np.sum(confusion_matrix)


def cal_rec_2(confusion_matrix):
    return np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis=1)


def cal_pre_2(confusion_matrix):
    return np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis=0)


def cal_diag(confusion_matrix):
    c_m = np.sum(confusion_matrix[:, 1:4], axis=1)
    return c_m / np.sum(confusion_matrix, axis=1)


def img_plot(img):
    plt.figure()  # 设置窗口大小
    plt.suptitle('Multi_Image')  # 图片名称
    for i in range(128):
        plt.subplot(12, 12, i + 1)
        plt.imshow(img[0, 0, i, :, :])
    plt.show()

def hist_plot(data):

    plt.hist(data, bins=200, facecolor="blue", edgecolor="black", alpha=0.7)

    plt.xlabel("x")

    plt.ylabel("y")

    plt.title("hist")
    plt.show()

def plot_with_labels(lowDWeights, labels):
    plt.figure(figsize=(15, 15))
    plt.cla() #clear当前活动的坐标轴

    X, Y = lowDWeights[:, 0], lowDWeights[:, 1] #把Tensor的第1列和第2列,也就是TSNE之后的前两个特征提取出来,作为X,Y
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9))
        #plt.text(x, y, s, backgroundcolor=c, fontsize=9)
        plt.text(x, y, str(s),color=c,fontdict={'weight': 'bold', 'size': 7}) #在指定位置放置文本
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.show()
    #plt.pause(0.01)



def dist_to_single(state_dict):
    n_dict = OrderedDict()
    for i in state_dict:
        n_dict[i[7:]] = state_dict[i]
    return n_dict
def single_to_dist(state_dict):
    n_dict = OrderedDict()
    for i in state_dict:
        n_dict['module.'+i] = state_dict[i]
    return n_dict

if __name__ == "__main__":
    confusion_matrix = torch.zeros([4, 4], requires_grad=False)
    pred = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 2, 1, 2, 3])
    gt = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    calculate_confusion_matrix(gt, pred, confusion_matrix)
    print(confusion_matrix)

    print(confusion_matrix[1, :])
    print(confusion_matrix[:, 1])
    C = confusion_matrix.cpu().numpy()
    print(cal_rec_2(confusion_matrix.cpu().numpy()))
    print(cal_diag(C))

    # y_true, y_pred = calculate_result(confusion_matrix)
    # print('------Weighted------')
    # print('Weighted precision', precision_score(y_true, y_pred, average='weighted', zero_division=0))
    # print('Weighted recall', recall_score(y_true, y_pred, average='weighted', zero_division=0))
    # print('Weighted f1-score', f1_score(y_true, y_pred, average='weighted', zero_division=0))
    # print('------Macro------')
    # print('Macro precision', precision_score(y_true, y_pred, average='macro', zero_division=0))
    # print('Macro recall', recall_score(y_true, y_pred, average='macro', zero_division=0))
    # print('Macro f1-score', f1_score(y_true, y_pred, average='macro', zero_division=0))
    # print('------Micro------')
    # print('Micro precision', precision_score(y_true, y_pred, average='micro', zero_division=0))
    # print('Micro recall', recall_score(y_true, y_pred, average='micro', zero_division=0))
    # print('Micro f1-score', f1_score(y_true, y_pred, average='micro', zero_division=0))
    # print(accuracy_score(y_true, y_pred))

    print(cal_mean_acc(confusion_matrix.cpu().numpy()))
    print(cal_diag(C))

#   a = cal_rec_2(confusion_matrix.cpu().numpy())
#
