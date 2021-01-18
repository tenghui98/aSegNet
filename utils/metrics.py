import numpy as np


class Evaluator(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Stats(self):
        TN, FP, FN, TP = self.confusion_matrix[0][0], self.confusion_matrix[0][1], \
                         self.confusion_matrix[1][0], self.confusion_matrix[1][1]
        alpha = 0.00001
        recall = TP / (TP + FN + alpha)
        specficity = TN / (TN + FP + alpha)
        fpr = FP / (FP + TN + alpha)
        fnr = FN / (TP + FN + alpha)
        pbc = 100.0 * (FN + FP) / (TP + FP + FN + TN + alpha)
        precision = TP / ((TP + FP) + alpha)
        fmeasure = 2.0 * (recall * precision) / ((recall + precision) + alpha)
        mcc = (1.0 * ((TP * TN) - (FP * FN))) / np.sqrt((TP + FP) * (TP + FN) * (TN + FN) * (TN + FP) + alpha)
        pwc = (100.0 * (FP + FN)) / (TP + FP + TN + FN + alpha)

        return recall, precision, fmeasure

    def add_batch(self, gt_image, pred_image):
        assert gt_image.shape == pred_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pred_image)

    def _generate_matrix(self, gt_image, pred_image):
        idx = np.where((gt_image == 0) | (gt_image == 1))
        gt = gt_image[idx]
        pred = pred_image[idx]
        label = self.n_classes * gt.astype('int') + pred
        count = np.bincount(label, minlength=self.n_classes ** 2)
        confusion_matrix = count.reshape(self.n_classes, self.n_classes)
        return confusion_matrix

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
