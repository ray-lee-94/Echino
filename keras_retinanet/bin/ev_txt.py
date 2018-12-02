#coding=utf-8
import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks,yticks
import os
import cv2
import xlwt
import xlrd
import shutil

count = 0
count_error = 0
count50 = 0
count60 = 0
count70 = 0
count80 = 0
count90 = 0
count95 = 0
def evaluate(test_folder, test_image, txt_folder):

    true = txt_folder + '/test.txt'
    train = txt_folder + '/train.txt'
    source = test_image
    # test_folder = output_path
    out = test_folder + '/output.txt'
    performance = test_folder + '/performance.txt'
    # out_information = test_folder + '/output_information.txt'
    confusion_matrix_jpg = test_folder + '/confusion_matrix.jpg'
    IOU_JPG = test_folder + '/IOU.jpg'
    xls = test_folder + '/test.xls'
    prebox = test_folder + '/pre'
    grobox = test_folder + '/gro'
    fixbox = test_folder + '/fix'
    fixtrue = test_folder + '/fixtrue/'
    notfind = test_folder + '/notfind'
    notfind_true = test_folder + '/notfindtrue'
    finderror = test_folder + '/finderror/'
    others = finderror + "/others"
    Type = ['AE1', 'AE2', 'AE3', 'CE1', 'CE2', 'CE3', 'CE4', 'CE5', 'CL']
    if not os.path.exists(prebox):
        os.mkdir(prebox)
    if not os.path.exists(grobox):
        os.mkdir(grobox)
    if not os.path.exists(fixbox):
        os.mkdir(fixbox)
    if not os.path.exists(fixtrue):
        os.mkdir(fixtrue)
    if not os.path.exists(notfind):
        os.mkdir(notfind)
    if not os.path.exists(notfind_true):
        os.mkdir(notfind_true)
    if not os.path.exists(finderror):
        os.mkdir(finderror)
    if not os.path.exists(others):
        os.mkdir(others)
    for t1 in Type:
        for t2 in Type:
            if t1 != t2:
                fold = finderror + t1 + "_" + t2
                if not os.path.exists(fold):
                    os.mkdir(fold)
    for t1 in Type:
        for t2 in Type:
            if t1 == t2:
                fold = fixtrue + t1 + "_" + t2
                if not os.path.exists(fold):
                    os.mkdir(fold)
    dtrue = {}
    dout = {}
    dtrain = {}
    names = []
    ious = []

    plt.figure(1)
    plt.figure(2)
    p = open(performance, 'w')
    # fopen = open(out_information, 'r')
    # messsages = fopen.readlines()
    workbook = xlwt.Workbook(encoding='utf-8')
    xlsheet = workbook.add_sheet("result", cell_overwrite_ok=True)

    def ca_iou(x11, y11, x12, y12, x21, y21, x22, y22):
        p_area = (x12 - x11) * (y12 - y11)
        g_area = (x22 - x21) * (y22 - y21)
        x1 = max(x11, x21)
        y1 = max(y11, y21)
        x2 = min(x12, x22)
        y2 = min(y12, y22)
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area = w * h
        if area <= 0:
            iou = 0
        else:
            iou = area / (p_area + g_area - area)
        return iou

    def IOU_(iou):
        global count, count_error, count50, count60, count70, count80, count90, count95
        count = count + 1
        if (iou < 0):
            print("error!!")
        elif (iou < 0.5):
            count_error += 1
        elif (iou < 0.6):
            count50 = count50 + 1
        elif (iou < 0.7):
            count60 = count60 + 1
        elif (iou < 0.8):
            count70 = count70 + 1
        elif (iou < 0.9):
            count80 = count80 + 1
        elif (iou < 0.95):
            count90 = count90 + 1
        elif (iou <= 1):
            count95 += 1
        else:
            print("error!")
        ious.append(iou)

    def drawbox():
        with open(true) as tr:
            trs = tr.readlines()
            for line in trs:
                name, tp, x1, y1, x2, y2, *_ = line.split(' ')
                srpath = os.path.join(source, name)
                if os.path.exists(os.path.join(grobox, name)):
                    srpath = os.path.join(grobox, name)
                sr = cv2.imread(srpath)
                cv2.rectangle(sr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                cv2.putText(sr, tp, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
                cv2.imwrite(os.path.join(grobox, name), sr)
        with open(out) as pre:
            pres = pre.readlines()
            for line in pres:
                name, tp, x1, y1, x2, y2, score = line.split(' ')
                srpath = os.path.join(source, name)
                if os.path.exists(os.path.join(prebox, name)):
                    srpath = os.path.join(prebox, name)
                sr = cv2.imread(srpath)
                cv2.rectangle(sr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                cv2.putText(sr, tp, (int(x2), int(y1) + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                cv2.putText(sr, str(score), (int(x2), int(y1) - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                cv2.imwrite(os.path.join(prebox, name), sr)

                gropath = os.path.join(grobox, name)
                if os.path.exists(os.path.join(fixbox, name)):
                    gropath = os.path.join(fixbox, name)
                sr = cv2.imread(gropath)
                cv2.rectangle(sr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                cv2.putText(sr, tp, (int(x2), int(y1) + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                cv2.putText(sr, str(score), (int(x2), int(y1) - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                cv2.imwrite(os.path.join(fixbox, name), sr)

        files = os.listdir(source)
        for file in files:
            if not os.path.exists(os.path.join(prebox, file)):
                shutil.copyfile(os.path.join(source, file), os.path.join(notfind, file))
                shutil.copyfile(os.path.join(grobox, file), os.path.join(notfind_true, file))

    def class_True(tp1, tp2, name):
        cla = fixtrue + tp1 + "_" + tp2
        shutil.copyfile(os.path.join(fixbox, name), os.path.join(cla, name))

    def misclass(tp1, tp2, name):
        sim = finderror + tp1 + "_" + tp2
        shutil.copyfile(os.path.join(fixbox, name), os.path.join(sim, name))

    def performance():
        TP = {}  # True Positive
        FP = {}  # False Positive
        TN = {}  # True Negative
        FN = {}  # False Negative
        F1 = {}  # F1 = 2 * P * R / (P + R) ##（查准率 Precision）P = TP/(TP+FP) ##（查全率/召回率 Recall ）R = TP/(TP + FN)
        TPR = {}  # sensitivity  /Recall
        PRECISION = {}
        TPR_average = 0
        PRECISION_average = 0
        F1_average = 0
        flag = 0
        with open(true) as t:
            for line in t:
                name, tp, *_ = line.split(' ')
                if tp not in dtrue:
                    dtrue[tp] = 1
                else:
                    dtrue[tp] = dtrue[tp] + 1
        with open(out) as t:
            for line in t:
                name, tp, *_ = line.split(' ')
                names.append(name)
                if tp not in dout:
                    dout[tp] = 1
                else:
                    dout[tp] = dout[tp] + 1
        with open(train) as t:
            for line in t:
                name, tp, *_ = line.split(' ')
                if tp not in dtrain:
                    dtrain[tp] = 1
                else:
                    dtrain[tp] = dtrain[tp] + 1
        train_count = sum(dtrain[k] for k in sorted(dtrain.keys()))
        truth_count = sum(dtrue[k] for k in sorted(dtrue.keys()))
        pred_count = sum(dout[k] for k in sorted(dout.keys()))

        for key in dout.keys():
            with open(true) as t:
                for linet in t:
                    name1, tp1, x11, y11, x12, y12, *_ = linet.split(' ')
                    if name1 in names:
                        if key == tp1:
                            with open(out) as pre:
                                for linep in pre:
                                    name2, tp2, x21, y21, x22, y22, *_ = linep.split(' ')
                                    if name1 == name2:
                                        iou = ca_iou(float(x11), float(y11), float(x12), float(y12), float(x21),
                                                     float(y21), float(x22), float(y22))
                                        if iou > 0.4:
                                            IOU_(iou)
                                            if tp1 == tp2:  #
                                                if key not in TP:
                                                    TP[key] = 1
                                                else:
                                                    TP[key] = TP[key] + 1
                                                class_True(tp1, tp2, name1)
                                            else:
                                                if key not in FN:
                                                    FN[key] = 1
                                                else:
                                                    FN[key] = FN[key] + 1
                                                misclass(tp1, tp2, name1)
                        if key != tp1:
                            with open(out) as pre:
                                for linep in pre:
                                    name2, tp2, x21, y21, x22, y22, *_ = linep.split(' ')
                                    if name1 == name2:
                                        iou = ca_iou(float(x11), float(y11), float(x12), float(y12), float(x21),
                                                     float(y21), float(x22), float(y22))
                                        if iou > 0.4:
                                            if key != tp2:
                                                if key not in TN:
                                                    TN[key] = 1
                                                else:
                                                    TN[key] = TN[key] + 1
                                            else:
                                                if key not in FP:
                                                    FP[key] = 1
                                                else:
                                                    FP[key] = FP[key] + 1
                                                    # misclass(tp1, tp2, name1)

                    else:
                        if key == tp1:
                            if key not in FN:
                                FN[key] = 1
                            else:
                                FN[key] = FN[key] + 1
                        else:
                            if key not in TN:
                                TN[key] = 1
                            else:
                                TN[key] = TN[key] + 1
        with open(out) as pre:
            for linep in pre:
                name1, tp1, x11, y11, x12, y12, *_ = linep.split(' ')
                flag = 0
                with open(true) as t:
                    for linet in t:
                        name2, tp2, x21, y21, x22, y22, *_ = linet.split(' ')
                        if name1 == name2:
                            iou = ca_iou(float(x11), float(y11), float(x12), float(y12), float(x21), float(y21),
                                         float(x22), float(y22))
                            if iou > 0.4:
                                flag = 1
                                break
                    if flag == 0:
                        iou = 0
                        IOU_(iou)
                        if tp1 not in FP:
                            FP[tp1] = 1
                        else:
                            FP[tp1] = FP[tp1] + 1
                        shutil.copyfile(os.path.join(fixbox, name1), os.path.join(others, name1))
        for key in dout.keys():
            if key not in TP:
                TP[key] = 0
            if key not in FP:
                FP[key] = 0
            if key not in TN:
                TN[key] = 0
            if key not in FN:
                FN[key] = 0
        for t1 in Type:
            for t2 in Type:
                if t1 != t2:
                    fold_ = finderror + t1 + "_" + t2
                    if not os.listdir(fold_):
                        os.rmdir(fold_)
        TP_count = sum(TP[k] for k in sorted(TP.keys()))
        FP_count = sum(FP[k] for k in sorted(FP.keys()))
        TN_count = sum(TN[k] for k in sorted(TN.keys()))
        FN_count = sum(FN[k] for k in sorted(FN.keys()))
        print("***********************  Performance  *********************", file=p)
        p.writelines("first line: train     ///     train_count: ")
        print(train_count, file=p)
        print([(k, dtrain[k]) for k in sorted(dtrain.keys())], file=p)
        p.writelines("second line: ground truth     ///     truth_count: ")
        print(truth_count, file=p)
        print([(k, dtrue[k]) for k in sorted(dtrue.keys())], file=p)
        p.writelines("third line: predictions     ///     pred_count: ")
        print(pred_count, file=p)
        print([(k, dout[k]) for k in sorted(dout.keys())], file=p)
        print(["TP", [(k, TP[k]) for k in sorted(TP.keys())]], file=p)
        print(["FP", [(k, FP[k]) for k in sorted(FP.keys())]], file=p)
        print(["TN", [(k, TN[k]) for k in sorted(TN.keys())]], file=p)
        print(["FN", [(k, FN[k]) for k in sorted(FN.keys())]], file=p)

        for key in sorted(dout.keys()):
            TPR[key] = 1.0 * TP[key] / (TP[key] + FN[key])
            PRECISION[key] = 1.0 * TP[key] / (TP[key] + FP[key])
            TPR_average += 1.0 * TPR[key] * dtrue[key] / truth_count
            PRECISION_average += 1.0 * PRECISION[key] * dtrue[key] / truth_count
        print([[(k, TPR[k]) for k in sorted(TPR.keys())], "recall or true positive rate"], file=p)
        print([[(k, PRECISION[k]) for k in sorted(PRECISION.keys())], "precision"], file=p)

        for key in dout.keys():
            F1[key] = 2.0 * PRECISION[key] * TPR[key] / (PRECISION[key] + TPR[key])
            F1_average += 1.0 * F1[key] * dtrue[key] / truth_count
        print([[(k, F1[k]) for k in sorted(F1.keys())], "F1 score"], file=p)
        print(TPR_average)
        print(PRECISION_average)
        print((F1_average))

        xlsheet.write_merge(9, 9, 12, 12, train_count)
        xlsheet.write_merge(10, 10, 12, 12, truth_count)
        xlsheet.write_merge(11, 11, 12, 12, pred_count)
        xlsheet.write_merge(12, 12, 12, 12, TP_count)
        xlsheet.write_merge(13, 13, 12, 12, FP_count)
        xlsheet.write_merge(14, 14, 12, 12, TN_count)
        xlsheet.write_merge(15, 15, 12, 12, FN_count)
        xlsheet.write_merge(16, 16, 12, 12, TPR_average)
        xlsheet.write_merge(17, 17, 12, 12, PRECISION_average)
        xlsheet.write_merge(18, 18, 12, 12, F1_average)

        i = 0
        for key in sorted(dtrain.keys()):
            xlsheet.write(9, i + 3, dtrain[key])
            i = i + 1
        i = 0
        for key in sorted(dtrue.keys()):
            xlsheet.write(10, i + 3, dtrue[key])
            i = i + 1
        i = 0
        for key in sorted(dout.keys()):
            xlsheet.write(11, i + 3, dout[key])
            i = i + 1
        i = 0
        for key in sorted(TP.keys()):
            xlsheet.write(12, i + 3, TP[key])
            i = i + 1
        i = 0
        for key in sorted(FP.keys()):
            xlsheet.write(13, i + 3, FP[key])
            i = i + 1
        i = 0
        for key in sorted(TN.keys()):
            xlsheet.write(14, i + 3, TN[key])
            i = i + 1
        i = 0
        for key in sorted(FN.keys()):
            xlsheet.write(15, i + 3, FN[key])
            i = i + 1
        i = 0
        for key in sorted(TPR.keys()):
            xlsheet.write(16, i + 3, TPR[key])
            i = i + 1
        i = 0
        for key in sorted(PRECISION.keys()):
            xlsheet.write(17, i + 3, PRECISION[key])
            i = i + 1
        i = 0
        for key in sorted(F1.keys()):
            xlsheet.write(18, i + 3, F1[key])
            i = i + 1
        i = 0

    def drawIOU():
        ious.sort()
        plt.figure(1)
        plt.plot(ious, np.arange(count))
        plt.title("IOU")
        plt.savefig(IOU_JPG)
        # plt.show()
        print("**********************  IOU  **********************", file=p)
        print(["all in count ", count], file=p)
        print(["0<iou<0.5 ", count_error / count], file=p)
        print(["0.5<=iou<0.6 ", count50 / count], file=p)
        print(["0.6<=iou<0.7 ", count60 / count], file=p)
        print(["0.7<=iou<0.8 ", count70 / count], file=p)
        print(["0.8<=iou<0.9 ", count80 / count], file=p)
        print(["0.9<=iou<0.95 ", count90 / count], file=p)
        print(["0.95<=iou<=1  ", count95 / count], file=p)
        IOU = [count_error / count, count50 / count, count60 / count, count70 / count, count80 / count, count90 / count,
               count95 / count]
        xlsheet.write_merge(22, 22, 3, 3, count)
        for i in range(len(IOU)):
            xlsheet.write(24 + i, 3, IOU[i])
            i = i + 1
        i = 0
        print("**********************  IOU_detection  **********************", file=p)
        truth_count = sum(dtrue[k] for k in sorted(dtrue.keys()))
        print(["all in annotations ", truth_count], file=p)
        print([" iou>=0.5 ", (count50 + count60 + count70 + count80 + count90 + count95) / truth_count], file=p)
        print([" iou>=0.6 ", (count60 + count70 + count80 + count90 + count95) / truth_count], file=p)
        print([" iou>=0.7 ", (count70 + count80 + count90 + count95) / truth_count], file=p)
        print([" iou>=0.8 ", (count80 + count90 + count95) / truth_count], file=p)
        print([" iou>=0.9 ", (count90 + count95) / truth_count], file=p)
        print([" iou>=0.95", count95 / truth_count], file=p)

        IOU_detection = [(count50 + count60 + count70 + count80 + count90 + count95) / truth_count,
                         (count60 + count70 + count80 + count90 + count95) / truth_count,
                         (count70 + count80 + count90 + count95) / truth_count,
                         (count80 + count90 + count95) / truth_count,
                         (count90 + count95) / truth_count,
                         count95 / truth_count
                         ]
        xlsheet.write_merge(48, 48, 3, 3, truth_count)
        for i in range(len(IOU_detection)):
            xlsheet.write(50 + i, 3, IOU_detection[i])
            i = i + 1
        i = 0

    def confusion_matrix():
        cm = {}
        cmatrix = np.zeros([9, 9])
        with open(true) as t:
            for linet in t:
                name1, tp1, *_ = linet.split(' ')
                if name1 in names:
                    with open(out) as pre:
                        for linep in pre:
                            name2, tp2, *_ = linep.split(' ')
                            if name1 == name2:
                                if tp1 + tp2 not in cm:
                                    cm[tp1 + tp2] = 1
                                else:
                                    cm[tp1 + tp2] += 1
        for index1, key1 in enumerate(Type):
            for index2, key2 in enumerate(Type):
                if key1 + key2 not in cm:
                    cm[key1 + key2] = 0
                cmatrix[index1, index2] = cm[key1 + key2]
        print("***********************  confusion matrix  *********************", file=p)
        print(cmatrix, file=p)
        plt.figure(2)
        figure, ax = plt.subplots()
        xticks(np.arange(9), Type)
        yticks(np.arange(9), Type)
        ax.xaxis.tick_top()
        plt.text(2, 9, 'confusion_matrix')
        plt.imshow(cmatrix.astype(np.uint8), cmap=plt.cm.gray)
        # plt.show()
        plt.savefig(confusion_matrix_jpg)

        for i in range(len(cmatrix[0])):
            for j in range(len(cmatrix[1])):
                xlsheet.write(i + 36, j + 2, cmatrix[i, j])

    def els():
        xlsheet.write_merge(1, 4, 1, 11, '包虫病测试结果')
        xlsheet.write_merge(6, 6, 1, 2, '表1：分类结果')
        xlsheet.write_merge(21, 21, 1, 2, '表2：IOU统计')
        xlsheet.write_merge(47, 47, 1, 2, '表4：IOU_detection')
        xlsheet.write_merge(48, 48, 1, 2, 'number of annotations:')
        xlsheet.write_merge(22, 22, 1, 2, '测试结果label数量：')
        xlsheet.write_merge(33, 33, 1, 2, '表3：混淆矩阵')
        xlsheet.write_merge(8, 8, 12, 12, 'Total/average')
        xlsheet.write_merge(21, 21, 6, 6, '注')
        xlsheet.write_merge(23, 23, 6, 7, 'RECALL=')
        xlsheet.write_merge(23, 23, 8, 9, 'TP/(TP+FN)')
        xlsheet.write_merge(25, 25, 6, 7, 'PRECISION=')
        xlsheet.write_merge(25, 25, 8, 9, 'TP/(TP+FP)')
        xlsheet.write_merge(27, 27, 6, 7, 'F1 score')
        xlsheet.write_merge(27, 27, 8, 9, '2*P*R/(P+R)')
        xlsheet.write_merge(29, 29, 6, 12, 'train、test、prediction、TP、FP、TN、FN(最后一列为该行求和)')
        xlsheet.write_merge(31, 31, 6, 12, 'RECALL/TPR、PRECISION、F1 SOCRE(最后一列为该行加权平均)')
        items = ['train', 'ground true', 'prediction', 'TP', 'FP', 'TN', 'FN', 'RECALL/TPR', 'PRECISION', 'F1 SOCRE']
        iou = ['0<=IOU<0.5', '0.5<=IOU<0.6', '0.6<=IOU<0.7', '0.7<=IOU<0.8', '0.8<=IOU<0.9', '0.9<=IOU<0.95',
               '0.95<=IOU<=1']
        iou_detection = ['iou>=0.5', 'iou>=0.6', 'iou>=0.7', 'iou>=0.8', 'iou>=0.9', 'iou>=0.95']
        for i in range(len(Type)):
            xlsheet.write(8, i + 3, Type[i])
        for i in range(len(Type)):
            xlsheet.write(35, i + 2, Type[i])
        for i in range(len(Type)):
            xlsheet.write(36 + i, 1, Type[i])
        for i in range(len(items)):
            xlsheet.write_merge(i + 9, i + 9, 1, 2, items[i])
        for i in range(len(iou)):
            xlsheet.write_merge(i + 24, i + 24, 1, 2, iou[i])
        for i in range(len(iou_detection)):
            xlsheet.write_merge(i + 50, i + 50, 1, 2, iou_detection[i])
        i = 0
        # for message in messsages:
        #     xlsheet.write_merge(58 + i, 58 + i, 1, 10, message)
        #     i = i + 1
        workbook.save(xls)
    drawbox()
    performance()
    drawIOU()
    confusion_matrix()
    els()
evaluate('rtn_center_23',test_image='/data/wen/Dataset/data_maker/COCO_maker/new_data/test',
         txt_folder='/data/wen/Dataset/data_maker/COCO_maker/new_data/')