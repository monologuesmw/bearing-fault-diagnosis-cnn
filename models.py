# -*- coding: utf-8 -*-
# @Time    : 2020/1/7 13:28
# @Author  : smw
# @Email   : monologuesmw@163.com
# @File    : models.py
# @Software: PyCharm

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pandas as pd
import os
import sys
import time
import datetime
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def run_few_shot_learning_fault_diagnosis(phase="TEST_ONLY", test_size=10, lr=2e-4, batch=40, epoch_siamese=50,
                                          filename="few_shot_learning_fault_diagnosis"):
    """
    基于wdcnn模型的Siamese网络进行轴承故障诊断
    :param phase:  由用户选择程序的阶段： TEST_ONLY： 只进行测试【使用本地保存的权重】； TRAIN_AND_TEST : 训练并测试【时间很久】
    :param test_size: 参与测试样本的个数  30条以内速度比较快， 最大为375  【出于对速度的考量】
    :param lr: 学习率  可设置 1e-1, 1e-2, 1e-3, 1e-4, 2e-4, 1e-5
    :param batch: 训练的batch_size 可设置 20, 40, 60  根据服务器内存的大小 可适当增长
    :param epoch_siamese: 训练代数   20代比较快， 再大 速度较慢， 但精度取决于
    :param filename:  结果保存名称
    :return:
    """
    # path created

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
    if phase == "TRAIN_AND_TEST":
        checkpoint_dir = ".\\checkpoint\\%s" % TIMESTAMP
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            save_path = checkpoint_dir
    elif phase == "TEST_ONLY":
        TIMESTAMP_LOCAL = "LOCAL_MODULE"
        checkpoint_dir = ".\\checkpoint\\%s"% TIMESTAMP_LOCAL
        save_path = checkpoint_dir
    else:
        raise ValueError

    checkpoint_dir = os.path.join(checkpoint_dir, filename+".ckpt")
    # checkpoint_dir = checkpoint_dir + "//"
    # Tensorboard
    # logdir = ".\\log\\%s" % TIMESTAMP
    # if not os.path.exists(logdir):
    #     os.makedirs(logdir)

    result_dir = ".\\result\\%s" % TIMESTAMP
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # load data
    data = np.load("Case_Western_Reserve_University_Bearing_fault_data.npz")
    x_train, x_test, y_train, y_test, n_classes, classes = data['arr_0'], data['arr_1'], data['arr_2'], \
                                                           data["arr_3"], data["arr_4"], data["arr_5"]

    few_shot_learning_obj = Few_Shot_Learning_Fault_diagnosis(lr=lr, batch_size=batch, epoch_siamese=epoch_siamese,
                                                                save_path=save_path, checkpoint_dir=checkpoint_dir)

    siamese_process_obj = Siamese_process(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, test_num=test_size)
    if phase == "TRAIN_AND_TEST":
        # train phase
        siamese_process_obj.train_and_test_one_shot(few_shot_learning_obj)

    # test phase
    prod_preds, shot_many_preds, shot_many_true = siamese_process_obj.test_many_shot(few_shot_learning_obj, shot_num=5, s="test")

    prod_preds = prod_preds.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    y_test = y_test[0:test_size]

    cm = confusion_matrix(y_test, prod_preds)
    labels_name = list(range(len(np.unique(y_test))))
    # plot confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
    plt.title("HAR Confusion Matrix")  # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(result_dir, "HAR_cm.png"), format="png")
    plt.show()

    # save result in order to have a look
    acc = np.equal(prod_preds, y_test)
    acc_ = np.mean(acc)
    acc_ = format(acc_, '.4f')
    acc_ = np.array([acc_])

    res = np.hstack((prod_preds, y_test))
    res_Dataframe = pd.DataFrame(res, columns=["predict", "y_label"])
    res_Dataframe.to_csv(os.path.join(result_dir, filename + ".csv"))

    np.savetxt(os.path.join(result_dir, "accuracy.txt"), acc_, fmt="%s")


class Few_Shot_Learning_Fault_diagnosis(object):
    def __init__(self, lr, batch_size, epoch_siamese, save_path, checkpoint_dir):
        self.lr = lr
        self.batch_size = batch_size
        self.epoch_siamese = epoch_siamese
        self.save_path = save_path
        self.checkpoint_dir = checkpoint_dir  # save model

        self.num_class = 5
        self.keep_prob = 0.5

        # inputs placeholder of base structure  [double: left and right]
        # this structure does not have output placeholder
        self.inputs_base_structure_left = tf.placeholder(dtype=tf.float32, shape=[None, 2048, 2], name="inputs_left")  # initial a inputs to siamese_network
        self.inputs_base_structure_right = tf.placeholder(dtype=tf.float32, shape=[None, 2048, 2], name="inputs_right")
        # self.inputs_raw = tf.reshape(self.inputs, shape=[-1, 2048, 2, 1])
        tf.summary.histogram("inputs_left", self.inputs_base_structure_left)
        tf.summary.histogram("inputs_right", self.inputs_base_structure_right)

        # inputs placeholder of rest structure  [double: left and right]
        # the inputs of this structure is base structure output
        # self.inputs_rest_structure_left = tf.placeholder(dtype=tf.float32, shape=[None, 100], name="rest_inputs_left")
        # self.inputs_rest_structure_right = tf.placeholder(dtype=tf.float32, shape=[None, 100], name="rest_input_right")
        # tf.summary.histogram("rest_inputs_left", self.inputs_rest_structure_left)
        # tf.summary.histogram("rest_inputs_right", self.inputs_rest_structure_right)

        # outputs placeholder of all structure   the output of Siamese Network is just a prob
        self.outputs_label = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.outputs_label_raw = tf.reshape(self.outputs_label, shape=[-1, 1])
        # outputs placeholder of wdcnn, the output of wdcnn is a one_hot label in order to description the fault class
        self.outputs_label_onehot = tf.placeholder(dtype=tf.float32, shape=[None, self.num_class])
        # self.outputs_label_onehot = tf.reshape(self.inputs_label, shape=[-1, self.num_class])

        with tf.variable_scope("Siamese_Network_Structure") as scope:
            # base structure output    delete reason: may cause base structure variables did not optimize because of divided!
            # self.siamese_base_output_left = self.siamese_base_structure(inputs=self.inputs_base_structure_left, reuse=False)
            # self.siamese_base_output_right = self.siamese_base_structure(inputs=self.inputs_base_structure_right, reuse=True)
            # final output
            self.siamese_output = self.siamese_network_structure()
            scope.reuse_variables()
            self.siamese_output_test = self.siamese_network_structure(s="test")


        self.loss_siam = tf.losses.sigmoid_cross_entropy(logits=self.siamese_output,
                                                        multi_class_labels=self.outputs_label_raw,
                                                        scope="loss_siam")
        tf.summary.scalar("lost_siam", self.loss_siam)
        self.optimizer_siam = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_siam)

    def siamese_base_structure(self, inputs, reuse):
        # left_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 2048, 2])
        # right_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 2048, 2])   在类内不能使用嵌套函数
        with slim.arg_scope([slim.conv1d], padding="same", activation_fn=slim.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.005)
                            ):
            net = slim.conv1d(inputs=inputs, num_outputs=16, kernel_size=64, stride=16, reuse=reuse, scope="conv_1")
            # tf.summary.histogram("conv_1", net)
            def_max_pool = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding="VALID", name="max_pool_2")
            net = def_max_pool(net)
            # tf.summary.histogram("max_pool_2", net)

            net = slim.conv1d(net, num_outputs=32, kernel_size=3, stride=1, reuse=reuse, scope="conv_3")
            # tf.summary.histogram("conv_3", net)
            def_max_pool = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding="VALID", name="max_pool_4")
            net = def_max_pool(net)
            # tf.summary.histogram("max_pool_4", net)

            net = slim.conv1d(net, num_outputs=64, kernel_size=2, stride=1, reuse=reuse, scope="conv_5")
            # tf.summary.histogram("conv_5", net)
            def_max_pool = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding="VALID", name="max_pool_6")
            net = def_max_pool(net)
            # tf.summary.histogram("max_pool_6", net)

            net = slim.conv1d(net, num_outputs=64, kernel_size=3, stride=1, reuse=reuse, scope="conv_7")
            # tf.summary.histogram("conv_7", net)
            def_max_pool = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding="VALID", name="max_pool_8")
            net = def_max_pool(net)
            # tf.summary.histogram("max_pool_8", net)

            net = slim.conv1d(net, num_outputs=64, kernel_size=3, stride=1, padding="VALID", reuse=reuse, scope="conv_9")
            # tf.summary.histogram("conv_9", net)
            def_max_pool = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding="VALID", name="max_pool_10")
            net = def_max_pool(net)
            # tf.summary.histogram("max_pool_10", net)

            net = slim.flatten(net, scope="flatten_11")
            # tf.summary.histogram("flatten_11", net)

            output_step_one = slim.fully_connected(net, num_outputs=100, activation_fn=tf.nn.sigmoid, reuse=reuse,
                                                   weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                   weights_regularizer=slim.l2_regularizer(0.005),
                                                   scope="fully_connected_12")
            # tf.summary.histogram("fully_connected_12", output_step_one)
        return output_step_one

    def siamese_network_structure(self, s="train"):
        if s=="train":
            # siamese_network_structure rest
            left_ouput = self.siamese_base_structure(inputs=self.inputs_base_structure_left, reuse=False)
        else:
            left_ouput = self.siamese_base_structure(inputs=self.inputs_base_structure_left, reuse=True)
        right_output = self.siamese_base_structure(inputs=self.inputs_base_structure_right, reuse=True)  # siam network two results

        L1_distance = tf.math.abs(left_ouput - right_output,
                                  name="L1_distance")  # two tensor result substract
        # tf.summary.histogram("L1_distance_13", L1_distance)
        net = slim.dropout(L1_distance, keep_prob=self.keep_prob, scope="dropout_14")
        # tf.summary.histogram("dropout_14", net)
        a = tf.Variable(tf.zeros([1]))
        if s =="train":
            prob_output = slim.fully_connected(net, num_outputs=1, activation_fn=tf.nn.sigmoid,
                                               weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                               weights_regularizer=slim.l2_regularizer(0.005), reuse=False,
                                               scope="fully_connected_15")
        else:
            # biases_initializer = slim.zero_initializer(ref=a),
            # biases_regularizer = slim.l2_regularizer(0.005),
            prob_output = slim.fully_connected(net, num_outputs=1, activation_fn=tf.nn.sigmoid,
                                               weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                               weights_regularizer=slim.l2_regularizer(0.005), reuse=True,
                                               scope="fully_connected_15")
        # tf.summary.histogram("fully_connected_15", prob_output)
        return prob_output


class Siamese_process(object):
    def __init__(self, x_train, x_test, y_train, y_test, test_num):
        # data prepare
        self.data = {"train": x_train, "test": x_test}
        self.labels = {"train": y_train, "test": y_test}

        self.classes = {"train": sorted(list(set(y_train))), "test": sorted(list(set(y_test)))}
        self.indices = {"train": [np.where(y_train == i)[0] for i in self.classes["train"]],
                        "test": [np.where(y_test == i)[0] for i in self.classes["test"]]}   # 获得各类的索引
        self.test_num = len(self.labels["test"])
        self.k = test_num

    def get_batch_for_train(self, batch_size=32, s="train"):
        """
        creat sample pairs for training stage
        :param batch_size: 一次训练的个数
        :param s: 训练
        :return: 一个batch训练样本输入， 一个batch训练样本输出， left input label
        """

        x_train = self.data[s]
        num_classes = len(self.classes[s])
        x_indices = self.indices[s]     # 不同类别的索引

        _, w, h = x_train.shape

        # randomly choice the number of batch_size from classes label
        categories = np.random.choice(num_classes, size=(batch_size,), replace=True)   # left input label
        # create a list for a batch of sample pairs
        pairs = [np.zeros((batch_size, w, h, 1)) for i in range(2)]
        # create a vector for a batch of label
        label = np.zeros((batch_size, 1))
        # make the vector of the label one half of 0, and another half of 1.
        # description diff class and same class
        boundary = batch_size // 2
        label[boundary:, 0] = 1    # true label
        for i in range(batch_size):
            # the class, one of the current sample pairs
            category = categories[i]
            # the number of current class
            n_examples = len(x_indices[category])
            if n_examples == 0:
                raise FileNotFoundError("the number of {} class examples is not found".format(i))
            # create a random int number [0, n_examples-1]
            idx = np.random.randint(0, n_examples)
            # current idx as the sample of left input
            pairs[0][i, :, :, :] = x_train[x_indices[category][idx]].reshape(w, h, 1)
            # judge i is the id of same class or diff class
            if i >= batch_size // 2:  # same class
                category_2 = category   # the class of right input is same of left input
                idx_2 = (idx + np.random.randint(1, n_examples)) % n_examples  # is different of idx
            else:  # diff class
                category_2 = (category + np.random.randint(1, num_classes)) % num_classes
                n_examples_2 = len(x_indices[category_2])
                if n_examples_2 == 0:
                    raise FileNotFoundError("the number of {} class examples is not found".format(i))
                idx_2 = np.random.randint(0, n_examples_2)
            pairs[1][i, :, :, :] = x_train[x_indices[category_2][idx_2]].reshape(w, h, 1)
        return pairs, label, categories

    def train_and_test_one_shot(self, diag_obj):
        batch = diag_obj.batch_size
        x_train = self.data["train"]
        total_number_train = x_train.shape[0]

        sess = tf.Session()
        # merge_summary_op = tf.summary.merge_all()
        # summary_writer = tf.summary.FileWriter(diag_obj.log_dir, sess.graph)

        saver = tf.train.Saver(max_to_keep=1)
        sess.run(tf.global_variables_initializer())

        print("Siamese is training!")
        for step in range(diag_obj.epoch_siamese):
            print("Epoch:%d" % step)
            avg_cost = 0
            total_batch = int(total_number_train // batch)
            for batch_step in range(total_batch):
                # load batch data,  categories is prepared for wdcnn
                inputs_pairs, label, categories = self.get_batch_for_train(batch_size=batch)
                categories_one_hot_label = sess.run(tf.one_hot(categories, depth=diag_obj.num_class))
                # new way
                _, loss = sess.run([diag_obj.optimizer_siam, diag_obj.loss_siam],feed_dict={diag_obj.inputs_base_structure_left:
                                                                                            inputs_pairs[0].reshape([batch, 2048, 2]),
                                                                                            diag_obj.inputs_base_structure_right:
                                                                                            inputs_pairs[1].reshape([batch, 2048, 2]),
                                                                                            diag_obj.outputs_label:
                                                                                            label,
                                                                                            diag_obj.outputs_label_onehot:
                                                                                            categories_one_hot_label})
                # # this way may cause base structure does not optimize
                # # firstly, create the outputs of the right and left branch
                # output_left_base = sess.run([diag_obj.siamese_base_output_left], feed_dict={diag_obj.inputs_base_structure_left:
                #                                                                              inputs_pairs[0].reshape([batch, 2048, 2])})
                # output_right_base = sess.run([diag_obj.siamese_base_output_right],feed_dict={diag_obj.inputs_base_structure_right:
                #                                                                              inputs_pairs[1].reshape([batch, 2048, 2])})
                # output_left_base = np.array(output_left_base)
                # output_right_base = np.array(output_right_base)
                # # Meanwhile, calculate loss and optimizer
                # _, loss = sess.run([diag_obj.optimizer_siam, diag_obj.loss_siam], feed_dict={diag_obj.inputs_rest_structure_left: output_left_base.reshape([batch, 100]),
                #                                                                    diag_obj.inputs_rest_structure_right: output_right_base.reshape([batch, 100]),
                #                                                                    diag_obj.outputs_label: label})

                avg_cost += loss / total_batch

                # summary_str = sess.run(merge_summary_op, feed_dict={diag_obj.inputs_base_structure_left: inputs_pairs[0].reshape([batch, 2048, 2]),
                #                                                     diag_obj.inputs_base_structure_right: inputs_pairs[1].reshape([batch, 2048, 2]),
                #                                                     diag_obj.outputs_label: label,
                #                                                     diag_obj.outputs_label_onehot: categories_one_hot_label
                #                                                     })
                # summary_writer.add_summary(summary_str, global_step=step)

                # print("Epoch:%d, batch: %d, avg_cost: %g" % (step, batch_step, avg_cost))
            print("Epoch:%d, avg_cost: %g" % (step, avg_cost))
            saver.save(sess, diag_obj.checkpoint_dir, global_step=step)
        sess.close()

    def train_wdcnn(self, diag_obj):
        batch = diag_obj.batch_size
        x_train = self.data["train"]
        total_number_train = x_train.shape[0]

        sess = tf.Session()
        merge_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(diag_obj.log_dir, sess.graph)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        print("wdcnn is training")
        for step in range(diag_obj.epoch_wdcnn):
            print("Epoch:%d" % step)
            avg_cost = 0
            total_batch = int(total_number_train // batch)
            for batch_num in range(total_batch):
                # load batch data,  categories is prepared for wdcnn
                inputs_pairs, label, categories = self.get_batch_for_train(batch_size=batch)
                categories = tf.one_hot(categories, depth=diag_obj.num_class)
                true_one_hot_label = sess.run([categories])
                true_one_hot_label = np.array(true_one_hot_label)
                _, loss = sess.run([diag_obj.optimizer_wdcnn, diag_obj.loss_wdcnn], feed_dict={diag_obj.outputs_label_onehot:
                                                                                               true_one_hot_label.reshape([batch, diag_obj.num_class]),
                                                                                               diag_obj.inputs_base_structure_left:
                                                                                               inputs_pairs[0].reshape([batch, 2048, 2])
                                                                                               })
                summary_str = sess.run(merge_summary_op, feed_dict={
                    diag_obj.inputs_base_structure_left: inputs_pairs[0].reshape([batch, 2048, 2]),
                    diag_obj.outputs_label_onehot: true_one_hot_label.reshape([batch, diag_obj.num_class]),
                    diag_obj.inputs_base_structure_right: inputs_pairs[1].reshape([batch, 2048, 2]),
                    diag_obj.outputs_label: label
                })
                summary_writer.add_summary(summary_str, global_step=step)

                avg_cost += loss / total_batch
            print("Epoch:%d, avg_cost: %g" % (step, avg_cost))

    def test_many_shot(self, diag_obj, shot_num=5, s="test"):
        # use test_one_shot2 to many times
        shot_many_preds = []
        shot_many_prods = []
        shot_many_true = []
        for i in range(shot_num):
            preds, prob_all = self.test_oneshot2(diag_obj, s="test")
            preds = np.array(preds)
            # prob_all = np.array(prob_all)
            shot_many_preds.append(preds[:, 0])
            shot_many_true.append(preds[:, 1])
            shot_many_prods.append(prob_all)
        pick = np.sum(shot_many_prods, axis=0)
        pick = pick.reshape([self.k, diag_obj.num_class])
        # rank_num = np.argmax(pick, axis=1)
        prod_preds = np.argmax(pick, axis=1)
        return prod_preds, shot_many_preds, shot_many_true

    def test_oneshot2(self, diag_obj, s="test"):
        x_inputs = self.data[s]
        y_inputs = self.labels[s]
        k = self.k   # the number of test data
        # k = 10
        prob_all = []
        preds = []
        for idx in range(k):  # test data cycle.  in order to create N way k shot sample, and predicts
            inputs, targets, categories = self.make_many_shot_task_data(idx=idx, s=s)
            # adjust input shape
            num_class, w, h, _ = inputs[0].shape
            inputs[0] = inputs[0].reshape([num_class, w, h])
            inputs[1] = inputs[1].reshape([num_class, w, h])
            prob = self.test_one_shot(diag_obj, inputs[0], inputs[1])
            preds.append([categories[np.argmax(prob)], categories[np.argmax(targets)]])  # save final result and true result
            prob_all.append(prob)   # save all result
        return preds, prob_all

    def make_many_shot_task_data(self, idx, s):
        # create N way k shot sample pairs
        # train data info  in order to construct the N way k shot sample
        x_train = self.data["train"]
        # label_train = self.labels["train"]
        indice_train = self.indices["train"]
        class_train = self.classes["train"]
        N = len(indice_train)    # the number of sample classes

        # test data and labels
        x_test = self.data[s]
        label_test = self.labels[s]
        _, w, h = x_test.shape  # (375, 2048, 2)

        test_image = np.asarray([x_test[idx]]*N).reshape([N, w, h, 1])   # left input   N个idx sample
        support_set = np.zeros((N, w, h))   # right input
        # create a sample for each classes
        for n in range(N):  # current class id
            support_set[n, :, :] = x_train[np.random.choice(indice_train[n], size=(1,), replace=False)]
        support_set = support_set.reshape((N, w, h, 1))   # create a (N, 1) sample N个类 一个样本

        current_true_label = class_train.index(label_test[idx])
        targets = np.zeros((N, 1))   # 真值样本
        targets[current_true_label, 0] = 1    # shape  (5, 1)

        pairs = [test_image, support_set]
        current_categories = class_train
        return pairs, targets, current_categories

    def test_one_shot(self, diag_obj, inputs_left, inputs_right):
        sess = tf.Session()
        saver = tf.train.Saver()
        module_file = tf.train.latest_checkpoint(diag_obj.save_path)
        # saver = tf.train.import_meta_graph("./checkpoint/2020-01-15T16-27-10/few_shot_learning_fault_diagnosis.cpkt-99.meta")
        saver.restore(sess, module_file)
        print("Currently is the testing stage!")
        # sess.run(tf.global_variables_initializer)
        probs = sess.run([diag_obj.siamese_output_test], feed_dict={diag_obj.inputs_base_structure_left:
                                                        inputs_left,
                                                        diag_obj.inputs_base_structure_right:
                                                        inputs_right,
                                                        })
        sess.close()
        return probs

if __name__ == '__main__':
    # run_few_shot_learning_fault_diagnosis(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
    #                                       sys.argv[5])
    run_few_shot_learning_fault_diagnosis(phase="TEST_ONLY", test_size=10, lr=2e-4, batch=40, epoch_siamese=50,
                                          filename="few_shot_learning_fault_diagnosis")
