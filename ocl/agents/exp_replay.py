import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter
import random
import copy
import numpy as np

DISCOUNTING_FACTOR = 0.9
BASELINE = 0.75

class ExperienceReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.mean = torch.zeros(1)
        self.cov = torch.eye(1) # Default batch size is 10
        self.val_loaders = None
        self.u = 0
        self.t = 1

    def rmal_al(self, batch_x, batch_y, mean, covariance, u, t, budget):
        """
        Algorithm 1: RMAL-AL algorithm

        inputs:
        clf:            f               - the classifier
        batch_x:        {x_1, ..., x_T} - the batch of newly labeled samples
        batch_y:        {y_1, ..., y_T} - the batch of newly labeled samples
        al_policy:      pi_phi          - the current AL agent policy
        budget:         b               - the number of samples u_t selected for training until time t

        outputs:
        subset_x:       {x_1, ..., x_u} - the subset of x_1, ..., x_N selected by the active learner
        subset_y:       {y_1, ..., y_u} - the subset of y_1, ..., y_N selected by the active learner
        u:              u_t             - the number of samples selected for training until time t
        t:              t               - the number of samples seen by the active learner

        notes:
        batch_size:     T               - the number of samples required for retraining
        init_data:      D_0             - initial dataset
        """
        
        batch_size = len(batch_x)
        subset_x = []
        subset_y = []

        # gaussian = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance)
        gaussian = torch.distributions.normal.Normal(mean, covariance)
        # gaussian = maybe_cuda(gaussian, self.cuda)
        for i in range(batch_size):
            # print(f"self.cuda: {self.cuda}")
            # batch_x[i] = maybe_cuda(torch.tensor(batch_x[i]), self.cuda)
            # clf is initially trained on dataset D0 I am not sure how you guys are implementung this part
            logits = self.model.forward(batch_x) # makes predictions on the datapoint x_t, the prediction is the state s_t
            _, s_t = torch.max(logits, 1)
            s_t = s_t[i]
            # print(f"s_t: {s_t}")

            al_policy = torch.exp(gaussian.log_prob((s_t.to(device='cpu'))))
            bernoulli = torch.distributions.bernoulli.Bernoulli(al_policy)
            a_i = bernoulli.sample()
            if (u/t < budget and a_i == 1):
                subset_x.append(batch_x[i])
                subset_y.append(batch_y[i])
                u += 1
            t += 1
        # print(f"type_batch_x: {type(batch_x)}, batch_x: {batch_x}, type_subset_x: {type(subset_x)}, subset_x: {subset_x}")
        # batch_x is tensor, subset_x is list
        # exit()
        return subset_x, subset_y, u, t
    
    def accuracy(self, clf):
        test_loaders = self.val_loaders
        clf.eval()
        acc_array = np.zeros(len(test_loaders))
        if self.params.trick['ncm_trick'] or self.params.agent in ['ICARL', 'SCR', 'SCP']:
            exemplar_means = {}
            cls_exemplar = {cls: [] for cls in self.old_labels}
            buffer_filled = self.buffer.current_index
            for x, y in zip(self.buffer.buffer_img[:buffer_filled], self.buffer.buffer_label[:buffer_filled]):
                cls_exemplar[y.item()].append(x)
            for cls, exemplar in cls_exemplar.items():
                features = []
                # Extract feature for each exemplar in p_y
                for ex in exemplar:
                    feature = clf.features(ex.unsqueeze(0)).detach().clone()
                    feature = feature.squeeze()
                    feature.data = feature.data / feature.data.norm()  # Normalize
                    features.append(feature)
                if len(features) == 0:
                    mu_y = maybe_cuda(torch.normal(0, 1, size=tuple(clf.features(x.unsqueeze(0)).detach().size())), self.cuda)
                    mu_y = mu_y.squeeze()
                else:
                    features = torch.stack(features)
                    mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
                exemplar_means[cls] = mu_y
        with torch.no_grad():
            if self.params.error_analysis:
                error = 0
                no = 0
                nn = 0
                oo = 0
                on = 0
                new_class_score = AverageMeter()
                old_class_score = AverageMeter()
                correct_lb = []
                predict_lb = []
            for task, test_loader in enumerate(test_loaders):
                acc = AverageMeter()
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    batch_x = maybe_cuda(batch_x, self.cuda)
                    batch_y = maybe_cuda(batch_y, self.cuda)
                    if self.params.trick['ncm_trick'] or self.params.agent in ['ICARL', 'SCR', 'SCP']:
                        feature = clf.features(batch_x)  # (batch_size, feature_size)
                        for j in range(feature.size(0)):  # Normalize
                            feature.data[j] = feature.data[j] / feature.data[j].norm()
                        feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
                        means = torch.stack([exemplar_means[cls] for cls in self.old_labels])  # (n_classes, feature_size)

                        #old ncm
                        means = torch.stack([means] * batch_x.size(0))  # (batch_size, n_classes, feature_size)
                        means = means.transpose(1, 2)
                        feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)
                        dists = (feature - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)
                        _, pred_label = dists.min(1)
                        # may be faster
                        # feature = feature.squeeze(2).T
                        # _, preds = torch.matmul(means, feature).max(0)
                        correct_cnt = (np.array(self.old_labels)[
                                           pred_label.tolist()] == batch_y.cpu().numpy()).sum().item() / batch_y.size(0)
                    else:
                        logits = clf.forward(batch_x)
                        _, pred_label = torch.max(logits, 1)
                        correct_cnt = (pred_label == batch_y).sum().item()/batch_y.size(0)

                    if self.params.error_analysis:
                        correct_lb += [task] * len(batch_y)
                        for i in pred_label:
                            predict_lb.append(self.class_task_map[i.item()])
                        if task < self.task_seen-1:
                            # old test
                            total = (pred_label != batch_y).sum().item()
                            wrong = pred_label[pred_label != batch_y]
                            error += total
                            on_tmp = sum([(wrong == i).sum().item() for i in self.new_labels_zombie])
                            oo += total - on_tmp
                            on += on_tmp
                            old_class_score.update(logits[:, list(set(self.old_labels) - set(self.new_labels_zombie))].mean().item(), batch_y.size(0))
                        elif task == self.task_seen -1:
                            # new test
                            total = (pred_label != batch_y).sum().item()
                            error += total
                            wrong = pred_label[pred_label != batch_y]
                            no_tmp = sum([(wrong == i).sum().item() for i in list(set(self.old_labels) - set(self.new_labels_zombie))])
                            no += no_tmp
                            nn += total - no_tmp
                            new_class_score.update(logits[:, self.new_labels_zombie].mean().item(), batch_y.size(0))
                        else:
                            pass
                    acc.update(correct_cnt, batch_y.size(0))
                acc_array[task] = acc.avg()
        
        return acc_array

    # Validation set has to be decided
    # def accuracy(self, clf):


        
    #     print(f"val_set: {self.val_set}")
    #     x_train = [x[0] for x in self.val_set]
    #     y_train = [x[1] for x in self.val_set]
    #     x_train = torch.tensor(torch.stack(list(x_train)))
    #     y_train = torch.tensor(torch.stack(list(y_train)))
    #     x_train = maybe_cuda(x_train, self.cuda)
    #     y_train = maybe_cuda(y_train, self.cuda)
    #     logits = clf.forward(x_train)
    #     _, pred_label = torch.max(logits, 1)
    #     acc = (pred_label == y_train).sum().item() / y_train.size(0)
    #     return acc
    
    def update_policy(self, mean, covariance, batch_x, batch_y, episodes=5, learning_rate=1e-6):
        """
        Algorithm 2: Update Agent

        inputs:
        al_policy:      pi_phi          - the current AL agent policy
        clf:            f               - the trained classifier since last batch update
        train_set:      D               - the training set
        batch:          N               - the batch of newly labeled samples (z_1, y_1), ..., (z_T, y_T)
        episodes:       E               - the number of training episodes

        outputs:
        new_al_policy:  pi'             - the updated AL agent policy
        """
        # M <- {} - initialize the memory to be an empty set
        memory = set()
        
        
        # finding accuracy on validation set
        acc_old = self.accuracy(self.model)
        
        for e in range(episodes):
            proxy_clf = copy.deepcopy(self.model)
            ordering = list(range(len(batch_x)))
            
            # random.shuffle(batch)

            log_probs = []
            rewards = []

            gaussian = torch.distributions.normal.Normal(mean, covariance)
            
            # for (z_t, y_t) in batch:
            for i in ordering:

                # z_t = maybe_cuda(z_t, self.cuda)
                # batch_y = maybe_cuda(batch_y, self.cuda)
                logits = self.model.forward(batch_x) # makes predictions on the datapoint z_t
                _, s_t = torch.max(logits, 1)
                s_t = s_t[i]

                al_policy = gaussian.log_prob(s_t.to(device='cpu'))
                pi = torch.exp(al_policy)
                log_probs.append(al_policy) # storing the log probs here

                bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor(pi))
                a_t = bernoulli.sample()
                #log_probs.append(bernoulli.log_prob(a_t))
                r_t = torch.tensor(0.0)
                if (a_t == 1):
                    memory.add(i)

                    new_set = memory
                    x_train_new = [batch_x[j] for j in new_set]
                    y_train_new = [batch_y[j] for j in new_set]
                    # proxy_clf = self.train_learner(x_train_new,y_train_new) # train_learner takes x and y inputs separately
                    # proxy_clf = copy.deepcopy(self.model)
                    x_train_new = maybe_cuda(torch.tensor(torch.stack(x_train_new)), self.cuda)
                    y_train_new = maybe_cuda(torch.tensor(torch.stack(y_train_new)), self.cuda)
                    
                    logits = proxy_clf.forward(x_train_new)
                    l = self.criterion(logits, y_train_new)
                    self.opt.zero_grad()
                    l.backward()
                    self.opt.step()

                    acc = self.accuracy(proxy_clf)
                    # reward signal r_t which is subsequently used to update the agent
                    r_t = (acc-acc_old)/acc_old
                    self.model = copy.deepcopy(proxy_clf)
                    acc_old = acc
                else:
                    pt = {i}

                    new_set = memory | pt
                    x_train_new = [batch_x[j] for j in new_set]
                    y_train_new = [batch_y[j] for j in new_set]
                    
                    x_train_new = maybe_cuda(torch.tensor(torch.stack(x_train_new)), self.cuda)
                    y_train_new = maybe_cuda(torch.tensor(torch.stack(y_train_new)), self.cuda)
                    
                    cf_clf = copy.deepcopy(self.model)
                    logits = cf_clf.forward(x_train_new)
                    l = self.criterion(logits, y_train_new)
                    self.opt.zero_grad()
                    l.backward()
                    self.opt.step()
                    
                    acc_cf = self.accuracy(cf_clf)
                    r_t = -(acc_cf-acc_old)/acc_old
                    # proxy_clf remains the same
                    # acc remains the same

                rewards.append(r_t)
            
            # total_log_probs = torch.sum(torch.stack(log_probs))
            # baseline_term_at_t = [item - BASELINE for item in discounted_rewards_at_t]
            


            m = 1
            total_loss = torch.tensor(0.0)
            for k in range(m):
                loss = torch.tensor(0.0)
                discounted_rewards_at_t = []
                for t in range(len(batch_x)):
                    discounted_rewards = []
                    for t_prime in range(t, len(batch_x)):
                        discounted_rewards.append(DISCOUNTING_FACTOR ** (t_prime-t) * rewards[t_prime])
                    discounted_rewards_at_t.append(torch.sum(torch.tensor(discounted_rewards)))
                    baseline_term_at_t = [item - BASELINE for item in discounted_rewards_at_t]
                    loss += torch.sum(torch.mul(log_probs[t], torch.tensor(torch.stack(baseline_term_at_t))))
                total_loss += loss
            total_loss = torch.div(total_loss, m)
            # total_loss.backward()
            mean_grad = sum((log_probs[t]-mean)*baseline_term_at_t[t] for t in range(len(log_probs)))*(1/covariance)
            cov_grad = sum((((log_probs[t]-mean) ** 2) - (len(log_probs) * covariance)) * baseline_term_at_t[t] for t in range(len(log_probs)) ) / (2 * (covariance ** 2))

            

            mean = torch.add(
                mean, 
                torch.mul(
                    learning_rate, 
                    mean_grad
                )
            )
            covariance = torch.add(
                covariance, 
                torch.mul(
                    learning_rate, 
                    cov_grad
                )
            )
            # print(f"DEBUG: covariance: {covariance}")

            if covariance <= 0:
                covariance = 0.0001

        return mean, covariance
    
    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses_batch = AverageMeter()
        losses_mem = AverageMeter()
        acc_batch = AverageMeter()
        acc_mem = AverageMeter()

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                for j in range(self.mem_iters):
                    logits = self.model.forward(batch_x)
                    loss = self.criterion(logits, batch_y)
                    if self.params.trick['kd_trick']:
                        loss = 1 / (self.task_seen + 1) * loss + (1 - 1 / (self.task_seen + 1)) * \
                                   self.kd_manager.get_kd_loss(logits, batch_x)
                    if self.params.trick['kd_trick_star']:
                        loss = 1/((self.task_seen + 1) ** 0.5) * loss + \
                               (1 - 1/((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(logits, batch_x)
                    _, pred_label = torch.max(logits, 1)
                    correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
                    # update tracker
                    acc_batch.update(correct_cnt, batch_y.size(0))
                    losses_batch.update(loss, batch_y.size(0))
                    # backward
                    self.opt.zero_grad()
                    loss.backward()

                    # mem update
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    if mem_x.size(0) > 0:
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        mem_logits = self.model.forward(mem_x)
                        loss_mem = self.criterion(mem_logits, mem_y)
                        if self.params.trick['kd_trick']:
                            loss_mem = 1 / (self.task_seen + 1) * loss_mem + (1 - 1 / (self.task_seen + 1)) * \
                                       self.kd_manager.get_kd_loss(mem_logits, mem_x)
                        if self.params.trick['kd_trick_star']:
                            loss_mem = 1 / ((self.task_seen + 1) ** 0.5) * loss_mem + \
                                   (1 - 1 / ((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(mem_logits,
                                                                                                         mem_x)
                        # update tracker
                        losses_mem.update(loss_mem, mem_y.size(0))
                        _, pred_label = torch.max(mem_logits, 1)
                        correct_cnt = (pred_label == mem_y).sum().item() / mem_y.size(0)
                        acc_mem.update(correct_cnt, mem_y.size(0))

                        loss_mem.backward()

                    if self.params.update == 'ASER' or self.params.retrieve == 'ASER':
                        # opt update
                        self.opt.zero_grad()
                        combined_batch = torch.cat((mem_x, batch_x))
                        combined_labels = torch.cat((mem_y, batch_y))
                        combined_logits = self.model.forward(combined_batch)
                        loss_combined = self.criterion(combined_logits, combined_labels)
                        loss_combined.backward()
                        self.opt.step()
                    else:
                        self.opt.step()

                # update mem
                self.buffer.update(batch_x, batch_y)

                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                            .format(i, losses_batch.avg(), acc_batch.avg())
                    )
                    print(
                        '==>>> it: {}, mem avg. loss: {:.6f}, '
                        'running mem acc: {:.3f}'
                            .format(i, losses_mem.avg(), acc_mem.avg())
                    )
        self.after_train()

    # def train_learner(self, x_train, y_train, data_continuum, val_loaders):
    #     # test_loaders = setup_test_loader(data_continuum.test_data(), defaul_params)
    #     # self.val_set = data_continuum.val_data()
    #     self.val_loaders = val_loaders

    #     self.before_train(x_train, y_train)
    #     # set up loader
    #     train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
    #     train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
    #                                    drop_last=True)
    #     # set up model
    #     self.model = self.model.train()

    #     # setup tracker
    #     losses_batch = AverageMeter()
    #     losses_mem = AverageMeter()
    #     acc_batch = AverageMeter()
    #     acc_mem = AverageMeter()

    #     for ep in range(self.epoch):
    #         for i, batch_data in enumerate(train_loader):
    #             # batch update
    #             batch_x, batch_y = batch_data
    #             # active learning: filter batch
    #             batch_x = maybe_cuda(batch_x, self.cuda)
    #             batch_y = maybe_cuda(batch_y, self.cuda)

    #             if self.params.budget < 1.0 and i > 5:
    #                 # batch_x, batch_y, mean, covariance, u, t, budget
    #                 batch_x, batch_y, self.u, self.t = self.rmal_al(batch_x, batch_y, self.mean, self.cov, self.u, self.t, self.params.budget)

    #                 if len(batch_x) == 0:
    #                     continue
    #                 else:
    #                     batch_x = torch.tensor(torch.stack(batch_x))
    #                     batch_y = torch.tensor(torch.stack(batch_y))
    #                     batch_x = maybe_cuda(batch_x, self.cuda)
    #                     batch_y = maybe_cuda(batch_y, self.cuda)
    #             # batch_x = maybe_cuda(batch_x, self.cuda)
    #             # batch_y = maybe_cuda(batch_y, self.cuda)
    #             for j in range(self.mem_iters):
    #                 logits = self.model.forward(batch_x)
    #                 loss = self.criterion(logits, batch_y)
    #                 if self.params.trick['kd_trick']:
    #                     loss = 1 / (self.task_seen + 1) * loss + (1 - 1 / (self.task_seen + 1)) * \
    #                                self.kd_manager.get_kd_loss(logits, batch_x)
    #                 if self.params.trick['kd_trick_star']:
    #                     loss = 1/((self.task_seen + 1) ** 0.5) * loss + \
    #                            (1 - 1/((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(logits, batch_x)
    #                 _, pred_label = torch.max(logits, 1)
    #                 correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
    #                 # update tracker
    #                 acc_batch.update(correct_cnt, batch_y.size(0))
    #                 losses_batch.update(loss, batch_y.size(0))
    #                 # backward
    #                 self.opt.zero_grad()
    #                 loss.backward()

    #                 # active learning: update policy
    #                 if self.params.budget < 1.0 and i > 5:
    #                     # batch = [(batch_x[j], batch_y[j]) for j in range(batch_x.size(0))]
    #                     self.mean, self.cov = self.update_policy(self.mean, self.cov, batch_x, batch_y)
    #                 # mem update
    #                 mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
    #                 if mem_x.size(0) > 0:
    #                     mem_x = maybe_cuda(mem_x, self.cuda)
    #                     mem_y = maybe_cuda(mem_y, self.cuda)
    #                     mem_logits = self.model.forward(mem_x)
    #                     loss_mem = self.criterion(mem_logits, mem_y)
    #                     if self.params.trick['kd_trick']:
    #                         loss_mem = 1 / (self.task_seen + 1) * loss_mem + (1 - 1 / (self.task_seen + 1)) * \
    #                                    self.kd_manager.get_kd_loss(mem_logits, mem_x)
    #                     if self.params.trick['kd_trick_star']:
    #                         loss_mem = 1 / ((self.task_seen + 1) ** 0.5) * loss_mem + \
    #                                (1 - 1 / ((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(mem_logits,
    #                                                                                                      mem_x)
    #                     # update tracker
    #                     losses_mem.update(loss_mem, mem_y.size(0))
    #                     _, pred_label = torch.max(mem_logits, 1)
    #                     correct_cnt = (pred_label == mem_y).sum().item() / mem_y.size(0)
    #                     acc_mem.update(correct_cnt, mem_y.size(0))

    #                     loss_mem.backward()

    #                 if self.params.update == 'ASER' or self.params.retrieve == 'ASER':
    #                     # opt update
    #                     self.opt.zero_grad()
    #                     combined_batch = torch.cat((mem_x, batch_x))
    #                     combined_labels = torch.cat((mem_y, batch_y))
    #                     combined_logits = self.model.forward(combined_batch)
    #                     loss_combined = self.criterion(combined_logits, combined_labels)
    #                     loss_combined.backward()
    #                     self.opt.step()
    #                 else:
    #                     self.opt.step()

    #             # update mem
    #             self.buffer.update(batch_x, batch_y)

    #             if i % 100 == 1 and self.verbose:
    #                 print(
    #                     '==>>> it: {}, avg. loss: {:.6f}, '
    #                     'running train acc: {:.3f}'
    #                         .format(i, losses_batch.avg(), acc_batch.avg())
    #                 )
    #                 print(
    #                     '==>>> it: {}, mem avg. loss: {:.6f}, '
    #                     'running mem acc: {:.3f}'
    #                         .format(i, losses_mem.avg(), acc_mem.avg())
    #                 )
    #     self.after_train()

    
