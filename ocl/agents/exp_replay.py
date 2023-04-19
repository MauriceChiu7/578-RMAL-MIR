import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter
import random
import copy

DISCOUNTING_FACTOR = 0.9
BASELINE = 0.75

class ExperienceReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.mean = torch.zeros(10)
        self.cov = torch.eye(10) # Default batch size is 10
        self.val_set = []
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

        gaussian = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance)
        for i in range(batch_size):
            batch_x[i] = maybe_cuda(torch.tensor(batch_x[i]), self.cuda)
            # clf is initially trained on dataset D0 I am not sure how you guys are implementung this part
            logits = self.model.forward(batch_x[i]) # makes predictions on the datapoint x_t, the prediction is the state s_t
            _, s_t = torch.max(logits, 1)
            al_policy = torch.exp(gaussian.log_prob(s_t))
            bernoulli = torch.distributions.bernoulli.Bernoulli(al_policy)
            a_i = bernoulli.sample()
            if (u/t < budget and a_i == 1):
                subset_x = subset_x.append(batch_x[i])
                subset_y = subset_y.append(batch_y[i])
                u += 1
            t += 1
        return subset_x, subset_y, u, t
    
    # Validation set has to be decided
    def accuracy(self, clf):
        x_train = [x[0] for x in self.val_set]
        y_train = [x[1] for x in self.val_set]
        x_train = maybe_cuda(x_train, self.cuda)
        y_train = maybe_cuda(y_train, self.cuda)
        logits = clf.forward(x_train)
        _, pred_label = torch.max(logits, 1)
        acc = (pred_label == y_train).sum().item() / y_train.size(0)
        return acc
    
    def update_policy(self, mean, covariance, batch, episodes=5, learning_rate=1e-6):
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
        acc_old = self.accuracy(self.model, self.val_set)
        
        for e in range(episodes):
            proxy_clf = copy.deepcopy(self.model)
            
            random.shuffle(batch)

            log_probs = []
            rewards = []

            gaussian = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance)
            
            for (z_t, y_t) in batch:

                z_t = maybe_cuda(z_t, self.cuda)
                # batch_y = maybe_cuda(batch_y, self.cuda)
                logits = self.model.forward(z_t) # makes predictions on the datapoint z_t
                _, s_t = torch.max(logits, 1)
                
                al_policy = gaussian.log_prob(s_t)
                # pi = torch.exp(al_policy)
                log_probs.append(al_policy) # storing the log probs here

                bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor(al_policy))
                a_t = bernoulli.sample()
                #log_probs.append(bernoulli.log_prob(a_t))
                r_t = torch.tensor(0, requires_grad=True)
                if (a_t == 1):
                    memory = memory.add((z_t, y_t))

                    new_set = memory
                    x_train_new = [x[0] for x in new_set]
                    y_train_new = [x[1] for x in new_set]
                    # proxy_clf = self.train_learner(x_train_new,y_train_new) # train_learner takes x and y inputs separately
                    # proxy_clf = copy.deepcopy(self.model)
                    logits = proxy_clf.forward(x_train_new)
                    l = self.criterion(logits, y_train_new)
                    self.opt.zero_grad()
                    l.backward()
                    self.opt.step()

                    acc = self.accuracy(proxy_clf, self.val_set)
                    # reward signal r_t which is subsequently used to update the agent
                    r_t = (acc-acc_old)/acc_old
                    self.model = copy.deepcopy(proxy_clf)
                    acc_old = acc
                else:
                    pt = {(z_t,y_t)}

                    new_set = memory | pt
                    x_train_new = [x[0] for x in new_set]
                    y_train_new = [x[1] for x in new_set]
                    # cf_clf = self.train_learner(x_train_new,y_train_new) # train_learner takes x and y inputs separately
                    cf_clf = copy.deepcopy(self.model)
                    logits = cf_clf.forward(x_train_new)
                    l = self.criterion(logits, y_train_new)
                    self.opt.zero_grad()
                    l.backward()
                    self.opt.step()
                    
                    acc_cf = self.accuracy(cf_clf, self.val_set)
                    r_t = -(acc_cf-acc_old)/acc_old
                    # proxy_clf remains the same
                    # acc remains the same

                rewards.append(r_t)
            

            m = 1
            total_loss = torch.tensor(0)
            for k in range(m):
                loss = torch.tensor(0)
                discounted_rewards_at_t = []
                for t in range(len(batch)):
                    discounted_rewards = []
                    for t_prime in range(t, len(batch)):
                        discounted_rewards.append(DISCOUNTING_FACTOR ** (t_prime-t) * rewards[t_prime])
                    discounted_rewards_at_t.append(torch.sum(torch.tensor(discounted_rewards)))
                    baseline_term_at_t = discounted_rewards_at_t - BASELINE
                    loss += torch.sum(torch.mul(log_probs[t], baseline_term_at_t))
                total_loss += loss
            total_loss = torch.div(total_loss, m)
            total_loss.backward()

            mean = torch.add(
                mean, 
                torch.mul(
                    learning_rate, 
                    mean.grad
                )
            )
            covariance = torch.add(
                covariance, 
                torch.mul(
                    learning_rate, 
                    covariance.grad
                )
            )
            print(f"DEBUG: covariance: {covariance}")
        return mean, covariance
    
    def train_learner(self, x_train, y_train, data_continuum):
        
        self.val_set = data_continuum.val_data()

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
                # active learning: filter batch
                if self.params.budget < 1.0:
                    # batch_x, batch_y, mean, covariance, u, t, budget
                    batch_x, batch_y, self.u, self.t = self.rmal_al(batch_x, batch_y, self.mean, self.cov, self.u, self.t, self.params.budget)

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

                    # active learning: update policy
                    if self.params.budget < 1.0:
                        batch = [(batch_x[j], batch_y[j]) for j in range(batch_x.size(0))]
                        self.mean, self.cov = self.update_policy(self.mean, self.cov, batch)
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

    
