import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter
import random

DISCOUNTING_FACTOR = 0.9

class ExperienceReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.mean = torch.zeros(10)
        self.cov = torch.eye(10) # Default batch size is 10

    def rmal_al(batch_x, batch_y, mean, covariance, u, t, budget):
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
        for i in range(len(batch_size)):
            # s_t = clf.predict(batch_x[i])
            al_policy = torch.exp(gaussian.log_prob(s_t))
            bernoulli = torch.distributions.bernoulli.Bernoulli(al_policy)
            a_i = bernoulli.sample()
            if (u/t < budget and a_i == 1):
                subset_x = subset_x.append(batch_x[i])
                subset_y = subset_y.append(batch_y[i])
                u += 1
        return subset_x, subset_y, u, t+batch_size
    
    # Validation set has to be decided
    def update_policy(mean,covariance, clf, train_set, batch, episodes, validation_set=VALIDATION_SET, learning_rate=1e-6):
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
        acc_old = accuracy(clf, validation_set)
        training_set = set(train_set)
        for e in range(episodes):
            #proxy_clf = clf
            random.shuffle(batch)
            log_probs = [] # don't know why this is used
            rewards = []
            gaussian = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance)
            for (z_t, y_t) in batch:
                s_t = clf.predict(z_t)
                al_policy = torch.exp(gaussian.log_prob(s_t))
                log_probs.append(al_policy)
                bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor(al_policy))
                a_t = bernoulli.sample()
                #log_probs.append(bernoulli.log_prob(a_t))
                r_t = torch.tensor(0)
                if (a_t == 1):
                    memory = memory.add((z_t, y_t))
                    proxy_clf = train_model(training_set | memory)
                    acc = accuracy(proxy_clf, validation_set)
                    # reward signal r_t which is subsequently used to update the agent
                    r_t = (acc-acc_old)/acc_old
                    clf = proxy_clf
                    acc_old = acc
                else:
                    pt = {(z_t,y_t)}
                    cf_clf = train_model(training_set | memory | pt)
                    acc_cf = accuracy(cf_clf, validation_set)
                    r_t = -(acc_cf-acc_old)/acc_old
                    # proxy_clf remains the same
                    # acc remains the same

                rewards.append(r_t)
            


            discounted_rewards = []
            for t in range(len(batch)):
                for i in range(len(batch)):
                    discounted_rewards.append(DISCOUNTING_FACTOR ** (i-t) * rewards[i])

            total_rewards = torch.sum(torch.tensor(discounted_rewards))
            # TODO: al_policy here should really be the parameters of the policy
            
            al_policy.backward()
            

            mean = torch.sum(
                mean, 
                torch.mul(
                    torch.mul(
                        learning_rate, 
                        gaussian.mean
                    )

                )
            )
            covariance = covariance + gaussian.covariance_matrix
            print(f"DEBUG: covariance: {covariance}")
        

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
                # active learning: filter batch
                if self.params.budget < 1.0:
                    # batch_x, batch_y, mean, covariance, u, t, budget
                    self.rmal_al(batch_x, batch_y, self.mean, self.cov, self.u, self.t, self.params.budget)
                    pass
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
                        # TODO call algorithm 2
                        pass

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

    