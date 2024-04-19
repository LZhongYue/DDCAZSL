import os
import math
import glob
import json
import random
import argparse
import classifier as classifier2
from utils import *
import torch.nn as nn
import torch.optim as optim
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch.nn.functional as F
from time import gmtime, strftime
import torch.backends.cudnn as cudnn
from dataset_GBU import FeatDataLayer, DATA_LOADER
from sklearn.metrics.pairwise import cosine_similarity
from torch.autograd import Variable
import torch.autograd as autograd


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA2')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--image_embedding', default='res101', type=str)
parser.add_argument('--class_embedding', default='att', type=str)

parser.add_argument('--gen_nepoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate to train generater')
parser.add_argument('--weight_decay', type=float, default=3e-6, help='weight_decay')

parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--batchsize', type=int, default=256, help='input batch size')
parser.add_argument('--nSample', type=int, default=30000, help='number features to generate per class')
parser.add_argument('--num_coupling_layers', type=int, default=3, help='number of coupling layers')

parser.add_argument('--disp_interval', type=int, default=200)
parser.add_argument('--save_interval', type=int, default=10000)
parser.add_argument('--evl_interval', type=int, default=100)
parser.add_argument('--manualSeed', type=int, default=4436, help='manual seed')  # 12518
parser.add_argument('--input_dim', type=int, default=85, help='dimension of the global semantic vectors')

parser.add_argument('--prototype', type=float, default=4.1, help='weight of the prototype loss')
parser.add_argument('--cprototype', type=float, default=1, help='weight of the prototype loss')
parser.add_argument('--cons', type=float, default=0.01, help='weight of the prototype loss')
parser.add_argument('--f', type=float, default=1, help='weight of the prototype loss')
parser.add_argument('--pi', type=float, default=0.15, help='degree of the perturbation')
parser.add_argument('--dropout', type=float, default=0.0, help='probability of dropping a dimension'
                                                               'in the perturbation noise')
parser.add_argument('--ngh', type=int, default=1024, help='degree of the perturbation')
parser.add_argument('--nhF', type=int, default=2048, help='size of the hidden units comparator network F')

parser.add_argument('--zsl', default=False)
parser.add_argument('--clusters', type=int, default=3)
parser.add_argument('--gpu', default="2", help='index of GPU to use')

parser.add_argument('--embedSize', type=int, default=2048, help='size of embedding h')
parser.add_argument('--outzSize', type=int, default=512, help='size of non-liner projection z')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=85 , help='size of semantic features')

parser.add_argument('--map_lr', type=float, default=0.001, help='weight of the classification loss when learning G')
parser.add_argument('--cls_map_lr', type=float, default=0.01, help='weight of the score function when learning G')
parser.add_argument('--att_dec_lr', type=float, default=0.0001, help='weight of the classification loss when learning G')
parser.add_argument('--cls_dec_lr', type=float, default=0.0001, help='weight of the score function when learning G')

parser.add_argument('--ins_weight', type=float, default=0.001, help='weight of the classification loss when learning G')
parser.add_argument('--cls_weight', type=float, default=0.001, help='weight of the score function when learning G')
parser.add_argument('--nclass_all', type=int, default=50, help='number of all classes')
parser.add_argument('--nclass_seen', type=int, default=40, help='number of all classes')

parser.add_argument('--gammaD', type=int, default=10, help='weight on the W-GAN loss')
parser.add_argument('--gammaG', type=float, default=0.001, help='weight on the W-GAN loss')
parser.add_argument('--lr_d', type=float, default=0.0004, help='learning rate to train discriminator')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--critic_iter', type=int, default=2, help='critic iteration, following WGAN-GP')

opt = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ': ')))


def train():
    dataset = DATA_LOADER(opt)
    opt.C_dim = dataset.att_dim
    opt.X_dim = dataset.feature_dim
    opt.y_dim = dataset.ntrain_class
    out_dir = 'out/{}/s_d-{}_mask-{}_cl-{}_ns-{}_wd-{}_lr-{}_nS-{}_bs-{}_ps-{}'.format(opt.dataset, opt.input_dim,
                                                                                            opt.dropout,
                                                                                            opt.num_coupling_layers,
                                                                                            opt.nSample,
                                                                                            opt.weight_decay, opt.lr,
                                                                                            opt.nSample, opt.batchsize,
                                                                                            opt.pi)
    os.makedirs(out_dir, exist_ok=True)
    print("The output dictionary is {}".format(out_dir))

    log_dir = out_dir + '/log_{}.txt'.format(opt.dataset)
    with open(log_dir, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    dataset.feature_dim = dataset.train_feature.shape[1]
    data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.cpu().numpy(), opt)
    opt.niter = int(dataset.ntrain / opt.batchsize) * opt.gen_nepoch

    result_gzsl_soft = Result()
    flow = cINN(opt).cuda()
    netMap = Embedding_Net(opt).cuda()
    netD = Dec_Att(opt).cuda()
    classifier_att = classifier2.LINEAR_LOGSOFTMAX(opt.attSize, opt.nclass_all).cuda()
    netDis = Discriminator_cINN(opt).cuda()
    print(flow)
    classifier = classifier2.LINEAR_LOGSOFTMAX(opt.outzSize, opt.nclass_seen).cuda()
    cls_criterion = nn.NLLLoss()
    optimizer = optim.Adam(list(flow.trainable_parameters) , lr=opt.lr,weight_decay=opt.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.92, step_size=15)
    optimizerMap = optim.Adam(netMap.parameters(), lr=opt.map_lr, betas=(0.5, 0.999))
    cls_optimizer = optim.Adam(classifier.parameters(), lr=opt.cls_map_lr, betas=(0.5, 0.999))
    netD_optimizer = optim.Adam(netD.parameters(), lr=opt.att_dec_lr, betas=(0.5, 0.999))
    cls_att_optimizer = optim.Adam(classifier_att.parameters(), lr=opt.cls_dec_lr, betas=(0.05, 0.999))
    optimizerDis = optim.Adam(netDis.parameters(), lr=opt.lr_d, betas=(0.5, 0.999))
    mse = nn.MSELoss()
    with open(log_dir, 'a') as f:
        f.write('\n')
        f.write('GSMFlow Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')
    start_step = 0
    prototype_loss = 0
    x_mean = torch.from_numpy(dataset.tr_cls_centroid).cuda()
    iters = math.ceil(dataset.ntrain / opt.batchsize)
    for it in range(start_step, opt.niter + 1):

        blobs = data_layer.forward()
        feat_data = blobs['data']  # image data
        labels_numpy = blobs['labels'].astype(int)  # class labels
        labels = torch.from_numpy(labels_numpy.astype('int')).cuda()

        C = np.array([dataset.train_att[i, :] for i in labels])
        C = torch.from_numpy(C.astype('float32')).cuda()
        X = torch.from_numpy(feat_data).cuda()

        ############
        all_attr = torch.from_numpy(dataset.attribute).cuda()
        all_label = torch.arange(0, dataset.ntrain_class + dataset.ntest_class, dtype=torch.long).cuda()
        #############
        #训练判别器
        ##############
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        for p in netDis.parameters():  # unfreeze discrimator
            p.requires_grad = True
        for iter_d in range(opt.critic_iter):
            netDis.zero_grad()

            z = opt.pi *torch.randn(opt.batchsize, 2048).cuda()
            mask = torch.cuda.FloatTensor(2048).uniform_() > opt.dropout
            z = mask * z
            X_fake, akk = flow.reverse_sample(z, C)
            criticD_real = netDis(X, C)
            criticD_real = opt.gammaD * criticD_real.mean()
            criticD_real.backward(mone)

            criticD_fake = netDis(X_fake, C)
            criticD_fake = opt.gammaD * criticD_fake.mean()
            criticD_fake.backward(one)
            gradient_penalty = opt.gammaD * calc_gradient_penalty(netDis, X, X_fake, C)
            gradient_penalty.backward()

            optimizerDis.step()
        for p in netDis.parameters():  # freeze discrimator
            p.requires_grad = False
        #############
        #训练Map
        ##############
        netMap.zero_grad()
        classifier.zero_grad()
        netD.zero_grad()
        classifier_att.zero_grad()

        embed_real, outz_real = netMap(X)
        cls_out = classifier(outz_real)
        real_ins_contras_loss = cls_criterion(cls_out, labels)

        # Class LOSS
        emd_all = torch.cat((embed_real,C),1)
        dec_att = netD(emd_all)
        att_all = torch.cat((dec_att, all_attr), 0)
        cls_out_att = classifier_att(att_all)
        rec_att_loss = cls_criterion(cls_out_att, torch.cat((labels, all_label), 0))

        D_cost = real_ins_contras_loss + rec_att_loss

        D_cost.backward()
        optimizerMap.step()
        cls_optimizer.step()
        netD_optimizer.step()
        cls_att_optimizer.step()

        flow.zero_grad()
        z = opt.pi * torch.randn(opt.batchsize, 2048).cuda()
        mask = torch.cuda.FloatTensor(2048).uniform_() > opt.dropout
        z = mask * z
        X1 = X + z
        z_, log_jac_det = flow(X1, C)

        loss = opt.f * (torch.mean(z_ ** 2) / 2 - torch.mean(log_jac_det) / 2048)

        fake, _ = flow.reverse_sample(z, C)

        criticG_fake = netDis(fake, C).mean()
        G_cost = -criticG_fake * opt.gammaG

        embed_fake, outz_fake = netMap(fake)
        cls_out_fake = classifier(outz_fake)
        fake_ins_contras_loss = cls_criterion(cls_out_fake, labels)*opt.ins_weight

        dec_fake_att = netD(torch.cat((embed_fake,C),1))
        cls_out_fake_att = classifier_att(dec_fake_att)
        rec_fake_att_loss = opt.cls_weight * cls_criterion(cls_out_fake_att, labels)

        loss_all = fake_ins_contras_loss + loss + rec_fake_att_loss + G_cost
        loss_all.backward()
        optimizer.step()
        if it % iters == 0:
            lr_scheduler.step()
        if it % opt.disp_interval == 0 and it:
            log_text = 'Iter-[{}/{}]; loss: {:.3f}; '.format(it, opt.niter, loss.item())

            log_print(log_text, log_dir)

        if it % opt.evl_interval == 0 and it > 0:
            flow.eval()
            netMap.eval()

            gen_feat, gen_label = synthesize_feature(flow, dataset, opt)

            train_X = torch.cat((dataset.train_feature, gen_feat), 0)
            train_Y = torch.cat((dataset.train_label, gen_label + dataset.ntrain_class), 0)

            """ GZSL"""

            cls = classifier2.CLASSIFIER(opt, train_X, train_Y,netMap,opt.embedSize,  dataset, dataset.test_seen_feature,
                                         dataset.test_unseen_feature,
                                         dataset.ntrain_class + dataset.ntest_class, True, opt.classifier_lr, 0.5, 30,
                                         1200, True)

            result_gzsl_soft.update_gzsl(it, cls.acc_unseen, cls.acc_seen, cls.H)

            log_print("GZSL Softmax:", log_dir)
            log_print("U->T {:.2f}%  S->T {:.2f}%  H {:.2f}%  Best_H [{:.2f}% {:.2f}% {:.2f}% | Iter-{}]".format(
                cls.acc_unseen, cls.acc_seen, cls.H, result_gzsl_soft.best_acc_U_T, result_gzsl_soft.best_acc_S_T,
                result_gzsl_soft.best_acc, result_gzsl_soft.best_iter), log_dir)

            flow.train()
            netMap.train()

class cINN(nn.Module):
    '''cINN for class-conditional MNISt generation'''

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cinn = self.build_inn(opt)

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)

    def build_inn(self, opt):

        def subnet(ch_in, ch_out):
            return nn.Sequential(nn.Linear(ch_in, 2048),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(2048, ch_out)
                                 )

        cond = Ff.ConditionNode(opt.input_dim)
        nodes = [Ff.InputNode(2048)]
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))

        for k in range(opt.num_coupling_layers):
            # nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor': subnet, 'clamp': 1.0},
                                 conditions=cond))

        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)

    def forward(self, x, c):
        z, jac = self.cinn.forward(x, c, jac=True)
        return z, jac

    def reverse_sample(self, z, c):
        return self.cinn(z, c, rev=True)


class Embedding_Net(nn.Module):
    def __init__(self, opt):
        super(Embedding_Net, self).__init__()

        self.fc1 = nn.Linear(opt.resSize, opt.embedSize)
        self.fc2 = nn.Linear(opt.embedSize, opt.outzSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, features):
        embedding= self.relu(self.fc1(features))
        out_z = F.normalize(self.fc2(embedding), dim=1)
        return embedding,out_z

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Dec_Att(nn.Module):
    def __init__(self, opt):
        super(Dec_Att, self).__init__()
        self.fc1 = nn.Linear(opt.embedSize+opt.attSize, opt.nhF)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.nhF, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, input):
        h = self.lrelu(self.fc1(input))
        h = self.fc2(h)
        return h
class Discriminator_cINN(nn.Module):
    def __init__(self, opt):
        super(Discriminator_cINN, self).__init__()
        self.fc1 = nn.Linear(2048 + opt.input_dim, 4096)
        self.fc2 = nn.Linear(4096, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        return h

def calc_gradient_penalty(netD,real_data, fake_data, input_att):
    alpha = torch.rand(opt.batchsize, 1)
    alpha = alpha.expand(real_data.size())

    alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size())

    ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

if __name__ == "__main__":
    train()
