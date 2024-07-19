import os, sys, argparse, re
import random, time
import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from config import *
from utils import *
import utils.helpers as helpers
from models.model import refinenet, model_init
import csv
import pdb
import time
from tqdm import tqdm
#from tqdm import tqdm_notebook as tqdm
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Full Pipeline Training')

    # Dataset
    parser.add_argument('-d', '--train-dir', type=str, default=TRAIN_DIR,
                        help='Path to the training set directory.')
    parser.add_argument('--train-dir_va', type=str, default=TRAIN_DIR_va,
                        help='Path to the train set va directory.')
    parser.add_argument('--val-dir', type=str, default=VAL_DIR,
                        help='Path to the validation set directory.')
    parser.add_argument('--test-dir', type=str, default=TEST_DIR,
                        help='Path to the validation set directory.')
    parser.add_argument('--train-list', type=str, default=TRAIN_LIST,
                        help='Path to the training set list.')
    parser.add_argument('--train-list_va', type=str, default=TRAIN_LIST_va,
                        help='Path to the training set va list.')
    parser.add_argument('--val-list', type=str, default=VAL_LIST,
                        help='Path to the validation set list.')
    parser.add_argument('--test-list', type=str, default=TEST_LIST,
                        help='Path to the validation set list.')
    parser.add_argument('--shorter-side', type=int, default=SHORTER_SIDE,
                        help='Shorter side transformation.')
    parser.add_argument('--crop-size', type=int, default=CROP_SIZE,
                        help='Crop size for training,')
    parser.add_argument('--input-size', type=int, default=RESIZE_SIZE,
                        help='Final input size of the model')
    parser.add_argument('--normalise-params', type=list, default=NORMALISE_PARAMS,
                        help='Normalisation parameters [scale, mean, std],')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size to train the segmenter model.')
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS,
                        help="Number of workers for pytorch's dataloader.")
    parser.add_argument('--num-classes', type=int, default=NUM_CLASSES,
                        help='Number of output classes for each task.')
    parser.add_argument('--low-scale', type=float, default=LOW_SCALE,
                        help='Lower bound for random scale')
    parser.add_argument('--high-scale', type=float, default=HIGH_SCALE,
                        help='Upper bound for random scale')
    parser.add_argument('--ignore-label', type=int, default=IGNORE_LABEL,
                        help='Label to ignore during training')

    # Encoder
    parser.add_argument('--enc', type=str, default=ENC,
                        help='Encoder net type.')
    parser.add_argument('--enc-pretrained', type=bool, default=ENC_PRETRAINED,
                        help='Whether to init with imagenet weights.')
    # General
    parser.add_argument('--name', default='', type=str,
                        help='model name')
    parser.add_argument('--gpu', type=int, nargs='+', default=[1],
                        help='select gpu.')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='If true, only validate segmentation.')
    parser.add_argument('--freeze-bn', type=bool, nargs='+', default=FREEZE_BN,
                        help='Whether to keep batch norm statistics intact.')
    parser.add_argument('--num-epoch', type=int, nargs='+', default=NUM_SEGM_EPOCHS,
                        help='Number of epochs to train for segmentation network.')
    parser.add_argument('--random-seed', type=int, default=RANDOM_SEED,
                        help='Seed to provide (near-)reproducibility.')
    parser.add_argument('-c', '--ckpt', default='model', type=str, metavar='PATH',
                        help='path to save checkpoint (default: model)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--val-every', type=int, default=VAL_EVERY,
                        help='How often to validate current architecture.')
    parser.add_argument('--print-network', action='store_true', default=True,
                        help='Whether print newtork paramemters.')
    parser.add_argument('--print-loss', action='store_true', default=True,
                        help='Whether print losses during training.')
    parser.add_argument('--save-image', type=int, default=100,
                        help='Number to save images during evaluating, -1 to save all.')
    parser.add_argument('-i', '--input', default=['RGB','FLOW','MEL','DELTA'], type=str, nargs='+',
                        help='input type (image, depth)')

    # Optimisers
    parser.add_argument('--lr-enc', type=float, nargs='+', default=LR_ENC,
                        help='Learning rate for encoder.')
    parser.add_argument('--lr-dec', type=float, nargs='+', default=LR_DEC,
                        help='Learning rate for decoder.')
    parser.add_argument('--mom-enc', type=float, default=MOM_ENC,
                        help='Momentum for encoder.')
    parser.add_argument('--mom-dec', type=float, default=MOM_DEC,
                        help='Momentum for decoder.')
    parser.add_argument('--wd-enc', type=float, default=WD_ENC,
                        help='Weight decay for encoder.')
    parser.add_argument('--wd-dec', type=float, default=WD_DEC,
                        help='Weight decay for decoder.')
    parser.add_argument('--optim-dec', type=str, default=OPTIM_DEC,
                        help='Optimiser algorithm for decoder.')
    parser.add_argument('--lamda', type=float, default=LAMDA,
                        help='Lamda for L1 norm.')
    parser.add_argument('-t', '--bn-threshold', type=float, default=BN_threshold,
                        help='Threshold for slimming BNs.')
    return parser.parse_args()


def create_segmenter(num_layers, num_classes, num_parallel, bn_threshold, gpu):
    """Create Encoder; for now only ResNet [50,101,152]"""
    #print('b1')
    #print(num_parallel)
    #print('b2')
    segmenter = refinenet(num_layers, num_classes, num_parallel, bn_threshold)
    assert(torch.cuda.is_available())
    segmenter.to(gpu[0])
    segmenter = torch.nn.DataParallel(segmenter, gpu)
    return segmenter


def create_loaders(dataset, inputs, train_dir, train_dir_va,val_dir, test_dir,train_list,train_list_va, val_list,test_list,
                   shorter_side, crop_size, input_size, low_scale, high_scale,
                   normalise_params, batch_size, num_workers, ignore_label):

    # Torch libraries
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split
    # Custom libraries
    from utils.datasets import SegDataset as Dataset
    from utils.datasets import SegDataset_va as Dataset_va
    from utils.datasets import SegDataset_TEST as Dataset_TEST
    from utils.transforms import Normalise, Pad, RandomCrop, RandomMirror, ResizeAndScale, \
                                 CropAlignToMask, ResizeAlignToMask, ToTensor, ResizeInputs

    #input_names, input_mask_idxs = ['rgb', 'depth'], [0, 2, 1]change for emotion
    #input_names = ['RGB','FLOW','MEL','DELTA']

    AlignToMask = CropAlignToMask if dataset == 'nyudv2' else ResizeAlignToMask
    composed_trn = transforms.Compose([
        ResizeAlignToMask(),
        #ResizeAndScale(shorter_side, low_scale, high_scale),
        #Pad(crop_size, [123.675, 116.28 , 103.53], ignore_label),
        #RandomMirror(),
        #RandomCrop(crop_size),
        ResizeInputs(input_size),
        Normalise(*normalise_params),
        ToTensor()
    ])
    composed_val = transforms.Compose([
        ResizeAlignToMask(),
        ResizeInputs(input_size),
        Normalise(*normalise_params),
        ToTensor()
    ])
    # Training and validation sets

    trainset = Dataset(dataset=dataset, data_file=train_list, data_dir=train_dir,
                       #input_names=input_names, input_mask_idxs=input_mask_idxs,change for emotion

                       transform_trn=composed_trn, transform_val=composed_val,
                       stage='train', ignore_label=ignore_label)
    trainset_va = Dataset_va(dataset=dataset, data_file=train_list_va, data_dir=train_dir_va,
                       # input_names=input_names, input_mask_idxs=input_mask_idxs,change for emotion

                       transform_trn=composed_trn, transform_val=composed_val,
                       stage='train', ignore_label=ignore_label)

    validset = Dataset(dataset=dataset, data_file=val_list, data_dir=val_dir,
                       #input_names=input_names, input_mask_idxs=input_mask_idxs,change for emotion

                       transform_trn=None, transform_val=composed_val, stage='val',
                       ignore_label=ignore_label)
    testset = Dataset_TEST(dataset=dataset, data_file=test_list, data_dir=test_dir,
                       # input_names=input_names, input_mask_idxs=input_mask_idxs,change for emotion

                       transform_trn=None, transform_val=composed_val, stage='val',
                       ignore_label=ignore_label)
    print_log('Created train set {} examples, val set {} examples'.format(len(trainset), len(validset)))
    # Training and validation loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    train_loader_va = DataLoader(trainset_va, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)#DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader,train_loader_va, val_loader,test_loader


def create_optimisers(lr_enc, lr_dec, mom_enc, mom_dec, wd_enc, wd_dec, param_enc, param_dec, optim_dec):
    """Create optimisers for encoder, decoder and controller"""
    optim_enc = torch.optim.SGD(param_enc, lr=lr_enc, momentum=mom_enc, weight_decay=wd_enc)
   # torch.optim.lr_scheduler.StepLR(optim_enc, 10, gamma=0.1, last_epoch=-1)
    #if optim_dec == 'sgd':
    #    optim_dec = torch.optim.SGD(param_dec, lr=lr_dec, momentum=mom_dec, weight_decay=wd_dec)
        #torch.optim.lr_scheduler.StepLR(optim_dec, 10, gamma=0.1, last_epoch=-1)
    #elif optim_dec == 'adam':
    #    optim_dec = torch.optim.Adam(param_dec, lr=lr_dec, weight_decay=wd_dec, eps=1e-3)

    return optim_enc#, optim_dec


def load_ckpt(ckpt_path, ckpt_dict):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    for (k, v) in ckpt_dict.items():
        if k in ckpt:
            v.load_state_dict(ckpt[k])
    best_val = ckpt.get('best_val', 0)
    epoch_start = ckpt.get('epoch_start', 0)
    print_log('Found checkpoint at {} with best_val {:.4f} at epoch {}'.
        format(ckpt_path, best_val, epoch_start))
    return best_val, epoch_start


def L1_penalty(var):
    return torch.pow(var,2).sum()


def train(segmenter,  train_loader, optim_enc, optim_dec,optim_enc_lr,optim_dec_lr, epoch,
          segm_crit, freeze_bn, slim_params, lamda, bn_threshold, print_loss=False):

    #pdb.set_trace()
    train_loader.dataset.set_stage('train')
    segmenter.train()
    if freeze_bn:
        for module in segmenter.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()

    for i, sample in enumerate(train_loader):

        #print('Epoch:',epoch,'Time',time.asctime( time.localtime(time.time()) ),'train input:',i, 'audio' , sample['audio'].shape, 'video', sample['video'].shape,'label',sample['label'])
        #print(sample['name'])
        #print('Epoch:', epoch, 'Time', time.asctime(time.localtime(time.time())), 'train input:', i)#, 'audio',
              #sample['audio'].shape, 'video', sample['video'].shape)
        start = time.time()
        #if epoch ==5:
        #    pdb.set_trace()
        inputs = sample['RGB'].cuda().float()

        #target = sample['mask'].cuda().long()change for emotion
        target = sample['label']
        # Compute outputs

        #outputs,_= segmenter(inputs)

        outputs, _, outputs_v, outputs_a = segmenter(inputs)
        #outputs_4, _, outputs_4v, outputs_4a = segmenter(inputs_4)
        #outputs_W, _, outputs_wv, outputs_wa = segmenter(inputs_W)
        loss = 0




        for output in outputs:


        #output=outputs
            soft_output = nn.Softmax()(output)
            #print(torch.squeeze(soft_output))
            # Compute loss and backpropagate

            loss += segm_crit(torch.squeeze(output), target.cuda())

            #loss_clf += segm_crit(torch.squeeze(output), target.cuda())


            L1_norm = sum([L1_penalty(m).cuda() for m in slim_params])


            loss += lamda * L1_norm  # this is actually counted for len(outputs) times
            #loss_l1 += lamda * L1_norm

        optim_enc.zero_grad()


        loss.backward()
        #if print_loss:
        #    print('step: %-3d: loss=%.2f' % (i, loss), flush=True)
        print('step: %-3d: loss=%.2f  ' % (i, loss))
        optim_enc.step()
        #optim_dec.step()

        #print('s6')
        losses.update(loss.item())
        batch_time.update(time.time() - start)

    #if epoch % 5 ==0:
    optim_enc_lr.step()
    #optim_dec_lr.step()
    slim_params_list = []
    #pdb.set_trace()
    for slim_param in slim_params:
        slim_params_list.extend(slim_param.cpu().data.numpy())
    slim_params_list = np.array(sorted(slim_params_list))
    print('Epoch %d, 3%% smallest slim_params: %.4f' % (epoch, \
        slim_params_list[len(slim_params_list) // 33]), flush=True)
    print('Epoch %d, portion of slim_params < %.e: %.4f' % (epoch, bn_threshold, \
        sum(slim_params_list < bn_threshold) / len(slim_params_list)), flush=True)

def train_va(segmenter,  train_loader_va, optim_enc, optim_dec,optim_enc_lr,optim_dec_lr, epoch,
          segm_crit, freeze_bn, slim_params, lamda, bn_threshold, print_loss=False):

    #pdb.set_trace()
    train_loader_va.dataset.set_stage('train')
    segmenter.train()
    if freeze_bn:
        for module in segmenter.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()

    for i, sample in enumerate(train_loader_va):

        #print('Epoch:',epoch,'Time',time.asctime( time.localtime(time.time()) ),'train input:',i, 'audio' , sample['audio'].shape, 'video', sample['video'].shape,'label',sample['label'])
        #print(sample['name'])
        #print('Epoch:', epoch, 'Time', time.asctime(time.localtime(time.time())), 'train input:', i)#, 'audio',
              #sample['audio'].shape, 'video', sample['video'].shape)
        start = time.time()
        #if epoch ==5:
        #    pdb.set_trace()
        inputs = sample['RGB'].cuda().float()

        #target = sample['mask'].cuda().long()change for emotion
        target_v = sample['label_v']
        target_a = sample['label_a']
        # Compute outputs

        #outputs,_= segmenter(inputs)

        outputs, _, outputs_v, outputs_a = segmenter(inputs)
        #outputs_4, _, outputs_4v, outputs_4a = segmenter(inputs_4)
        #outputs_W, _, outputs_wv, outputs_wa = segmenter(inputs_W)
        loss = 0




        for output in outputs:


        #output=outputs
            soft_output = nn.Softmax()(output)
            #print(torch.squeeze(soft_output))
            # Compute loss and backpropagate

            loss += segm_crit(torch.squeeze(outputs_v), target_v.cuda())
            loss += segm_crit(torch.squeeze(outputs_a), target_a.cuda())

            #loss_clf += segm_crit(torch.squeeze(output), target.cuda())


            L1_norm = sum([L1_penalty(m).cuda() for m in slim_params])


            loss += lamda * L1_norm  # this is actually counted for len(outputs) times
            #loss_l1 += lamda * L1_norm

        optim_enc.zero_grad()


        loss.backward()
        #if print_loss:
        #    print('step: %-3d: loss=%.2f' % (i, loss), flush=True)
        print('step: %-3d: loss=%.2f  ' % (i, loss))
        optim_enc.step()
        #optim_dec.step()

        #print('s6')
        losses.update(loss.item())
        batch_time.update(time.time() - start)

    #if epoch % 5 ==0:
    optim_enc_lr.step()
    #optim_dec_lr.step()
    slim_params_list = []
    #pdb.set_trace()
    for slim_param in slim_params:
        slim_params_list.extend(slim_param.cpu().data.numpy())
    slim_params_list = np.array(sorted(slim_params_list))
    print('Epoch %d, 3%% smallest slim_params: %.4f' % (epoch, \
        slim_params_list[len(slim_params_list) // 33]), flush=True)
    print('Epoch %d, portion of slim_params < %.e: %.4f' % (epoch, bn_threshold, \
        sum(slim_params_list < bn_threshold) / len(slim_params_list)), flush=True)

def validate(segmenter,  val_loader, epoch, num_classes=-1, save_image=0):

    outputhuizong0=[]
    outputhuizong1 = []
    global best_iou
    val_loader.dataset.set_stage('val')
    segmenter.eval()
    conf_mat = []
    gthuizong=[]
    for _ in range(1):
        conf_mat.append(np.zeros((num_classes, num_classes), dtype=int))
    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_loader,ncols=120)):
            # print('valid input:', sample['rgb'].shape, sample['depth'].shape, sample['mask'].shape)
            start = time.time()
            inputs = sample['RGB'].float().cuda()
            inputs_4 = sample['RGB_4'].float().cuda()
            inputs_W = sample['RGB_W'].float().cuda()
            #target = sample['mask']change for emotion
            target=sample['label']
            gt = target[:].data.cpu().tolist()#numpy().astype(np.uint8)[]
            gthuizong.extend(gt)
            #gt_idx = gt < num_classes  # Ignore every class index larger than the number of classes
            # Compute outputs
            outputs, _, outputs_v, outputs_a = segmenter(inputs)
            outputs_4, _, outputs_4v, outputs_4a = segmenter(inputs_4)
            outputs_W, _, outputs_wv, outputs_wa = segmenter(inputs_W)
            outputshuizong = [outputs[0]+outputs_4[0]+outputs_W[0]]
            for idx, output in enumerate(outputshuizong):
                #pdb.set_trace()
                #output=outputs
            #print(output)
                output1d = torch.squeeze(output).cpu().numpy()
                #cv2.resize(output[0, :num_classes].data.cpu().numpy().transpose(1, 2, 0),
                #                    target.size()[1:][::-1]).argmax(axis=2).astype(np.uint8)
                # Compute IoU
                if output1d.ndim==2:
                    outputmax=np.argmax(output1d,1).tolist()
                else:
                    outputmax = np.argmax(output1d).tolist()
                if idx==0:
                    if output1d.ndim == 2:
                        outputhuizong0.extend(outputmax)
                    else:
                        outputhuizong0.append(outputmax)
                if idx == 1:
                    if output1d.ndim == 2:
                        outputhuizong1.extend(outputmax)
                    else:
                        outputhuizong1.append(outputmax)


    conf_mat0= confusion_matrix(gthuizong, outputhuizong0)
    #conf_mat1 = confusion_matrix(gthuizong, outputhuizong1)
    report=classification_report(gthuizong, outputhuizong0,output_dict=True)
    #
    glob = np.diag(conf_mat0).sum()/np.float(conf_mat0.sum())
    #glob1 = np.diag(conf_mat1).sum() / np.float(conf_mat1.sum())
    testWAR=report['weighted avg']['recall']
    testUAR=report['macro avg']['recall']
    #import pdb
    #pdb.set_trace()

    print('matrix:\n',conf_mat0)
    #print('matrix1:\n', conf_mat1)
    print_log('Epoch %-4d    glob_acc=%-5.2f WAR=%-5.2f UAR=%-5.2f' %(epoch,  glob,testWAR,testUAR))
    print_log('')
    return glob,testUAR

def test(segmenter,  test_loader, epoch, num_classes=-1, save_image=0):

    outputhuizong0=[]
    outputhuizong0_v=[]
    outputhuizong0_a=[]
    outputhuizong0chazhi = []

    test_loader.dataset.set_stage('val')
    segmenter.eval()

    nameshuizong=[]
    gthuizong=[]

    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_loader,ncols=120)):
            # print('valid input:', sample['rgb'].shape, sample['depth'].shape, sample['mask'].shape)

            inputs = sample['RGB'].float().cuda()
            inputs_4 = sample['RGB_4'].float().cuda()
            inputs_W = sample['RGB_W'].float().cuda()
            names=sample['names']
            #gt=sample['label']
            #target = sample['mask']change for emotion
            #target=sample['label']
            nameshuizong.append(names)
            gthuizong.extend(gt.tolist())
            ij=0

            outputs,_,outputs_v,outputs_a = segmenter(inputs)
            outputs_4, _ ,outputs_4v,outputs_4a= segmenter(inputs_4)
            outputs_W, _,outputs_wv,outputs_wa = segmenter(inputs_W)
            outputshuizong=[outputs[0]+outputs_4[0]+outputs_W[0]]
            for idx, output in enumerate(outputshuizong):
                output1d_v=torch.squeeze(outputs_v).cpu().numpy()
                output1d_a = torch.squeeze(outputs_a).cpu().numpy()
                #outputs_v_huizong
                output1d = torch.squeeze(output).cpu().numpy()

                if output1d.ndim==2:
                    outputmax=np.argmax(output1d,1).tolist()
                    outputmax_v=np.argmax(output1d_v,1).tolist()
                    outputmax_a = np.argmax(output1d_a, 1).tolist()
                else:
                    outputmax = np.argmax(output1d).tolist()
                    outputmax_v = np.argmax(output1d_v).tolist()
                    outputmax_a = np.argmax(output1d_a).tolist()
                if idx==0:
                    if output1d.ndim == 2:
                        outputhuizong0.extend(outputmax)
                        outputhuizong0_v.extend(outputmax_v)
                        outputhuizong0_a.extend(outputmax_a)
                    else:
                        outputhuizong0.append(outputmax)
                        outputhuizong0_v.append(outputmax_v)
                        outputhuizong0_a.append(outputmax_a)








    return outputhuizong0,nameshuizong,outputhuizong0_v,outputhuizong0_a

def main():
    global args, best_iou
    best_iou = 0

    args = get_arguments()
    #pdb.set_trace()
    args.num_stages = len(args.lr_enc)

    ckpt_dir = os.path.join('./ckpt', args.ckpt)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.system('cp -r *py models utils data %s' % ckpt_dir)
    helpers.logger = open(os.path.join(ckpt_dir, 'log.txt'), 'w+')
    print_log(' '.join(sys.argv))
    #import pdb
    #pdb.set_trace()
    # Set random seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    # Generate Segmenter
    torch.cuda.set_device(args.gpu[0])
    segmenter = create_segmenter(args.enc, args.num_classes, 2,
                                 args.bn_threshold, args.gpu)
    #print('sb')
    #print(len(args.input))
    enc_params, dec_params, slim_params = [], [], []

    for name, param in segmenter.named_parameters():
        #if bool(re.match('.*conv1.*|.*bn1.*|.*layer.*', name)):
        #print(name)
        enc_params.append(param)
        if args.print_network:
            print_log(' Enc. parameter: {}'.format(name))
        #else:
        #    dec_params.append(param)
        #    if args.print_network:
        #        print_log(' Dec. parameter: {}'.format(name))
        if param.requires_grad:
            print(' grad. parameter: {}'.format(name))
        if param.requires_grad and name.endswith('weight') and 'bn2' in name:
            #if len(slim_params) % 2 == 0:
            #    slim_params.append(param[:len(param) // 2])
            #else:
            #    slim_params.append(param[len(param) // 2:])

            slim_params.append(param)
    if args.print_network:
        print_log('')
    segmenter = model_init(segmenter, args.enc, len(args.input), imagenet=args.enc_pretrained)
    print_log('Loaded Segmenter {}, ImageNet-Pre-Trained={}, #PARAMS={:3.2f}M'
          .format(args.enc, args.enc_pretrained, compute_params(segmenter) / 1e6))
    # Restore if any
    best_val, epoch_start = 0, 0
    if args.resume:
        if os.path.isfile(args.resume):
            best_val, epoch_start = load_ckpt(args.resume, {'segmenter': segmenter})
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume))
            return
    epoch_current = epoch_start
    # Criterion
    #segm_crit = nn.NLLLoss(ignore_index=args.ignore_label).cuda()
    segm_crit = nn.CrossEntropyLoss().cuda()
    # Saver
    saver = Saver(args=vars(args), ckpt_dir=ckpt_dir, best_val=best_val,
                  condition=lambda x, y: x > y)  # keep checkpoint with the best validation score
    miouzong = []
    miouzongUAR=[]
    for task_idx in range(args.num_stages):
        #task_idx=0
        total_epoch = sum([args.num_epoch[idx] for idx in range(task_idx + 1)])
        if epoch_start >= total_epoch:
            continue
        start = time.time()
        torch.cuda.empty_cache()
            # Create dataloaders
        train_loader,train_loader_va, val_loader,test_loader = create_loaders(
            DATASET, args.input, args.train_dir,args.train_dir_va, args.val_dir, args.test_dir, args.train_list,args.train_list_va, args.val_list,args.test_list,
            args.shorter_side, args.crop_size, args.input_size, args.low_scale, args.high_scale,
            args.normalise_params, args.batch_size, args.num_workers, args.ignore_label)
        if args.evaluate:
            return validate(segmenter, args.input, val_loader, 0, num_classes=args.num_classes,
                                save_image=args.save_image)

            # Optimisers
        print_log('Training Stage {}'.format(str(task_idx)))
        optim_enc = create_optimisers(
            args.lr_enc[0], args.lr_dec[0],
            args.mom_enc, args.mom_dec,
            args.wd_enc, args.wd_dec,
            enc_params, dec_params, args.optim_dec)
        optim_enc_lr=torch.optim.lr_scheduler.StepLR(optim_enc, 5, gamma=0.9)
            #optim_dec_lr =torch.optim.lr_scheduler.StepLR(optim_dec, 1, gamma=0.8)
        print(args.num_epoch)
        for epoch in range(min(args.num_epoch[task_idx], total_epoch - epoch_start)):
                #pdb.set_trace()
            train(segmenter, train_loader, optim_enc, optim_enc, optim_enc_lr, optim_enc_lr, epoch_current,
                      segm_crit, args.freeze_bn, slim_params, args.lamda, args.bn_threshold, args.print_loss)
            train_va(segmenter, train_loader_va, optim_enc, optim_enc,optim_enc_lr,optim_enc_lr, epoch_current,
                segm_crit, args.freeze_bn, slim_params, args.lamda, args.bn_threshold, args.print_loss)
            for param_group in optim_enc.param_groups:
                print('LR:')
                print(param_group['lr'])
                break
            if (epoch + 1) % (args.val_every) == 0:
                miou,miouUAR = validate(segmenter, val_loader, epoch_current, args.num_classes)
                miouzong.append(miou)
                miouzongUAR.append(miouUAR)

                print(miouzong)
                print("maxWAR is %.4f at step: %d" %(max(miouzong),np.argmax(miouzong)))
                if miou>=max(miouzong):
                    best_segmenter=segmenter
                print(miouzongUAR)
                print("maxUAR is %.4f at step: %d" % (max(miouzongUAR), np.argmax(miouzongUAR)))
                #outputhuizong, outputsb = test(segmenter, test_loader, epoch_current, args.num_classes)
                saver.save(miou, {'segmenter' : segmenter.state_dict(), 'epoch_start' : epoch_current})
            epoch_current += 1
        outputhuizong, outputsb,chazhisb,gtdsb = test(best_segmenter, test_loader, epoch_current, args.num_classes)

        ijsb=0
        with open('/data/lucheng/lisunan/ABAW7/DFEWrelabel40718.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(['videoname', 'newlabel', 'label1', 'label2'])
            for outputsb2 in outputsb:
                for outputsb3 in outputsb2:

                    writer.writerow([outputsb3, outputhuizong[ijsb],chazhisb[ijsb],gtdsb[ijsb]])
                    ijsb = ijsb + 1






        print_log('Stage {} finished, time spent {:.3f}min\n'.format(task_idx, (time.time() - start) / 60.))

    print_log('All stages are now finished. Best Val is {:.3f}'.format(saver.best_val))
    helpers.logger.close()


if __name__ == '__main__':
    main()

