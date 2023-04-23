import argparse
from cProfile import label
from sklearn.preprocessing import LabelBinarizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, MultiStepLR
import random
import os, sys
import numpy as np
from model.losses import SimCLR
from utils.visualization import tsne_visualization,roc_visualization
from data.DataLoader import loading_data, splitting_data, count_label_labellist
#from model.transformer import TransformerModel
from model.simple import DeepConvLSTMModel, LSTMModel, Net, LSTMNet
from model.resnet_big import ResNetBaseline, ResNetSupCon
from model.lossfunction import ConTimeLoss, SupConLoss
from tqdm.notebook import tqdm
from model.vit import ViT
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import gc
import time
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
#from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import torch.backends.cudnn as cudnn
from utils.utils import AverageMeter, warmup_learning_rate, accuracy, EarlyStopping, save_model
from sklearn.metrics import f1_score, roc_auc_score



# Training Function
def train(train_data, train_label, model, criterion, optimizer, epoch, batch_size):
    'for one epoch training'
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()

    iteration_num = (int)(len(train_data)/batch_size)
    #print(len(train_data), batch_size)
    #shuffle_list = list(zip(train_data, train_label))
    #random.shuffle(shuffle_list)
    #train_data, train_label= zip(*shuffle_list)
    
    
    # for each batch
    for i in range(iteration_num):
        data_time.update(time.time() - end)
    
        data = train_data[i*batch_size:(i+1)*batch_size].to(device)
        labels = train_label[i*batch_size:(i+1)*batch_size].to(device)

        # batch size
        bsz = labels.shape[0]
        # warm-up learning rate
        #warmup_learning_rate(opt, epoch, i, len(train_data), optimizer)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        # update metric
        losses.update(loss.item(), bsz)
        train_f1 = f1_score(labels.cpu(), output.argmax(dim=1).cpu(), average='macro')
        acc1 = (output.argmax(dim=1) == labels).float().mean()
        top1.update(acc1, bsz)
        #epoch_val_accuracy += acc / len(valid_loader)
        #    epoch_val_loss += val_loss / len(valid_loader)            


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        print('Train: [{0}][{1}/{2}]\t'
            'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'loss {loss.val:.3f} ({loss.avg:.3f})\t'
            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'F1 {train_f1:.3f}'.format(epoch, i + 1, iteration_num, batch_time=batch_time,
            data_time=data_time, loss=losses, top1=top1, train_f1=train_f1))
        sys.stdout.flush()

    return model, losses.avg, top1.avg

# Training Function
def train_simclr(train_data, val_data, max_epochs=500, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join("../saved_models/", 'SimCLR'),
                         gpus=1 if str(device)=='cuda:0' else 0,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc_top5'),
                                    LearningRateMonitor('epoch')],
                         progress_bar_refresh_rate=1)
   
    pl.seed_everything(42) # To be reproducable
    model = SimCLR(max_epochs=max_epochs, **kwargs)
    trainer.fit(model, train_data, val_data)
    model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    return model

# Validate the model
def validate(val_data, val_label, model, criterion):
    'for one epoch validation'
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        # for each batch
        end = time.time()
        for i in range(1):
            data = val_data.to(device)
            labels = val_label.to(device)
            bsz = labels.shape[0]

            # forward
            output = model(data)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1 = (output.argmax(dim=1) == labels).float().mean()
            top1.update(acc1, bsz)
            val_f1 = f1_score(labels.cpu(), output.argmax(dim=1).cpu(), average='macro')
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 1 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'F1 {val_f1:.3f}'.format(i, len(val_data), batch_time=batch_time,
                       loss=losses, top1=top1, val_f1=val_f1))

    print('*Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return model, losses.avg, top1.avg

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

def test(test_data, test_label, model, criterion, num_class):
    model.eval() 
    # test loss 및 accuracy을 모니터링하기 위해 list 초기화
    test_loss = 0.0
    model.eval() # prep model for evaluation

    
    # forward pass: 입력을 모델로 전달하여 예측된 출력 계산
    output = model(test_data)
    # calculate the loss
    loss = criterion(output, test_label)
        
    #print(output, test_label)
    # update test loss
    test_loss += loss.item()*test_data.size(0)
    # 출력된 확률을 예측된 클래스로 변환
    _, pred = torch.max(output, 1)
    #print(pred, test_label)
    # 예측과 실제 라벨과 비교
    correct = np.squeeze(pred.eq(test_label.data.view_as(pred)))
    # 각 object class에 대해 test accuracy 계산
    print('\nacc1 : {:.3f}\n'.format((output.argmax(dim=1) == test_label).float().mean()))
    print('test_f1 : {:.3f}\n'.format(f1_score(test_label.cpu(), output.argmax(dim=1).cpu(), average='macro')))
        

    # calculate and print avg test loss
    test_loss = test_loss/len(test_data)
    print('Test Loss: {:.6f}\n'.format(test_loss))
    
    # calculate confusion matrix
    stacked = torch.stack((test_label, output.argmax(dim=1)),dim=1)
    #print(stacked)
    cmt = torch.zeros(num_class, num_class, dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        cmt[tl, pl] = cmt[tl, pl] + 1
    print(cmt)

    return output, test_label 

def f1_score_except_NA(test_label, output):
    test_pred = output.argmax(dim=1).clone().detach()
    test_label_clone = test_label.clone().detach()
        
    count = 0
    while count < len(test_label_clone):
        if(test_label_clone [count] == (max(test_label_clone))):
             test_label_clone  = torch.cat([test_label_clone [0:count], test_label_clone [count+1:]])
             test_pred = torch.cat([test_pred[0:count], test_pred[count+1:]])
             count = count-1
        count = count+1   

    print('test_f1_except_NA : {:.3f}\n'.format(f1_score(test_label_clone .cpu(), test_pred.cpu(), average='macro')))
    return (f1_score(test_label_clone .cpu(), test_pred.cpu(), average='macro'))

# Setting the model 
def set_model(num_classes,feature_dim, dim, model_type, loss, temp= 0):
    if(model_type == 'simple'):
        model = Net(num_classes=num_classes, feature_dim= feature_dim, dim=dim).to(device)
    elif(model_type == 'transformer'):
        model = ViT(num_classes=num_classes, feature_dim= feature_dim, dim=dim).to(device)
    elif(model_type =='ResNet'):
        model = ResNetBaseline(in_channels=feature_dim, num_pred_classes=num_classes).to(device)
    elif(model_type =='BiLSTM'): 
        #model = LSTMModel(input_dim=feature_dim, hidden_dim=dim, layer_dim =2, output_dim=num_classes).to(device)
        model = LSTMNet(vocab_size = 300,embedding_dim = feature_dim, hidden_dim = dim,output_dim = num_classes,n_layers = 2,bidirectional = True,dropout = 0.2).to(device)
    elif(model_type == "DeepConvLSTM"):
        model = DeepConvLSTMModel(feature_dim=feature_dim, n_hidden=dim, n_layers=2, n_filters=64, n_classes= num_classes, filter_size=5, drop_prob=0.2)



    # loss function
    if(loss == 'CE'):
        criterion = nn.CrossEntropyLoss()
    elif(loss =='SupCon'):
        #model = ResNetSupCon()
        criterion = SupConLoss(temperature= temp)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion   

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(prog='NovelHAD', description='NovelHAD: Contrastive learning based novel human activity detection from time series sensor data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # for scehme
    parser.add_argument('--dataset', type=str, default='lapras', help='choose one of them: lapras, casas, aras_a, aras_b, opportunity')
    parser.add_argument('--padding', type=str, default='mean', help='choose one of them : no, max, mean')
    parser.add_argument('--timespan', type=int, default=10000, help='choose of the number of timespan between data points(1000 = 1sec, 60000 = 1min)')
    parser.add_argument('--min_seq', type=int, default=10, help='choose of the minimum number of data points in a example')
    parser.add_argument('--min_samples', type=int, default=20, help='choose of the minimum number of samples in each label')
    parser.add_argument('--arg_ood', type=int, default=-1, help='choose of label number that wants to delete')
    parser.add_argument('--version', type=str, default='CL', help='choose of version want to do : ND or CL')


    parser.add_argument('--aug_method', type=str, default='AddNoise', help='choose the data augmentation method')
    parser.add_argument('--aug_wise', type=str, default='Temporal', help='choose the data augmentation wise')

    parser.add_argument('--test_ratio', type=float, default=0.1, help='choose the number of test ratio')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='choose the number of vlaidation ratio')
    parser.add_argument('--overlapped_ratio', type=int, default= 50, help='choose the number of windows''overlapped ratio')
    parser.add_argument('--encoder', type=str, default='transformer', help='choose one of them: simple, transformer')

    # for training   
    parser.add_argument('--loss', type=str, default='CE', help='choose one of them: crossentropy loss, contrastive loss')
    parser.add_argument('--optimizer', type=str, default='', help='choose one of them: adam')
    parser.add_argument('--epochs', type=int, default= 20, help='choose the number of epochs')
    parser.add_argument('--patience', type=int, default=20, help='choose the number of patience for early stopping')
    parser.add_argument('--batch_size', type=int, default=64, help='choose the number of batch size')
    parser.add_argument('--lr', type=float, default=3e-5, help='choose the number of learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, help='choose the number of gamma')
    parser.add_argument('--seed', type=int, default=42, help='choose the number of seed')
    parser.add_argument('--temp', type=float, default=0.07, help='temperature for loss function')


    args = parser.parse_args()
    print('options', args)

    torch.cuda.empty_cache()

    # check gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('available device :', device)
    seed_everything(args.seed)

    # Dataset extraction def loading_data(dataset, padding, timespan, min_seq, min_samples): 
    data_list, label_list, num_classes = loading_data(args.dataset, args.padding, args.timespan, args.min_seq, args.min_samples, args.aug_method, args.aug_wise)

    # def tsne_visualization(data_list, label_list, dataset, dim):
    tsne_visualization(data_list, label_list, args.dataset, 3)

    num_classes, entire_list, train_list, valid_list, test_list, entire_label_list, train_label_list, \
    valid_label_list, test_label_list \
    = splitting_data(args.dataset, args.test_ratio, args.valid_ratio, args.padding, args.seed, \
                     args.timespan, args.min_seq, args.min_samples, args.aug_method, args.aug_wise, args.arg_ood, args.version)
    
    print('finishing data processing-------------------------')
    #types_label_list, _ =count_label_labellist(train_list, train_label_list)
    print('The number of classes: ', num_classes)

    print('train data shape:', train_list.shape)
    print('train label shape:', train_label_list.shape)   
    

    # set model for using   
    model, criterion = set_model(len(num_classes), len(train_list[0][0]), 64, args.encoder, args.loss, args.temp)
    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # set scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # set early stopping
    early_stopping = EarlyStopping(patience = args.patience, verbose = True)
    
    print('starting training and validation-------------------------------------------------------------')
    # training routine
    for epoch in range(args.epochs):
        #adjust_learning_rate(opt, optimizer, epoch)
        # train for one epoch
        time1 = time.time()
        model, loss, train_acc = train(train_list, train_label_list, model, criterion, optimizer, epoch, args.batch_size)
         
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        #print('train_loss', loss, epoch)
        #print('train_acc', train_acc, epoch)
        #print('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluation
        #model, loss, val_acc = validate(valid_list, valid_label_list, model, criterion)
        #print('val_loss', loss, epoch)
        #print('val_acc', val_acc, epoch)

        early_stopping(loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
        #if val_acc > best_acc:
        #    best_acc = val_acc

        #if epoch % opt.save_freq == 0:
        #    save_file = os.path.join(
        #        opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        #    save_model(model, optimizer, opt, epoch, save_file)
    model.load_state_dict(torch.load('checkpoint.pt'))
    
    # for embedding visualization
    embedding, test_label = test(entire_list, entire_label_list, model, criterion, len(num_classes))
    tsne_visualization(embedding.cpu(), entire_label_list.cpu(), args.dataset+'_'+ args.encoder +'_test', 2)
    
    # for testing
    output, test_label = test(test_list, test_label_list, model, criterion, len(num_classes))    
    save_model(model, optimizer, args, epoch, 'last.pth')

    if(args.version == "ND"):

        f1_score_before = f1_score_except_NA(test_label, output)

        print("Novelty detection")
        train_list = test_list
        train_label_list = output.argmax(dim=1)
        
        for epoch in range(args.epochs):
            #adjust_learning_rate(opt, optimizer, epoch)
            # train for one epoch
            time1 = time.time()
            model, loss, train_acc = train(train_list, train_label_list, model, criterion, optimizer, epoch, args.batch_size)
            
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            early_stopping(loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        model.load_state_dict(torch.load('checkpoint.pt'))

        output, test_label = test(valid_list, valid_label_list, model, criterion, len(num_classes))
        f1_score_after = f1_score_except_NA(test_label, output)

        
        save_model(model, optimizer, args, epoch, 'last.pth')

      

    
    # Define the softmax function
    softmax = torch.nn.Softmax(dim=1)
    output = softmax(output)
    print('test_auc : {:.3f}\n'.format(roc_auc_score(test_label.cpu(), output.cpu().detach().numpy(), multi_class='ovr')))
    roc_visualization(test_label.cpu(), output.cpu().detach().numpy())

    


    if(args.version == "ND"):
        print('before retraining : {:.3f}, after retraining : {:.3f} \n'.format(f1_score_before, f1_score_after))



    #simclr_model = train_simclr(in_channels=len(train_list[0][0]), num_pred_classes =len(num_classes), train_data= train_list, val_data= valid_list, hidden_dim=64,lr=5e-4,
    #temperature=0.07, weight_decay=1e-4, max_epochs=500)

torch.cuda.empty_cache()
