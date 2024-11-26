import os
import json
import torch
import argparse
import numpy as np
from utils import *
from model.model import MCGCL
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from utils.data_loader import TrainDataLoader, ValTestDataLoader

torch.cuda.current_device()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='junyi', help='dataset name: junyi/ASSIST')
parser.add_argument('--config_file', default='config.txt')
parser.add_argument('--save_dir', default='test')

parser.add_argument('--epoch', type=int, help='number of epochs to train for')
parser.add_argument('--emb_size', type=int, help='embedding size')
parser.add_argument('--l2', type=float, help='l2 penalty')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--layers', type=float, help='the number of layer used')
parser.add_argument('--beta', type=float, help='ssl task maginitude')
parser.add_argument('--gamma', type=float, help='ssl task maginitude')
parser.add_argument('--shuffle', type=bool, help = 'shuffle datasets')
parser.add_argument('--save_batch', type=int, help='batch to save intermediate result')
parser.add_argument('--gpu', type=int, help='The id of gpu.')
parser.add_argument('--batchsize', type=int,help='batch')
parser.add_argument('--method', type=str, default="MCGCL", help='variant')
args = parser.parse_args()
print(args)

device = torch.device((f'cuda:{args.gpu}') if torch.cuda.is_available() else 'cpu')

def main():
    folder_path = 'result/'+ args.save_dir
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open("data_config.json", 'r') as f:
        data_num = json.load(f)[args.dataset]
    print("Number of Students,  Exercises, Knowledge Concepts:",data_num)
    data_file = '../dataset/'+args.dataset+'/train_set.json'
    data_loader = TrainDataLoader(args.batchsize, data_file, data_num)
    model = MCGCL(args,data_num).to(device)
    loss_function = model.loss_function
    optimizer = model.optimizer
    with open('result/'+args.save_dir+'/result.txt', 'a', encoding='utf8') as f:
        f.write(f'layer= {args.layers}  batch={args.batchsize}  lr{args.lr}')
        
    for epoch in range(args.epoch):
        print('-------------------------------------------------------')
        data_loader.reset()
        pred_all, label_all, pred_labels = [], [], []
        batch_cnt = 0
        one_epoch_loss = 0.0
        while not data_loader.is_end():
            batch_cnt += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
            optimizer.zero_grad()
            output_1, losse, losss = model.forward(input_stu_ids, input_exer_ids, "train",epoch, batch_cnt, args.save_dir, args.method, args.save)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)
            loss = loss_function(torch.log(output+1e-10), labels)

            loss += losse + losss
            loss.backward()
            optimizer.step()
            
            pred_labels += torch.argmax(output, dim=1).to(torch.device('cpu')).tolist()
            pred_all += output_1.view(-1).to(torch.device('cpu')).tolist()
            label_all += labels.to(torch.device('cpu')).tolist()
            one_epoch_loss += loss.item()

        print(f"epoch: {epoch+1}")
        train_loss = one_epoch_loss / batch_cnt

        pred_all = np.array(pred_all)
        label_all = np.array(label_all)
        pred_labels = np.array(pred_labels)
        train_accuracy = accuracy_score(label_all, pred_labels)
        labels_float = label_all.astype(float)
        mse = mean_squared_error(labels_float, pred_all)
        rmse = np.sqrt(mse)
        auc = roc_auc_score(label_all, pred_all)

        print(f'acc: {train_accuracy:.4f}  auc: {auc:.4f}  rmse: {rmse:.4f}, loss : {train_loss:.4f}  ')
        with open(folder_path + '/result.txt', 'a', encoding='utf8') as f:
            f.write('------------------------------------------------------\n')
            f.write(f'epoch= {epoch+1}\n')
            f.write(f'layer= {args.layers}  batch={args.batchsize}  lr{args.lr}\n')
            f.write(f'train_loss = {loss} , acc = {train_accuracy} , auc = {auc} , rmse = {rmse}\n')

        if epoch %args.save_batch == 0:
            model_folder = folder_path + '/modal/'
            file_path = os.path.join(model_folder, 'model_epoch'+str(epoch + 1)+'.pth')
            save_snapshot(model, file_path)
        rmse, auc, acc = predict(args, model, epoch, data_num[2])
        
        print(f'acc: {acc:.4f}  auc: {auc:.4f}  rmse: {rmse:.4f}')
        with open('result/'+args.save_dir+'/result.txt', 'a', encoding='utf8') as f:
            f.write(f'test_acc = {acc} , auc = {auc} , rmse = {rmse}\n')

def predict(args, net, epoch, knowledge_num):
    data_file = '../dataset/'+args.dataset+'/test_set.json'
    data_loader = ValTestDataLoader(data_file,knowledge_num)
    data_loader.reset()
    net.eval()

    batch_cnt = 0
    pred_all, label_all, pred_labels= [], [], []
    while not data_loader.is_end():
        batch_cnt += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        if batch_cnt!=1:
            output_1,_,_ = net.forward(input_stu_ids, input_exer_ids, "test", epoch, batch_cnt, args.save_dir,args.method, args.save, user_emb, exer_emb)
        else:
            output_1, user_emb, exer_emb = net.forward(input_stu_ids, input_exer_ids,"test", epoch, batch_cnt, args.save_dir,args.method, args.save)
        output_0 = torch.ones(output_1.size()).to(device) - output_1
        output = torch.cat((output_0, output_1), 1)
        
        pred_labels += torch.argmax(output, dim=1).to(torch.device('cpu')).tolist()
        pred_all += output_1.view(-1).to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    pred_labels = np.array(pred_labels)
    accuracy = accuracy_score(label_all, pred_labels)
    labels_float = label_all.astype(float)
    mse = mean_squared_error(labels_float, pred_all)
    rmse = np.sqrt(mse)
    auc = roc_auc_score(label_all, pred_all)
    with open('result/'+args.save_dir+'/model_val.txt', 'a', encoding='utf8') as f:
        f.write(f'epoch= {epoch+1}, accuracy= {accuracy}, rmse= {rmse}, auc= {auc}\n')
    return rmse, auc, accuracy


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


if __name__ == '__main__':
    main()
