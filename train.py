import argparse
import os
import time
from datetime import datetime
import torch.utils.data
from torch.nn import DataParallel
from torch.optim.lr_scheduler import MultiStepLR
from dataManagement.DatasetHelper import DatasetHelper
from dataManagement.DatasetLoader import DatasetLoader
from dataManagement.CustomDataset import CustomDataset
from torch.utils.data import DataLoader
from models.mfcf import mfcf
try:
    import models
except:
    import sys

    sys.path.insert(0, './')

from models.utils import init_log, progress_bar

if __name__ == '__main__':
    parser = argparse.ArgumentParser("MFCF OFM 36 2024")
    parser.add_argument('--gpu', type=str, default='0', help='select the gpu or gpus to use')
    parser.add_argument('--dataset-path', type=str, default='./data')
    parser.add_argument('--numclasses', type=int, default=36)
    parser.add_argument('--max-epoch', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-words-x-doc', type=int, default=100)
    parser.add_argument('--num-words-to-keep', type=int, default=3000)
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--output-image-width', type=int, default=448)
    parser.add_argument('--proposal-num', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate for model")
    parser.add_argument('--wd', type=float, default=1e-4, help="wd rate for model")
    parser.add_argument('--momentum', type=float, default=0.9, help="sgd momentum")
    parser.add_argument('--adam-w', type=float, default=0.0, help="adam weight")
    parser.add_argument('--resume-path', type=str, default='', help="resume weight path")
    parser.add_argument('--save-dir', type=str, default='./runs')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    experiment_name = f'nts_net_' + datetime.now().strftime('%Y%m%d_%H%M%S')

    save_dir = os.path.join(args.save_dir, f'{experiment_name}_' + datetime.now().strftime('%Y%m%d_%H%M%S'))

    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)

    logging = init_log(save_dir)
    _print = logging.info
    # train and val dataset path
    train_muldata_path = f'{args.dataset_path}/mul_datasets/mul_train.txt'
    val_muldata_path = f'{args.dataset_path}/mul_datasets/mul_val.txt'

    # from txt get data_loader（train_data and val_data）
    data_loader = DatasetLoader()
    data_loader.load_data(train_muldata_path, val_muldata_path, delimiter='|')
    train_data = data_loader.get_train_data()
    val_data = data_loader.get_val_data()

    # text and label one hot code
    data_helper = DatasetHelper(args.num_words_to_keep)
    train_y, val_y = data_helper.preprocess_labels(train_data, val_data)
    train_i, val_i = data_helper.preprocess_images(train_data.get_images(), val_data.get_images())
    train_t, val_t = data_helper.preprocess_texts(train_data.get_texts(), val_data.get_texts(), args.num_words_x_doc)

    # labels、images、text  set to data_geter
    data_loader.set_train_data(train_y, train_i, train_t)
    data_loader.set_val_data(val_y, val_i, val_t)

    # get CustomDataset (train and val)
    train_custom_dataset = CustomDataset(data_loader.get_train_data())
    val_custom_dataset = CustomDataset(data_loader.get_val_data())

    train_loader = DataLoader(train_custom_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_custom_dataset, args.batch_size, shuffle=True)

    # define model
    mul_net = mfcf(num_classes=args.numclasses, vocab_size=args.num_words_to_keep,
                              embedding_size=args.embedding_dim, topN=6, device='cuda')

    creterion = torch.nn.CrossEntropyLoss()

    # define optimizer
    raw_optimizer = torch.optim.SGD(list(mul_net.parameters()), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.wd)
    scheduler = MultiStepLR(raw_optimizer, milestones=[20, 30, 40], gamma=0.1)

    mul_net = mul_net.cuda()
    mul_net = DataParallel(mul_net)

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    best_epoch = 0
    best_loss = 0
    best_acc = 0
    best_net_state_dict = {}
    start = time.time()
    for epoch in range(0, args.max_epoch):
        scheduler.step()
        ##########################  train the model  ###############################
        _print('--' * 50)
        train_loss = 0
        train_correct = 0
        train_total = 0
        mul_net.train()
        for i, data in enumerate(train_loader):
            labels, imgs, texts = data[0].cuda(), data[1].cuda(), data[2].cuda()
            batch_size = imgs.size(0)
            labels = torch.argmax(labels, dim=1)
            raw_optimizer.zero_grad()

            part_logits, top_n_prob, img_logits, res_logits, text_logits, mul_logits = mul_net(imgs, texts)
            # calculate loss
            part_loss = models.Arpn.list_loss(part_logits.view(batch_size * args.proposal_num, -1),
                                              labels.unsqueeze(1).repeat(1, args.proposal_num).view(-1)).view(batch_size, args.proposal_num)
            train_rank_loss = models.Arpn.ranking_loss(top_n_prob, part_loss, proposal_num=args.proposal_num)
            train_partcls_loss = creterion(part_logits.view(batch_size * args.proposal_num, -1),
                                           labels.unsqueeze(1).repeat(1, args.proposal_num).view(-1))
            train_img_loss = creterion(img_logits, labels)
            train_res_loss = creterion(res_logits, labels)
            train_text_loss = creterion(text_logits, labels)
            train_mul_loss = creterion(mul_logits, labels)

            total_loss = train_rank_loss + train_partcls_loss + train_img_loss + train_res_loss + train_mul_loss + train_text_loss

            train_total += batch_size
            train_loss += train_mul_loss.item() * batch_size
            _, mul_predicts = torch.max(mul_logits, 1)
            # Calculate
            train_correct += torch.sum(mul_predicts.data == labels.data)

            # backward
            total_loss.backward()
            raw_optimizer.step()

            progress_bar(i, len(train_loader), 'train')
            _print(f'Batch {i}/{len(train_loader)}   Loss: {train_mul_loss.item()}')

        train_loss = train_loss / train_total
        train_acc = float(train_correct / train_total)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        _print(
            'epoch:{} - train loss: {:.3f} and train acc: {:.3f} total sample: {}'.format(
                epoch,
                train_loss,
                train_acc,
                train_total))
        ##########################  evaluate net  ###############################
        test_loss = 0
        test_correct = 0
        test_total = 0
        mul_net.eval()
        for i, data in enumerate(val_loader):
            with torch.no_grad():
                labels, imgs, texts = data[0].cuda(), data[1].cuda(), data[2].cuda()
                batch_size = imgs.size(0)
                labels = torch.argmax(labels, dim=1)

                part_logits, top_n_prob, img_logits, res_logits, text_logits, mul_logits = mul_net(imgs, texts)
                # calculate loss
                vale_res_loss = creterion(mul_logits, labels)

                test_total += batch_size
                test_loss += train_mul_loss.item() * batch_size
                _, mul_predicts = torch.max(mul_logits, 1)
                test_correct += torch.sum(mul_predicts.data == labels.data)
                progress_bar(i, len(val_loader), 'eval val set')

        test_loss = test_loss / test_total
        test_acc = float(test_correct) / test_total
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        _print(
            'epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample: {}'.format(
                epoch,
                test_loss,
                test_acc,
                test_total))
        net_state_dict = mul_net.module.state_dict()

        if test_acc > best_acc:
            best_epoch = epoch
            best_acc = test_acc
            best_loss = test_loss
            best_net_state_dict = net_state_dict
            print(train_loss_list)
            print(test_loss_list)
    end = time.time()
    print(end - start)
    ##########################  save model  ###############################
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save({
        'epoch': best_epoch,
        'test_loss': best_loss,
        'test_acc': best_acc,
        'net_state_dict': best_net_state_dict},
        os.path.join(save_dir, '%03d.ckpt' % best_epoch))
    print(train_loss_list)
    print(test_loss_list)
    print(train_acc_list)
    print(test_acc_list)
    all_output = [str(end - start), '{', "\"train_acc_list\": " + str(train_acc_list), "\"test_acc_list\": " +
                  str(test_acc_list), "\"train_loss_list\": " + str(train_loss_list), "\"test_loss_list\": "
                  + str(test_loss_list), '}']
    with open(os.path.join(save_dir, 'output.txt'), 'w') as output:
        output.writelines(all_output)
    print('finishing training')
