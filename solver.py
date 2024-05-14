import torch
import torch.nn as nn
import os
from networks import get_model
from datasets import data_merge, data_merge_R
from optimizers import get_optimizer
from torch.utils.data import Dataset, DataLoader
from transformers import *
from utils import *
from configs import parse_args
import time
import numpy as np
import random
from loss import *
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(16)
np.random.seed(16)
random.seed(16)
# print(torch.__version__)
# print(torch.cuda.version())

def main(args):
    # st = time.time()
    # data_bank = data_merge(args.data_dir)
    data_bank = data_merge_R.data_merge(args.data_dir)

    # define train loader
    # train_set = data_bank.get_datasets(type='train', protocol=args.protocol, img_size=args.img_size, map_size=args.map_size, transform=transformer_train_pure(), debug_subset_size=args.debug_subset_size)
    train_set = data_bank.get_datasets(type='train', protocol=args.protocol, img_size=args.img_size, transform=transformer_train_pure(), debug_subset_size=args.debug_subset_size)

    num_workers = 16
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    print("Number of worker threads:", num_workers)
    max_iter = args.num_epochs*len(train_loader)
    # define model
    model = get_model(args.model_type, max_iter).cuda()
    # def optimizer
    optimizer = get_optimizer(
        args.optimizer, model, 
        lr=args.base_lr,
        momentum=args.momentum, 
        weight_decay=args.weight_decay)
    # def scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    model = nn.DataParallel(model).cuda()


    tb_root_path = os.path.join(args.result_path, args.result_name, "tb")
    check_folder(tb_root_path)
    # make dirs
    model_root_path = os.path.join(args.result_path, args.result_name, "model")
    check_folder(model_root_path)
    score_root_path = os.path.join(args.result_path, args.result_name, "score")
    check_folder(score_root_path)
    csv_root_path = os.path.join(args.result_path, args.result_name, "csv")
    check_folder(csv_root_path)

    # define loss
    binary_fuc = nn.CrossEntropyLoss()
    map_fuc = nn.MSELoss()
    contra_fun = ContrastLoss()

    # Initialize TensorBoard writer
    writer = SummaryWriter(tb_root_path)

    # metrics
    eva = {
        "best_epoch": -1,
        "best_HTER": 100,
        "best_auc": -100
    }
    for epoch in range(args.start_epoch, args.num_epochs):
        est = time.time()

        binary_loss_record = AvgrageMeter()
        constra_loss_record = AvgrageMeter()
        adv_loss_record = AvgrageMeter()
        loss_record = AvgrageMeter()
        # train
        model.train()
        for i, sample_batched in enumerate(train_loader):
            # print("HEY", sample_batched.shape)
            image_x, label, UUID = sample_batched["image_x"].cuda(), sample_batched["label"].cuda(), sample_batched["UUID"].cuda()
            if args.model_type in ["SSAN_R"]:
                rand_idx = torch.randperm(image_x.shape[0])
                cls_x1_x1, fea_x1_x1, fea_x1_x2, domain_invariant = model(image_x, image_x[rand_idx, :, :, :])
                binary_loss = binary_fuc(cls_x1_x1, label[:, 0].long())
                contrast_label = label[:, 0].long() == label[rand_idx, 0].long()
                contrast_label = torch.where(contrast_label==True, 1, -1)
                constra_loss = contra_fun(fea_x1_x1, fea_x1_x2, contrast_label)
                adv_loss = binary_fuc(domain_invariant, UUID.long())
                loss_all = binary_loss + constra_loss + adv_loss
            elif args.model_type in ["SSAN_M"]:
                map_x = sample_batched["map_x"].cuda()
                rand_idx = torch.randperm(image_x.shape[0])
                cls_x1_x1, fea_x1_x1, fea_x1_x2, domain_invariant = model(image_x, image_x[rand_idx, :, :, :])
                binary_loss = map_fuc(cls_x1_x1, map_x)
                contrast_label = label[:, 0].long() == label[rand_idx, 0].long()
                contrast_label = torch.where(contrast_label==True, 1, -1)
                constra_loss = contra_fun(fea_x1_x1, fea_x1_x2, contrast_label)
                adv_loss = binary_fuc(domain_invariant, UUID.long())
                loss_all = binary_loss + constra_loss + adv_loss
            n = image_x.shape[0]
            binary_loss_record.update(binary_loss.data, n)
            constra_loss_record.update(constra_loss.data, n)
            adv_loss_record.update(adv_loss.data, n)
            loss_record.update(loss_all.data, n)
            writer.add_scalar('training_binary_loss_by_iteration', binary_loss.data, i)
            writer.add_scalar('training_adv_loss_by_iteration', adv_loss.data, i)
            writer.add_scalar('training_constra_loss_by_iteration', constra_loss.data, i)
            writer.add_scalar('training_total_loss_by_iteration', loss_all.data, i)
            model.zero_grad()
            loss_all.backward()
            optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            # st = time.time()
            # writer.add_scalar('training_loss_by_iteration_bw', loss_all.data, i)

            if i % args.print_freq == args.print_freq - 1:
                print("epoch:{:d}, mini-batch:{:d}, lr={:.4f}, binary_loss={:.4f}, constra_loss={:.4f}, adv_loss={:.4f}, Loss={:.4f}".format(epoch + 1, i + 1, lr, binary_loss_record.avg, constra_loss_record.avg, adv_loss_record.avg, loss_record.avg))
        
        writer.add_scalar('training_binary_loss_by_epoch', binary_loss_record.avg, epoch)
        writer.add_scalar('training_constra_loss_by_epoch', constra_loss_record.avg, epoch)
        writer.add_scalar('training_adv_loss_by_epoch', adv_loss_record.avg, epoch)
        writer.add_scalar('training_loss_by_epoch', loss_record.avg, epoch)
        
        # whole epoch average
        print("epoch:{:d}, Train: lr={:f}, Loss={:.4f}".format(epoch + 1, lr, loss_record.avg))
        epe = time.time()
        print("Time spent on 1 epoch is: ", epe-est)
        scheduler.step()

        # test
        epoch_test = 1
        if epoch % epoch_test == epoch_test-1:
            test_data_dic = data_bank.get_datasets(type='val', protocol=args.protocol, img_size=args.img_size, transform=transformer_test_video(), debug_subset_size=args.debug_subset_size)
            score_path = os.path.join(score_root_path, "epoch_{}".format(epoch+1))
            check_folder(score_path)
            # for i, test_name in enumerate(test_data_dic.keys()):
                # print("[{}/{}]Testing {}...".format(i+1, len(test_data_dic), test_name))
                # test_set = test_data_dic[test_name]
            test_loader = DataLoader(test_data_dic, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
            HTER, auc_test = test(model, args, test_loader, score_path, epoch, writer, name=test_data_dic)
            if auc_test-HTER>=eva["best_auc"]-eva["best_HTER"]:
                eva["best_auc"] = auc_test
                eva["best_HTER"] = HTER
                eva["best_epoch"] = epoch+1
                model_path = os.path.join(model_root_path, "{}_p{}_best.pth".format(args.model_type, args.protocol))
                torch.save({
                    'epoch': epoch+1,
                    'state_dict':model.module.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'scheduler':scheduler,
                    'args':args,
                }, model_path)
                print("Model saved to {}".format(model_path))
            print("[Best result] epoch:{}, HTER={:.4f}, AUC={:.4f}".format(eva["best_epoch"],  eva["best_HTER"], eva["best_auc"]))
            model_path = os.path.join(model_root_path, "{}_p{}_recent.pth".format(args.model_type, args.protocol))
            torch.save({
                'epoch': epoch+1,
                'state_dict':model.module.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler,
                'args':args,
            }, model_path)
            print("Model saved to {}".format(model_path))

    # Close the TensorBoard writer
    writer.close()


def test(model, args, test_loader, score_root_path, epoch, writer, name=""):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        scores = []
        for i, sample_batched in enumerate(test_loader):
            image_x, label = sample_batched["image_x"].cuda(), sample_batched["label"].cuda()

            if args.model_type in ["SSAN_R"]:                                                   
                cls_x1_x1, fea_x1_x1, fea_x1_x2, _ = model(image_x, image_x)
                score_norm = torch.softmax(cls_x1_x1, dim=1)[:, 1]
            elif args.model_type in ["SSAN_M"]:
                pred_map, fea_x1_x1, fea_x1_x2, _ = model(image_x, image_x)
                score_norm = torch.sum(pred_map, dim=(1, 2))/(args.map_size*args.map_size)

            for ii in range(image_x.shape[0]):
                scores.append("{} {}\n".format(score_norm[ii], label[ii][0]))
                       
        map_score_val_filename = os.path.join(score_root_path, "{}_score.txt".format('patchnet'))
        print("score: write test scores to {}".format(map_score_val_filename))
        with open(map_score_val_filename, 'w') as file:
            file.writelines(scores)

        test_ACC, fpr, FRR, HTER, auc_test, test_err = performances_val(map_score_val_filename)

        print("## {} score:".format(name))
        print("epoch:{:d}, test:  val_ACC={:.4f}, HTER={:.4f}, AUC={:.4f}, val_err={:.4f}, ACC={:.4f}".format(epoch+1, test_ACC, HTER, auc_test, test_err, test_ACC))
        print("test phase cost {:.4f}s".format(time.time()-start_time))
        writer.add_scalar('validation_acc_by_epoch', test_ACC, epoch)
        writer.add_scalar('validation_hter_by_epoch', HTER, epoch)
        writer.add_scalar('validation_frr_by_epoch', FRR, epoch)
        writer.add_scalar('validation_frr_by_epoch', fpr, epoch)
        writer.add_scalar('validation_test_err_by_epoch', test_err, epoch)
    return HTER, auc_test

    

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main(args=args)