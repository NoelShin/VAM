import os
import argparse
from csv import reader


class BaseOptions(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--debug', action='store_true', default=False)
        parser.add_argument('--gpu_ids', type=str, default='0,1,2')
        parser.add_argument('--manual_seed', type=int, default=0)
        parser.add_argument('--additional_name', type=str, default="-test")

        # Backbone options
        parser.add_argument('--backbone_network', type=str, default='WideResNet',
                            help='Choose among [ResNet, WideResNet, ResNext]')
        parser.add_argument('--n_layers', type=int, default=18, help='# of weight layers.')
        parser.add_argument('--n_groups', type=int, default=32, help='ResNext cardinality.')
        parser.add_argument('--widening_factor', type=float, default=8.0, help='WideResNet parameter')
        parser.add_argument('--width_multiplier', type=float, default=1.0, help='MobileNet parameter')

        # Attention options
        parser.add_argument('--attention_module', type=str, default='VAM',
                            help='Choose among [BAM, CBAM, None, SE]')
        # parser.add_argument('--conversion_factor', type=int, default=8)
        parser.add_argument('--group_size', type=int, default=2, help='TAM parameter')

        parser.add_argument('--dataset', type=str, default='ImageNet1K', help='Dataset name. Choose among'
                                                                            '[CIFAR10, CIFAR100, ImageNet1K, MSCOCO, VOC2007]')

        parser.add_argument('--epoch_recent', type=int, default=0)
        parser.add_argument('--dir_checkpoints', type=str, default='./checkpoints')
        parser.add_argument('--iter_report', type=int, default=5)
        parser.add_argument('--iter_save', type=int, default=100000)
        parser.add_argument('--n_workers', type=int, default=16)
        parser.add_argument('--path_ImageNet', type=str, default="../../data/ImageNet2012")
        parser.add_argument('--path_label_train', type=str, default="./training_WNID2class.txt")
        parser.add_argument('--path_label_val', type=str, default="./ILSVRC2012_validation_ground_truth.txt")
        parser.add_argument('--resume', action='store_true', default=False)
        parser.add_argument('--resume_top1', action='store_true', default=False)
        parser.add_argument('--resume_top5', action='store_true', default=False)
        parser.add_argument('--resume_latest', action='store_true', default=True)

        self.parser = parser

    @staticmethod
    def define_hyper_params(args):
        if args.dataset in ['CIFAR10', 'CIFAR100']:
            args.batch_size = 128
            args.epochs = 300
            args.lr = 0.1
            args.momentum = 0.9
            args.weight_decay = 1e-4

        elif args.dataset == 'ImageNet1K':
            args.batch_size = 256  # default 256
            args.dir_dataset = args.path_ImageNet
            args.epochs = 100
            args.lr = 0.1
            args.momentum = 0.9
            args.weight_decay = 1e-4

        elif args.dataset == 'SVHN':
            args.batch_size = 128
            args.lr = 0.1
            args.epochs = 160
            args.momentum = 0.9
            args.weight_decay = 1e-4

    def parse(self):
        args = self.parser.parse_args()
        self.define_hyper_params(args)

        if args.dataset != 'ImageNet1K':
            args.dir_dataset = '/mnt/dataset/ImageNet' #{}'.format(args.dataset)

        model_name = args.backbone_network
        model_name += str(args.n_layers) if 'Res' in args.backbone_network else ''
        model_name += '_' + str(args.widening_factor) if args.backbone_network == 'WideResNet' else ''
        model_name += '_' + args.attention_module
        model_name += '_' + str(args.group_size) if args.attention_module == 'TAM' else ''
        model_name += '_' + args.additional_name

        args.dir_analysis = os.path.join(args.dir_checkpoints, args.dataset, model_name, 'Analysis')
        args.dir_model = os.path.join(args.dir_checkpoints, args.dataset, model_name, 'Model')
        os.makedirs(args.dir_analysis, exist_ok=True)
        os.makedirs(args.dir_model, exist_ok=True)

        args.path_log_analysis = os.path.join(args.dir_analysis, 'log.txt')
        if os.path.isfile(args.path_log_analysis) and not args.debug:
            answer = input("Already existed log {}. Do you want to overwrite it? [y/n] : ".format(model_name))
            if answer == 'y':
                with open(os.path.join(args.dir_analysis, 'train.txt'), 'wt') as log:
                    log.write('Epoch, Loss, Top1, Top5\n')
                    log.close()

                with open(args.path_log_analysis, 'wt') as log:
                    log.write('Epoch, Epoch_best_top1,  Epoch_best_top5, Training_loss, Top1_error, Top5_error\n')
                    log.close()

                with open(os.path.join(args.dir_model, 'options.txt'), 'wt') as log:
                    opt = vars(args)
                    print('-' * 50 + 'Options' + '-' * 50)
                    for k, v in sorted(opt.items()):
                        print(k, v)
                        log.write(str(k) + ', ' + str(v) + '\n')
                    print('-' * 107)
                    log.close()

            else:
                answer = input("Do you want to resume training of {}? [y/n] : ".format(model_name))
                if answer == 'y':
                    args.resume = True
                    best_top1 = {'Epoch': 0, 'Top1': 100.0}
                    best_top5 = {'Epoch': 0, 'Top5': 100.0}
                    epoch_recent = 0
                    with open(args.path_log_analysis, 'r') as log:
                        for i, row in enumerate(reader(log)):
                            if i == 0:
                                continue
                            else:
                                if int(row[1]) > int(best_top1['Epoch']):
                                    best_top1.update({'Epoch': row[1], 'Top1': float(row[4])})
                                if int(row[2]) > int(best_top5['Epoch']):
                                    best_top5.update({'Epoch': row[2], 'Top5': float(row[5])})
                            epoch_recent = int(row[0])
                        args.epoch_top1 = int(best_top1['Epoch'])
                        args.top1 = best_top1['Top1']

                        args.epoch_top5 = int(best_top5['Epoch'])
                        args.top5 = best_top5['Top5']
                        log.close()

                    if args.resume_latest:
                        assert os.path.isfile(os.path.join(args.dir_model, 'latest.pt'))
                        args.epoch_recent = epoch_recent
                        args.path_model = os.path.join(args.dir_model, 'latest.pt')
                        # print("Training resumes at epoch", args.epoch_recent)

                    elif args.resume_top1:
                        # if args.epoch_top1 > args.epoch_top5:
                        args.epoch_recent = args.epoch_top1
                        args.path_model = os.path.join(args.dir_model, 'top1_best.pt')
                    elif args.resume_top5:
                        args.epoch_recent = args.epoch_top5
                        args.path_model = os.path.join(args.dir_model, 'top5_best.pt')
                    print("Training resumes at epoch", args.epoch_recent)

                else:
                    NotImplementedError
        else:
            with open(os.path.join(args.dir_analysis, 'train.txt'), 'wt') as log:
                log.write('Epoch, Loss, Top1, Top5\n')
                log.close()

            with open(args.path_log_analysis, 'wt') as log:
                log.write('Epoch, Epoch_best_top1,  Epoch_best_top5, Training_loss, Top1_error, Top5_error\n')
                log.close()

            with open(os.path.join(args.dir_model, 'options.txt'), 'wt') as log:
                opt = vars(args)
                print('-' * 50 + 'Options' + '-' * 50)
                for k, v in sorted(opt.items()):
                    print(k, v)
                    log.write(str(k) + ', ' + str(v) + '\n')
                print('-' * 107)
                log.close()

        return args


if __name__ == '__main__':
    opt = BaseOptions().parse()
