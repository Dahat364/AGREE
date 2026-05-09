import argparse

def init():
    parser = argparse.ArgumentParser(description="PyTorch HKD-IAA AGREE Training - PARA")
    parser.add_argument('--path_to_PARA_images', type=str,
                        default='/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/IAADatasets/PARA/imgs/',
                        help='directory to images')

    parser.add_argument('--path_to_PARA_save_csv', type=str,
                        default="/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/IAADatasets/PARA/annotation/processed/",
                        help='directory to csv_folder')

    parser.add_argument('--path_to_save_ckpt', type=str,
                        default='/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/CheckPoints/MyWork/PARA/HKD-IAA-AGREE/',
                        help='directory to save ckpt')
    parser.add_argument('--path_to_text_features', type=str,
                        default='/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/IAADatasets/PARA/text_features/',
                        help='directory to pre-extracted text features')
    parser.add_argument('--swin_weight_path', type=str,
                        default='/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/weights/ELTA/pytorch_model.bin',
                        help='Swin pretrained weights path')

    parser.add_argument('--weight_decay',  type=int, default=5e-4, help='Weight decay')

    parser.add_argument('--init_lr', type=int, default=0.00003, help='learning_rate')

    # parser.add_argument('--min_lr', type=int, default=0.000001, help='min_lr')

    parser.add_argument('--num_epoch', type=int, default=100, help='epoch num for train')

    parser.add_argument('--batch_size', type=int,default=8,help='how many pictures to process one time')

    parser.add_argument('--train_num_workers', type=int, default=6, help ='num_workers')

    parser.add_argument('--test_num_workers', type=int, default=6, help='num_workers')

    parser.add_argument('--gpu_id', type=str, default='1', help='which gpu to use')
    parser.add_argument('--resume', type=str, default='',
                        help='checkpoint path for resuming training')


    args = parser.parse_args()
    return args
