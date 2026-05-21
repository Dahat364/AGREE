import argparse


def init():
    parser = argparse.ArgumentParser(description="PyTorch EAMB-Net AGREE Training - AADB")
    
    # 数据路径 (AADB)
    parser.add_argument('--path_to_images', type=str,
                        default='/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/IAADatasets/AADB/images/',
                        help='directory to images')
    parser.add_argument('--path_to_save_csv', type=str,
                        default="/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/IAADatasets/AADB/datasets/",
                        help='directory to csv_folder')
    parser.add_argument('--path_to_text_features', type=str,
                        default='/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/IAADatasets/AADB/text_features/',
                        help='directory to pre-extracted text features')
    parser.add_argument('--path_to_save_ckpt', type=str,
                        default='/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/CheckPoints/MyWork/AADB/EAMB-Net-AGREE/',
                        help='directory to save ckpt')

    # 训练参数
    parser.add_argument('--experiment_dir_name', type=str, default='.',
                        help='directory to project')
    parser.add_argument('--path_to_model_weight', type=str, default='',
                        help='directory to pretrain model')
    
    # 分层学习率
    parser.add_argument('--init_lr_emotion', type=float, default=0.000001, help='learning rate for emotion model')
    parser.add_argument('--init_lr_visual', type=float, default=0.00001, help='learning rate for visual backbone')
    parser.add_argument('--init_lr_fusion', type=float, default=0.00001, help='learning rate for fusion modules')
    parser.add_argument('--init_lr_head', type=float, default=0.0001, help='learning rate for prediction head')
    parser.add_argument('--init_lr', type=float, default=0.00003, help='default learning rate')
    
    parser.add_argument('--num_epoch', type=int, default=100, help='epoch num for train')
    parser.add_argument('--batch_size', type=int, default=2, help='how many pictures to process one time')
    parser.add_argument('--num_workers', type=int, default=6, help='num_workers')
    parser.add_argument('--gpu_id', type=str, default='0', help='which gpu to use')
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint for resuming')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')

    args = parser.parse_args()
    return args