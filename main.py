from StyleGAN_2 import StyleGAN
import argparse
from utils import *

os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
config=tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction=0.75

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of StyleGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='edit_test', help='[train, optimize_all, edit_test]')
    parser.add_argument('--dataset', type=str, default='New', help='The dataset name what you want to generate')

    parser.add_argument('--iteration', type=int, default=120, help='The number')
    parser.add_argument('--max_iteration', type=int, default=2500, help='The total number')

    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch')
    parser.add_argument('--gpu_num', type=int, default=1, help='The number of gpu')

    parser.add_argument('--progressive', type=str2bool, default=True, help='use progressive training')
    parser.add_argument('--sn', type=str2bool, default=False, help='use spectral normalization')

    parser.add_argument('--start_res', type=int, default=8, help='The number of starting resolution')
    parser.add_argument('--img_size', type=int, default=1024, help='The target size of image')


    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--update_dir', type=str, default='update',
                        help='Directory name to save the update_checkpoint')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        gan = StyleGAN(sess, args)

        # build graph
        gan.build_model()
        # show network architecture
        show_all_variables()

        if args.phase == 'train' :
            # launch the graph in a session
            gan.train()
            print(" [*] Training finished!")

        if args.phase == 'optimize_all' :
            gan.optimize_all()
            print(" [*] optimize finished!")
        if args.phase == 'edit_test' :
            gan.edit_test()
            print(" [*] Edit finished!")


if __name__ == '__main__':
    main()
