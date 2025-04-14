import onnxruntime as ort
import numpy as np
import torch
import argparse
import onnx
from data.datasets import build_dataset
from timm.utils import accuracy
from main import str2bool
import utils  

def get_args_parser():
    parser = argparse.ArgumentParser('CAS-ViT training and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='rcvit_xs', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")

    
    
    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                        help='Random erase prob (default: 0.0)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')


    # Dataset parameters

    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=200, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='image_folder', choices=['IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=43, type=int)

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')



    parser.add_argument('--min_crop_size_w', default=160, type=int)
    parser.add_argument('--max_crop_size_w', default=320, type=int)
    parser.add_argument('--min_crop_size_h', default=160, type=int)
    parser.add_argument('--max_crop_size_h', default=320, type=int)

    return parser

def evaluate_onnx(onnx_model_path, data_loader):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test'
    # Create ONNX Runtime session
    #so = ort.SessionOptions()
    #so.intra_op_num_threads = 4
    #so.inter_op_num_threads = 4
    #so.execution_mode = ort.ExecutionMode.ORT_PARALLEL  # Optional: disables parallelism

    session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
    print("Session providers:", session.get_providers())
    print("Current execution provider:", session.get_providers()[0])
    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print("Start eval")
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        # Convert the input tensor to numpy array
        input_numpy = images.to('cpu').contiguous().numpy().astype(np.float32)

        # Run inference
        outputs = session.run([output_name], {input_name: input_numpy})
        # Process the outputs as needed
        # For example, you can convert the output to a tensor and calculate loss or accuracy
        output_torch=torch.from_numpy(outputs[0])
        loss = criterion(output_torch, target)
        acc1, acc5 = accuracy(output_torch, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

# Path to the ONNX model
if __name__ == "__main__":
    parser = argparse.ArgumentParser('onnx run', parents=[get_args_parser()])
    args = parser.parse_args()
    onnx_model_path = args.resume
    dataset_val, _ = build_dataset(is_train=False,args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    #model checker
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    #eval
    evaluate_onnx(onnx_model_path, data_loader_val)
