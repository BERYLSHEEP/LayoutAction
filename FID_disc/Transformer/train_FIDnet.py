from cgi import test
from genericpath import exists
import os
import argparse
os.environ['OMP_NUM_THREADS'] = '1'  # noqa

import torch
from dataset import RicoLayout
from trainer import TrainerConfig
from model import FID_GPTnet, GPTConfig
from FID_trainer import FID_Trainer, Eval

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--exp", default="layout", help="experiment name")
    parser.add_argument("--log_dir", default="./logs", help="/path/to/logs/dir")
    parser.add_argument("--dataset", choices=["Rico", "PubLayNet", "MNIST", "COCO"], default="Rico", const='bbox',nargs='?')
    
    # test
    parser.add_argument("--is_test", type=bool, default=False)
    parser.add_argument("--model_path", type=str, default=None)

    # Rico options
    parser.add_argument("--train_dir", default="./instances_train", help="/path/to/train")
    parser.add_argument("--val_dir", default="./instances_val", help="/path/to/val")
    parser.add_argument("--test_dir",  default="./instances_test", help="/path/to/test")
    
    # FID metircs
    parser.add_argument("--pkl_path", type=str, default="")
    
    # Layout options
    parser.add_argument("--max_length", type=int, default=128, help="batch size")
    parser.add_argument('--precision', default=8, type=int)
    parser.add_argument('--element_order', default='raster')
    parser.add_argument('--attribute_order', default='cxywh')

    # Architecture/training options
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--iteration", type=int, default=int(1e+5), help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=4.5e-06, help="learning rate")
    parser.add_argument('--n_layer', default=4, type=int)
    parser.add_argument('--n_embd', default=512, type=int)
    parser.add_argument('--n_head', default=4, type=int)
    # parser.add_argument('--evaluate', action='store_true', help="evaluate only")
    parser.add_argument('--lr_decay', action='store_true', help="use learning rate decay")
    parser.add_argument('--warmup_iters', type=int, default=0, help="linear lr warmup iters")
    parser.add_argument('--final_iters', type=int, default=0, help="cosine lr final iters")
    parser.add_argument('--sample_every', type=int, default=100, help="sample every epoch")

    parser.add_argument('--command', type=str, choices=["train", "cal_FID", "cal_acc"], default="train")
    parser.add_argument('--is_enc_dec', default=False, action='store_true')
    parser.add_argument('--add_triplet_loss', action='store_true', default=False)

    args = parser.parse_args()

    log_dir = os.path.join(args.log_dir, args.exp)
    samples_dir = os.path.join(log_dir, "samples")
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_dataset = RicoLayout(args.train_dir)

    max_epoch = args.iteration * args.batch_size / len(train_dataset)
    max_epoch = int(torch.ceil(torch.tensor(max_epoch)).item())
    args.epochs = max_epoch

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_length,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      is_enc_dec=args.is_enc_dec)  # a GPT-1
    model = FID_GPTnet(mconf)
    tconf = TrainerConfig(max_epochs=args.epochs,
                          batch_size=args.batch_size,
                          lr_decay=args.lr_decay,
                          learning_rate=args.lr * args.batch_size,
                          warmup_iters=args.warmup_iters,
                          final_iters=args.final_iters,
                          ckpt_dir=ckpt_dir,
                          samples_dir=samples_dir,
                          sample_every=args.sample_every,
                          model_path=args.model_path,
                          evaluate_layout_path=os.path.join(log_dir, "generated_layout.pth"),
                          add_triplet_loss=args.add_triplet_loss)
    
    if args.command == "train":
        train_fake_dataset = RicoLayout(args.train_dir, is_fake=True)
        val_dataset = RicoLayout(args.test_dir, max_length=train_dataset.max_length)
        test_dataset = RicoLayout(args.val_dir, max_length=train_dataset.max_length)
        test_fake_dataset = RicoLayout(args.val_dir, max_length=test_dataset.max_length, is_fake=True)

        trainer = FID_Trainer(model, train_dataset, train_fake_dataset, test_dataset, test_fake_dataset, val_dataset, tconf, args)

        trainer.train()

    elif args.command == "cal_FID":
        train_dataset = RicoLayout(args.train_dir)
        test_dataset = RicoLayout(args.test_dir, max_length=train_dataset.max_length)
        test_fake_dataset = RicoLayout(args.test_dir, max_length=train_dataset.max_length, is_fake=True)
        val_dataset = RicoLayout(args.val_dir,  max_length=train_dataset.max_length)

        evaler = Eval(model=model, test_dataset=test_dataset, val_dataset=val_dataset, test_fake_dataset=test_fake_dataset, pkl_path=args.pkl_path, config=tconf)
        evaler.FID_cal()

    elif args.command == "cal_acc":
        test_dataset = RicoLayout(args.test_dir, max_length=train_dataset.max_length)
        test_fake_dataset = RicoLayout(args.test_dir, max_length=test_dataset.max_length, is_fake=True)

        evaler = Eval(model=model, test_dataset=test_dataset, test_fake_dataset=test_fake_dataset, config=tconf)
        evaler.cal_acc()
    

if __name__ == "__main__":
    main()