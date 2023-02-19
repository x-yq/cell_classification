import argparse

parser = argparse.ArgumentParser( description="Model parameters for cell type classification.")

parser.add_argument("--model", default="efficientnet_b0", help="Models")
parser.add_argument("--img_size", type=int, default=224, help="Required image size of model")

parser.add_argument("--trial_name", type=str, default="", help="name for training run")
parser.add_argument("--output_folder", type=str, default="", help="Folder to store the outputs of the training process")
parser.add_argument("--epochs", type=int, default=100, help="epochs to train")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=int, default=1e-4, help="learning rate")

parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
parser.add_argument("--workers", type=int, default=4, help="Num workers")
parser.add_argument("--pin_memory", type=int, default=True, help="Pin memory for dataloader")
parser.add_argument("--psudo_labels_csv", type=str, default="iter_1.csv", help="Current psudo labels file")
parser.add_argument("--train_splitting", type=float, default=0.6, help="Labeled data splitting")
parser.add_argument("--stoch_depth_prob", type=float, default=0.3, help="stochastic_depth_prob of student model")
parser.add_argument("--drop_out_prob", type=float, default=0.5, help="stochastic_depth_prob of student model")
parser.add_argument("--threshold", type=float, default=210, help="Maximal ratio of major class to minor class")
parser.add_argument("--num_psudo_labels", type=int, default=15000, help="Number of unlabeled data")