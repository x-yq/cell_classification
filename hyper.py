from argparse import ArgumentParser

# class ArgsParser:
#     def __init__(self) -> None:
#         self.parser = argparse.ArgumentParser( description="Model parameters for cell type classification.")
#         self.add_arg()

#     def add_arg(self):
#         self.parser.add_argument("--trial_name", type=str, default="", help="name for training run")
#         self.parser.add_argument("--img_size", type=int, default=224, help="Required image size of model")
#         self.parser.add_argument("--labeled_anno_file", type=str, default="", help="The path to labeled cells csv file.")
#         self.parser.add_argument("--unlabeled_anno_file", type=str, default="", help="The path to unlabeled cells csv file.")
#         self.parser.add_argument("--all_anno_file", type=str, default="", help="The path to all cells csv file.")
#         self.parser.add_argument("--output_folder", type=str, default="", help="Folder to store the outputs of the training process")
#         self.parser.add_argument("--image_folder", type=str, default="", help="The path to the folder of images")
#         self.parser.add_argument("--psudo_labels_csv", type=str, default="", help="Current file storing psudo labels")

#         self.parser.add_argument("--epochs", type=int, default=100, help="epochs to train")
#         self.parser.add_argument("--batch_size", type=int, default=64, help="batch size")
#         self.parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
#         self.parser.add_argument("--workers", type=int, default=4, help="Num workers")
#         self.parser.add_argument("--pin_memory", type=int, default=True, help="Pin memory for dataloader")
#         self.parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")

#         self.parser.add_argument("--threshold", type=int, default=200000, help="Maximal ratio of major class to minor class")
#         self.parser.add_argument("--num_psudo_labels", type=int, default=8650, help="Number of unlabeled data")
#         self.parser.add_argument("--ratio_label_unlabel", type=str, default="1:1:2:3", help="ratio of labeled data to unlabeled data in 3 iterations")
#         self.parser.add_argument("--stoch_depth_prob", type=float, default=0.3, help="stochastic_depth_prob of student model")
#         self.parser.add_argument("--drop_out_rate", type=float, default=0.5, help="drop_out_rate of student model")

parser = ArgumentParser( description="Model parameters for cell type classification.")

parser.add_argument("--trial_name", type=str, default="", help="name for training run")
parser.add_argument("--model_name", type=str, default="efficientnet_b0", help="name of models, which could be:")
parser.add_argument("--img_size", type=int, default=224, help="Required image size of model")
parser.add_argument("--labeled_anno_file", type=str, default="", help="The path to labeled cells csv file.")
parser.add_argument("--unlabeled_anno_file", type=str, default="", help="The path to unlabeled cells csv file.")
parser.add_argument("--output_folder", type=str, default="", help="Folder to store the outputs of the training process")
parser.add_argument("--image_folder", type=str, default="", help="The path to the folder of images")
parser.add_argument("--psudo_labels_csv", type=str, default="", help="Current file storing psudo labels")
parser.add_argument("--epochs", type=int, default=100, help="epochs to train")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("--workers", type=int, default=4, help="Num workers")
parser.add_argument("--pin_memory", type=int, default=True, help="Pin memory for dataloader")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
parser.add_argument("--threshold", type=int, default=200000, help="Maximal ratio of major class to minor class")
parser.add_argument("--num_psudo_labels", type=int, default=8650, help="Number of unlabeled data")
parser.add_argument("--ratio_label_unlabel", type=str, default="1:1:2:3", help="ratio of labeled data to unlabeled data in 3 iterations")
parser.add_argument("--stoch_depth_prob", type=float, default=0.3, help="stochastic_depth_prob of student model")
parser.add_argument("--drop_out_rate", type=float, default=0.5, help="drop_out_rate of student model")

