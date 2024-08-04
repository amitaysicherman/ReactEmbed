import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Configuration for the protein-molecule interaction model")

    # General settings
    parser.add_argument("--auth_token", type=str, default="", help="Authentication token for huggingface ESM3 access")
    parser.add_argument("--protein_embedding", type=str, default="ProtBertT5-xl", help="Protein embedding model to use",
                        choices=["ProtBert-BFD", "ProtBertT5-xl", "ESM-1B", "ESM2", "ESM3"])
    parser.add_argument("--molecule_embedding", type=str, default="pebchem10m", help="Molecule embedding model to use",
                        choices=["pebchem10m", "roberta", "chemberta"])
    parser.add_argument("--task_name", type=str, default="BACE", help="Name of the task to run",
                        choices=["BetaLactamase", "Fluorescence", "Stability", "HumanPPI", "BindingDB", "BACE", "BBBP",
                                 "ClinTox", "SIDER", "DrugBank", "Davis"])

    parser.add_argument("--experiment_name", type=str, default="default", help="Name of the experiment")
    parser.add_argument("--max_epochs_no_improve", type=int, default=15,
                        help="Maximum number of epochs with no improvement before early stopping")

    # Fusion model parameters
    parser.add_argument("--fusion_batch_size", type=int, default=8192, help="Batch size for fusion model training")
    parser.add_argument("--fusion_output_dim", type=int, default=1024, help="Output dimension of fusion model")
    parser.add_argument("--fusion_dropout", type=float, default=0.3, help="Dropout rate for fusion model")
    parser.add_argument("--fusion_learning_rate", type=float, default=1e-3, help="Learning rate for fusion model")
    parser.add_argument("--fusion_num_layers", type=int, default=1, help="Number of layers in fusion model")
    parser.add_argument("--fusion_hidden_dim", type=int, default=512, help="Hidden dimension in fusion model")
    parser.add_argument("--fusion_all_to_protein", type=int, default=1,
                        help="Protein anchors mechanism for fusion model (1) or not (0)")
    parser.add_argument("--fusion_train_all", type=int, default=1, help="Whether to train on all data (1) or not (0)")
    parser.add_argument("--fusion_max_epochs_no_improve", type=int, default=5,
                        help="Maximum epochs with no improvement for fusion model")
    parser.add_argument("--use_triplet_loss", type=int, default=1, help="Whether to use triplet loss (1) or not (0)")
    parser.add_argument("--fusion_epochs", type=int, default=50, help="Number of epochs to train fusion model")
    parser.add_argument("--fusion_name", type=str, default="", help="Name of the fusion model")

    # Downstream task parameters
    parser.add_argument("--downstream_batch_size", type=int, default=64, help="Batch size for downstream task")
    parser.add_argument("--downstream_learning_rate", type=float, default=5e-5,
                        help="Learning rate for downstream task")
    parser.add_argument("--downstream_num_layers", type=int, default=1, help="Number of layers in downstream model")
    parser.add_argument("--downstream_hidden_dim", type=int, default=-1,
                        help="Hidden dimension in downstream model (-1 for auto)")
    parser.add_argument("--downstream_dropout", type=float, default=0.0, help="Dropout rate for downstream model")
    parser.add_argument("--print_downstream_results", type=int, default=1,
                        help="Whether to print downstream results (1) or not (0)")
    parser.add_argument("--use_fusion_for_downstream", type=int, default=1,
                        help="Whether to use fusion model for downstream task (1) or not (0)")
    parser.add_argument("--use_pretrained_for_downstream", type=int, default=1,
                        help="Whether to use pretrained model for downstream task (1) or not (0)")

    args = parser.parse_args()

    return args
