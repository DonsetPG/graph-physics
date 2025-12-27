from model_with_layer_noise import MPNNs

def parse_method(args, n, c, d, device):
    
    model = MPNNs(d, args.hidden_channels, c, local_layers=args.local_layers, dropout=args.dropout, 
    heads=args.num_heads, pre_ln=args.pre_ln, pre_linear=args.pre_linear, res=args.res, res_x_only=getattr(args, "res_x_only", False), ln=args.ln, bn=args.bn, jk=args.jk, gnn = args.gnn, noise_std=args.noise_std, lani_std=args.lani_std, lani_k=args.lani_k,
    dropout_type=args.dropout_type, lani_dropout_k=args.lani_dropout_k, lani_dropout_n_iter=args.lani_dropout_n_iter, lani_dropout_mode=args.lani_dropout_mode, lani_dropout_dense_threshold=args.lani_dropout_dense_threshold).to(device)
    
    return model
        

def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='roman-empire')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--rand_split', action='store_true',
                        help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'],
                        help='evaluation metric')
    parser.add_argument('--model', type=str, default='MPNN')
    # GNN
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=7)
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--pre_ln', action='store_true')
    parser.add_argument('--pre_linear', action='store_true')
    parser.add_argument('--res', action='store_true', help='use residual connections for GNNs')
    parser.add_argument('--res_x_only', action='store_true',
                        help='if set, residual connection adds +x (identity) instead of +Linear(x); requires matching feature dims (e.g. use --pre_linear)')
    parser.add_argument('--ln', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--bn', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--jk', action='store_true', help='use JK for GNNs')
    parser.add_argument('--noise_std', '--noise-std', type=float, default=0.0, help='stddev of zero-mean isotropic Gaussian noise added after each layer')
    parser.add_argument('--lani_std', type=float, default=0.0, help='power budget of Laplacian-aligned noise added after each layer')
    parser.add_argument('--lani_k', type=int, default=5, help='number of top Laplacian eigenvectors to align noise with')
    
    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--dropout_type', type=str, default='vanilla', choices=['vanilla', 'lani'],
                        help='feature dropout type: vanilla (iid) or lani (Laplacian-aligned)')
    parser.add_argument('--lani_dropout_k', type=int, default=8,
                        help='LANI dropout: number of high-frequency eigenvectors')
    parser.add_argument('--lani_dropout_n_iter', type=int, default=30,
                        help='LANI dropout: subspace iterations for eigenvector approximation')
    parser.add_argument('--lani_dropout_mode', type=str, default='feature', choices=['feature', 'node'],
                        help="LANI dropout mode: 'feature' (per-feature mask) or 'node' (drop whole nodes)")
    parser.add_argument('--lani_dropout_dense_threshold', type=int, default=0,
                        help='LANI dropout: if num_nodes <= threshold, use exact dense EVD (0 = always approximate)')
    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=100, help='how often to print')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='./model/', help='where to save model')
