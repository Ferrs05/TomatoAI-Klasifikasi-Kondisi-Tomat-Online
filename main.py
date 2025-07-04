import argparse
from model_cnn import tune_and_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training CNN Tomat dengan hyperparameter tuning')
    parser.add_argument('--train_dir', type=str, default='data/train', help='Folder data latih')
    parser.add_argument('--test_dir', type=str, default='data/test', help='Folder data uji')
    parser.add_argument('--trials', type=int, default=5, help='Max trials untuk tuner')
    parser.add_argument('--epochs', type=int, default=20, help='Jumlah epoch akhir')
    args = parser.parse_args()

    tune_and_train(
        args.train_dir,
        args.test_dir,
        max_trials=args.trials,
        epochs=args.epochs
    )