import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import os

from utils.utils import *
from model.MultiBoxLoss import MultiBoxLoss
from utils.vialPositioningDataset import vialPositioningDataset

# Log File
log_dir = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join('experiments', log_dir)
os.mkdir(log_dir)


# Learning rate scheduling
def update_lr(optimizer, epoch):
    if epoch == 0:
        lr = config.init_lr
    elif epoch == 50:
        lr = 0.0001
    elif epoch == 100:
        lr = 0.00001
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model, config):
    # Setup loss and optimizer
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay)

    # Load training dataset
    train_dataset = vialPositioningDataset(config.image_dir_train, config.train_label, image_size=config.image_size,
                                           training=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4,
                              collate_fn=train_dataset.collate_fn)

    # Load validation dataset
    val_dataset = vialPositioningDataset(config.image_dir_train, config.val_label, image_size=config.image_size)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0,
                            collate_fn=val_dataset.collate_fn)

    print('Number of training images: ', len(train_dataset))
    print('Number of validation images: ', len(val_dataset))
    print('---------------------------------')

    # Training
    best_val_loss = np.inf

    for epoch in range(config.num_epochs):
        print('\n')
        print('Starting epoch {} / {}'.format(epoch, config.num_epochs - 1))

        model.train()
        total_loss = 0.0
        total_batch = 0

        # Update learning rate
        update_lr(optimizer, epoch)
        lr = get_lr(optimizer)
        print('Learning rate {}'.format(lr))

        for (imgs, targets, labels, _) in tqdm(train_loader):
            batch_size_this_iter = imgs.size(0)
            images = imgs.to(device)
            targets = [b.to(device) for b in targets]
            labels = [l.to(device) for l in labels]

            # Forward to compute loss
            predicted_locs, predicted_scores = model(images)
            loss = criterion(predicted_locs, predicted_scores, targets, labels)

            loss_this_iter = loss.item()
            total_loss += loss_this_iter * batch_size_this_iter
            total_batch += batch_size_this_iter

            # Backward prop
            optimizer.zero_grad()
            loss.backward()

            # Update model
            optimizer.step()

        print("\t\tTrain Loss: {:.4f}".format(total_loss / float(total_batch)))

        # Validation.
        model.eval()
        val_loss = 0.0
        total_batch = 0

        for (imgs, targets, labels, _) in tqdm(val_loader):
            batch_size_this_iter = imgs.size(0)
            images = imgs.to(device)
            targets = [b.to(device) for b in targets]
            labels = [l.to(device) for l in labels]

            # Forward to compute validation loss
            with torch.no_grad():
                predicted_locs, predicted_scores = model(images)
            vloss = criterion(predicted_locs, predicted_scores, targets, labels)

            loss_this_iter = vloss.item()
            val_loss += loss_this_iter * batch_size_this_iter
            total_batch += batch_size_this_iter
        val_loss /= float(total_batch)

        # Save results

        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, 'model.pth'))

        # Print
        print('Epoch [%d/%d], Val Loss: %.4f, Best Val Loss: %.4f' % (epoch, config.num_epochs - 1, val_loss,
                                                                      best_val_loss))


if __name__ == '__main__':
    train()
