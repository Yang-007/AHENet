import os

exp_name = 'AHENet_txt1'

lr = 1e-4

dataset_name = 'CVC-ClinicDB'

data_root = os.path.join('..', 'dataset', dataset_name)
train_dir = os.path.join(data_root, 'train1.txt')
test_dir = os.path.join(data_root, 'test1.txt')

if dataset_name == 'CVC-ClinicDB':
    train_batch_size = 16
    test_batch_size = 1
    epoch = 4000
    output_size = [256, 256]
    base_channel = 64
    inter_channel = base_channel // 2
    rdb_block_num = 4
    model_level = 4
    lr_decay = True
elif dataset_name == 'CVC-ColonDB':
    train_batch_size = 4
    test_batch_size = 1
    epoch = 4000
    output_size = [256, 256]
    base_channel = 64
    inter_channel = base_channel // 2
    rdb_block_num = 4
    model_level = 4
    lr_decay = True
else:
    assert False

test_per_ep = 1

weight_root = './weights/'
weight_dir = os.path.join(weight_root, '%s_%s' % (dataset_name, exp_name))
log_path = os.path.join(weight_dir, 'log.txt')
reload_from_checkpoint = False
reload_path = os.path.join(weight_dir, 'NewestNet.pth')
finetune_path = os.path.join(weight_dir, 'BestNet.pth')
start_epoch = 0
print('Exp Name:', exp_name)
print('Dataset:', dataset_name)
print('Log Path:', log_path)
print('Model Level:', model_level)
#print('Input Size:', input_size)
print('Epoch:', epoch)
