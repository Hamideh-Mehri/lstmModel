import sys
sys.path.insert(0, '/users/fs2/hmehri/pythonproject/Thesis/synthetic')

from lib.prepare_data import preprocess_data_czech
from lib.field_info import FieldInfo, FIELD_INFO_TCODE, FieldInfo_type2
from lib.tensor_encoder import TensorEncoder
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from lib.modules import Encoder_Decoder_lstm
from trainlstm import Train
import json
import random
import os

# Set seeds
# random.seed(0)
# np.random.seed(0)
# tf.random.set_seed(0)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def log_parameters(filename, parameters):
    log_entry = {'filename': filename, 'parameters': parameters}
    with open('parameter_log.json', 'a') as file:
        json.dump(log_entry, file)
        file.write('\n')  # New line for each entry


def make_batches(ds, buffer_size, batch_size):
    return ds.cache().shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

def create_tensor_dataset(encoder,bs, split=True):
    """bs is Batch Size
       if split=True, the input data is split into train and validation, otherwise the whole data is used for training """
    n_seqs, _, _ = encoder.inp_tensor.shape

    x_tr, x_cv, inds_tr, inds_cv, targ_tr, targ_cv = train_test_split(encoder.inp_tensor, np.arange(n_seqs), encoder.tar_tensor, test_size=0.2)

    # Create TensorFlow dataset
    ds_all = tf.data.Dataset.from_tensor_slices((encoder.inp_tensor.astype(np.float32), encoder.tar_tensor.astype(np.float32)))
    ds_tr = tf.data.Dataset.from_tensor_slices((x_tr.astype(np.float32), targ_tr.astype(np.float32)))
    ds_cv = tf.data.Dataset.from_tensor_slices((x_cv.astype(np.float32), targ_cv.astype(np.float32)))

    BUFFER_SIZE = ds_all.cardinality().numpy()

    all_batches =   make_batches(ds_all, BUFFER_SIZE, bs)
    train_batches = make_batches(ds_tr, BUFFER_SIZE, bs)
    val_batches =  make_batches(ds_cv, BUFFER_SIZE, bs)

    if split:
        return train_batches, val_batches
    else:
        return all_batches


def main():
    with tf.device('/gpu:0'):
        raw_data = pd.read_csv('../DATA/tr_by_acct_w_age.csv')
        data, LOG_AMOUNT_SCALE, TD_SCALE,ATTR_SCALE, START_DATE, _ = preprocess_data_czech(raw_data)
        data2 = data[['account_id','age','age_sc', 'tcode', 'tcode_num', 'datetime', 'month', 'dow', 'day','td', 'dtme', 'log_amount','log_amount_sc','td_sc',
                                   'type','operation', 'k_symbol', 'type_num', 'operation_num', 'k_symbol_num']]
        #data2 =  data[['account_id', 'tcode_num', 'age_sc', 'tcode', 'age']]
        df= data2.copy()

        confighyper = load_config('config_hyper.json')
        max_seq_len = confighyper['max_seq_len']
        min_seq_len = confighyper['min_seq_len']
        batch_size = confighyper['batch_size']
        epochs = confighyper['epochs'] 
        early_stop = confighyper['early_stop'] 
        len_generated_seq = confighyper['len_generated_seq'] 
        num_generated_seq = confighyper['num_generated_seq'] 
        synth_data_filename = confighyper["synth_data_filename"]
        strategy = confighyper['strategy']

        info = FieldInfo(strategy)
        #info = FIELD_INFO_TCODE()
        #info = FieldInfo_type2()

        
        encoder = TensorEncoder(df, info, max_seq_len, min_seq_len)
        encoder.encode_with_overlap(slide_step=80)

        n_seqs, seq_len, n_feat_inp = encoder.inp_tensor.shape
        raw_features = encoder.tar_tensor.shape[-1]    #7

        train_batches, val_batches = create_tensor_dataset(encoder, batch_size, split=True)

    
    
        config = {}
        config["ORDER"] = info.DATA_KEY_ORDER
        config["FIELD_STARTS_IN"] = info.FIELD_STARTS_IN
        config["FIELD_DIMS_IN"] = info.FIELD_DIMS_IN
        config["FIELD_STARTS_NET"] = info.FIELD_STARTS_NET
        config["FIELD_DIMS_NET"] = info.FIELD_DIMS_NET
        config["ACTIVATIONS"] = info.ACTIVATIONS

        lstm = Encoder_Decoder_lstm(config, n_feat_inp, conditional=True)
        train = Train(lstm)
        train.train(train_batches, val_batches, epochs=epochs, early_stop=early_stop)
        attributes = encoder.attributes

        synth = train.generate_synthetic_data(len_generated_seq, num_generated_seq, df, attributes, n_feat_inp)
        #synth = train.generate_synthetic_tcode(len_generated_seq,num_generated_seq, df, attributes, n_feat_inp)
        #synth = train.generate_synthetic_data_type2(len_generated_seq,num_generated_seq, df, attributes, n_feat_inp)

        synth.to_csv(synth_data_filename, index=False)
        
        log_parameters(synth_data_filename, confighyper)
        print('finish')
        


if __name__ == "__main__":
    main()


