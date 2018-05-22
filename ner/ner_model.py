# -*- coding: utf-8 -*-
"""
Created on Wed May  2 10:09:17 2018

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os, argparse, time, re
from models.model import BiLSTM_CRF
from models.utils import str2bool, get_logger, infer
from models.data import read_corpus, read_dictionary, tag2label, random_embedding, get_vocab


def run_ner_model(train_data='data_path', test_data='data_path', batch_size=64, epoch=30, hidden_dim=300, optimizer='Adam', 
                  CRF=True, lr=0.001, clip=5.0, dropout=0.5, update_embedding=True, pretrain_embedding='random', 
                  embedding_dim=300, shuffle=True, mode='pred', demo_model='1523352428'):
    '''
    识别医学命名实体
    
    输入参数：
        --train_data, type=str, help='train data source'
        --test_data, type=str, help='test data source'
        --batch_size, type=int, help='#sample of each minibatch'
        --epoch, type=int, help='#epoch of training'
        --hidden_dim, type=int, help='#dim of hidden state'
        --optimizer, type=str, help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD'
        --CRF, type=bool, help='use CRF at the top layer. if False, use Softmax'
        --lr, type=float, help='learning rate'
        --clip, type=float, help='gradient clipping'
        --dropout, type=float, help='dropout keep_prob'
        --update_embedding, type=bool, help='update embedding during training'
        --pretrain_embedding, type=str, help='use pretrained char embedding or init it randomly'
        --embedding_dim', type=int, help='random init char embedding_dim'
        --shuffle, type=bool, help='shuffle training data before each epoch'
        --mode, type=str, help='train/test/demo/pred'
    输出：
        --mode='train'时，输出训练模型与回测结果，保存在data_path_save路径下
        --mode='test'时，输出测试集上的模型预测结果，保存在data_path_save路径下
        --mode='demo'时，展示实体识别效果，在console中输入任意一句话，输出模型自动标记出实体的标签
        --mode='pred'时，输出需要标注的CSV文件列表（在original_data路径下），输出标注完毕后的CSV文件（在predictions路径下）
    '''
    
    ## Session configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # default: 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

    
    ## hyperparameters
    parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
    parser.add_argument('--train_data', type=str, default=train_data, help='train data source')
    parser.add_argument('--test_data', type=str, default=test_data, help='test data source')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='#sample of each minibatch')
    parser.add_argument('--epoch', type=int, default=epoch, help='#epoch of training')
    parser.add_argument('--hidden_dim', type=int, default=hidden_dim, help='#dim of hidden state')
    parser.add_argument('--optimizer', type=str, default=optimizer, help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
    parser.add_argument('--CRF', type=str2bool, default=CRF, help='use CRF at the top layer. if False, use Softmax')
    parser.add_argument('--lr', type=float, default=lr, help='learning rate')
    parser.add_argument('--clip', type=float, default=clip, help='gradient clipping')
    parser.add_argument('--dropout', type=float, default=dropout, help='dropout keep_prob')
    parser.add_argument('--update_embedding', type=str2bool, default=update_embedding, help='update embedding during training')
    parser.add_argument('--pretrain_embedding', type=str, default=pretrain_embedding, help='use pretrained char embedding or init it randomly')
    parser.add_argument('--embedding_dim', type=int, default=embedding_dim, help='random init char embedding_dim')
    parser.add_argument('--shuffle', type=str2bool, default=shuffle, help='shuffle training data before each epoch')
    parser.add_argument('--mode', type=str, default=mode, help='train/test/demo/pred')
    parser.add_argument('--demo_model', type=str, default=demo_model, help='model for test, demo and pred')
    args = parser.parse_args()
    
    
    ## get char embeddings
    if args.pretrain_embedding == 'random':
        word2id = read_dictionary(os.path.join(args.train_data, 'word2id.pkl'))
        embeddings = random_embedding(word2id, args.embedding_dim)
    else:
        word2id = get_vocab(os.path.join(args.train_data, 'vocab.txt'))
        _embeddings = pd.read_csv('data_path/vectors.txt', encoding='utf8', sep=' ')
        embeddings = np.array(_embeddings.iloc[:,1:], dtype='float32')
    
    
    ## read corpus and get training data
    if args.mode == 'train' or args.mode == 'test':
        train_path = os.path.join(args.train_data, 'train_data.txt')
        test_path = os.path.join(args.test_data, 'test_data.txt')
        train_data = read_corpus(train_path)
        test_data = read_corpus(test_path); test_size = len(test_data)
    
    
    ## paths setting
    paths = {}
    timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
    output_path = os.path.join(args.train_data+"_save", 'ner_'+timestamp)
    if not os.path.exists(output_path): os.makedirs(output_path)
    summary_path = os.path.join(output_path, "summaries")
    paths['summary_path'] = summary_path
    if not os.path.exists(summary_path): os.makedirs(summary_path)
    model_path = os.path.join(output_path, "checkpoints/")
    if not os.path.exists(model_path): os.makedirs(model_path)
    ckpt_prefix = os.path.join(model_path, "model")
    paths['model_path'] = ckpt_prefix
    result_path = os.path.join(output_path, "results")
    paths['result_path'] = result_path
    if not os.path.exists(result_path): os.makedirs(result_path)
    log_path = os.path.join(result_path, "log.txt")
    paths['log_path'] = log_path
    get_logger(log_path).info(str(args))
    
    
    ## training model
    if args.mode == 'train':
        model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
        model.build_graph()
    
        # hyperparameters-tuning, split train/dev
        dev_data = train_data[:1000]; dev_size = len(dev_data)
        train_data = train_data[1000:]; train_size = len(train_data)
        print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
        model.train(train=train_data, dev=dev_data)
    
        ## train model on the whole training data
    #    print("train data: {}".format(len(train_data)))
    #    model.train(train=train_data, dev=dev_data)  # use test_data as the dev_data to see overfitting phenomena
    
    ## testing model
    elif args.mode == 'test':
        ckpt_file = tf.train.latest_checkpoint(model_path)
        print(ckpt_file)
        paths['model_path'] = ckpt_file
        model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
        model.build_graph()
        print("test data: {}".format(test_size))
        model.test(test_data)
    
    ## demo
    elif args.mode == 'demo':
        ckpt_file = tf.train.latest_checkpoint(model_path)
        print(ckpt_file)
        paths['model_path'] = ckpt_file
        model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
        model.build_graph()
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            print('============= demo =============')
            saver.restore(sess, ckpt_file)
            r='[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`ò¢{|}~！？，。“”、\d\n：\tⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ（）【】～； ]'
            
            while(1):
                print('Please input your sentence:')
                demo_sent = input()
                if demo_sent == '' or demo_sent.isspace():
                    print('See you next time!')
                    break
                else:
                    demo_sent = re.sub(r,'',demo_sent)
                    demo_sent = list(demo_sent.strip())
                    demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                    tag = model.demo_one(sess, demo_data)
                    preds=zip(demo_data[0][0], tag)
                    content=''
                    for item in preds:
                        print(item)
                        if item[1]!=0:
                            content+=item[0]+'\t'+item[1]+'\n'
                        else:
                            content+=item[0]+'\t'+'O'+'\n'
                    body, dis, des=infer(content)
                    body_str='，'.join(body)
                    dis_str='，'.join(dis)
                    des_str='，'.join(des)
                    print('\n提取出的标签：')
                    print('BODY:{}'.format(body_str))
                    print('DISEASE:{}'.format(dis_str))
                    print('DESCRIPTION:{}'.format(des_str))
    
    ## predict
    elif args.mode == 'pred':
        ckpt_file = tf.train.latest_checkpoint(model_path)
        print(ckpt_file)
        paths['model_path']=ckpt_file
        model=BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
        model.build_graph()
        saver=tf.train.Saver()
        with tf.Session(config=config) as sess:
            print('============= predict ==============')
            saver.restore(sess, ckpt_file)
            r='[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`ò¢{|}~！？，。“”、\d\n：\tⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ（）【】～；]'
            
            files=os.listdir('data_path/original_data/')
            for file in files:
                f=open('data_path/original_data/'+file, encoding='utf8')
                fr=pd.read_csv(f, header=0)
                fr=fr.dropna()
                fw=pd.DataFrame(columns=['WYSText', 'WYGText', 'BODY', 'DISEASE', 'DESCRIPTION'])
                
                print(file+' starts to generate tags!!')
                for i in range(len(fr)):
                    line=[]
                    sent=list(fr.iloc[i,1])
                    if len(sent)>0:
                        data=[(sent, ['O']*len(sent))]
                        tag=model.demo_one(sess, data)
                        preds=zip(data[0][0], tag)
                        content=''
                        for pred in preds:
                            if pred[1]!=0:
                                content+=pred[0]+'\t'+pred[1]+'\n'
                            else:
                                content+=pred[0]+'\t'+'O'+'\n'
                        body, dis, des=infer(content)
                        body_str='，'.join(body)
                        dis_str='，'.join(dis)
                        des_str='，'.join(des)
                        line.append(fr.iloc[i,0])
                        line.append(fr.iloc[i,1])
                        line.append(body_str)
                        line.append(dis_str)
                        line.append(des_str)
                    else:
                        line.append(fr.iloc[i,0])
                        line.append(fr.iloc[i,1])
                        line.append('')
                        line.append('')
                        line.append('')
                    fw.loc[i]=line  
                fw.to_csv('data_path/predictions/'+file, encoding='utf8', index=False)
                print(file+' generation finished!!')
            
if __name__=='__main__':
    run_ner_model(mode='demo', demo_model='1525314729')