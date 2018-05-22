import logging, sys, argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity(tag_seq, char_seq):
    DIS = get_DIS_entity(tag_seq, char_seq)
    DES = get_DES_entity(tag_seq, char_seq)
    BODY = get_BODY_entity(tag_seq, char_seq)
    ADJ = get_ADJ_entity(tag_seq, char_seq)
    ADV = get_ADV_entity(tag_seq, char_seq)
    return DIS, DES, BODY, ADJ, ADV


def get_DIS_entity(tag_seq, char_seq):
    length = len(char_seq)
    DIS = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-DIS':
            if 'dis' in locals().keys():
                DIS.append((dis,i-1))
                del dis
            dis = char
            if i+1 == length:
                DIS.append((dis,i))
        if tag == 'I-DIS':
            if 'dis' in locals().keys():
                dis += char
            else:
                dis=char
            if i+1 == length:
                DIS.append((dis,i))
        if tag not in ['I-DIS', 'B-DIS']:
            if 'dis' in locals().keys():
                DIS.append((dis,i-1))
                del dis
            continue
    return DIS


def get_DES_entity(tag_seq, char_seq):
    length = len(char_seq)
    DES = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-DES':
            if 'des' in locals().keys():
                DES.append((des,i-1))
                del des
            des = char
            if i+1 == length:
                DES.append((des,i))
        if tag == 'I-DES':
            if 'des' in locals().keys():
                des += char
            else:
                des=char
            if i+1 == length:
                DES.append((des,i))
        if tag not in ['I-DES', 'B-DES']:
            if 'des' in locals().keys():
                DES.append((des,i-1))
                del des
            continue
    return DES


def get_BODY_entity(tag_seq, char_seq):
    length = len(char_seq)
    BODY = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-BODY':
            if 'body' in locals().keys():
                BODY.append(body)
                del body
            body = char
            if i+1 == length:
                BODY.append(body)
        if tag == 'I-BODY':
            if 'body' in locals().keys():
                body += char
            else:
                body=char
            if i+1 == length:
                BODY.append(body)
        if tag not in ['I-BODY', 'B-BODY']:
            if 'body' in locals().keys():
                BODY.append(body)
                del body
            continue
    return BODY
	
	
def get_ADJ_entity(tag_seq, char_seq):
    length = len(char_seq)
    ADJ = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-ADJ':
            if 'adj' in locals().keys():
                ADJ.append((adj,i-1))
                del adj
            adj = char
            if i+1 == length:
                ADJ.append((adj,i))
        if tag == 'I-ADJ':
            if 'adj' in locals().keys():
                adj += char
            else:
                adj=char
            if i+1 == length:
                ADJ.append((adj,i))
        if tag not in ['I-ADJ', 'B-ADJ']:
            if 'adj' in locals().keys():
                ADJ.append((adj,i-1))
                del adj
            continue
    return ADJ
	
	
def get_ADV_entity(tag_seq, char_seq):
    length = len(char_seq)
    ADV = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-ADV':
            if 'adv' in locals().keys():
                ADV.append(adv)
                del adv
            adv = char
            if i+1 == length:
                ADV.append(adv)
        if tag == 'I-ADV':
            if 'adv' in locals().keys():
                adv += char
            else:
                adv=char
            if i+1 == length:
                ADV.append(adv)
        if tag not in ['I-ADV', 'B-ADV']:
            if 'adv' in locals().keys():
                ADV.append(adv)
                del adv
            continue
    return ADV


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


def infer(content):
    tag_seq, char_seq=[], []
    items=content.split('\n')
    for item in items:
        if len(item)>0:
            parts=item.split('\t')
            char_seq.append(parts[0])
            tag_seq.append(parts[1])
    dis, des, body, adj, adv=get_entity(tag_seq, char_seq)
    # 拼接形容词与名词
    if len(adj)>0:
        for ad in adj:
            out=''
            # 获取形容词的位置
            a_pos=ad[1]
            if len(des)>0:
                # 在描述列表中寻找出现在形容词后一个位置的描述字段；有则令de_txt为该描述字段，否则令de_txt为空
                for i,de in enumerate(des):
                    de_pos=de[1]
                    if de_pos>a_pos:
                        de_txt=de[0]
                        break
                    else:
                        de_txt=''
            else:
                de_pos=9999999999
            if len(dis)>0:
                # 在疾病列表中寻找出现在形容词后一个位置的疾病字段；有则令di_txt为该疾病字段，否则令di_txt为空
                for j,di in enumerate(dis):
                    di_pos=di[1]
                    if di_pos>a_pos:
                        di_txt=di[0]
                        break
                    else:
                        di_txt=''
            else:
                di_pos=9999999999
            # 若疾病字段在描述字段前出现，则拼接形容词字段和疾病字段
            if di_pos<de_pos:
                out+=ad[0]+di_txt
                dis.remove(di)
                dis.append((out,di_pos))
            # 若描述字段在疾病字段前出现，则拼接形容词字段和描述字段
            elif de_pos<di_pos:
                out+=ad[0]+de_txt
                des.remove(de)
                des.append((out,de_pos))
            else:
                continue
    
    dis2, des2=[], []
    if len(dis)>0:
        for di2 in dis:
            dis2.append(di2[0])
    else:
        dis2=dis
    if len(des)>0:
        for de2 in des:
            des2.append(de2[0])
    else:
        des2=des
        
    return body, dis2, des2