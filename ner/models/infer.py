# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:24:19 2018

@author: Administrator
"""

from ner_model.utils import get_entity

fr=open('data_path/predictions/乳腺内门.txt', 'r', encoding='utf8')
contents=fr.read().split(' \tO\n')

for lines in contents:
    tag_seq=[]
    char_seq=[]
    items=lines.split('\n')
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
            a_pos=ad[1]
            if len(des)>0:
                for i,de in enumerate(des):
                    de_pos=de[1]
                    if de_pos>a_pos:
                        de_txt=de[0]
                        break
            else:
                de_pos=9999999999
            if len(dis)>0:
                for j,di in enumerate(dis):
                    di_pos=di[1]
                    if di_pos>a_pos:
                        di_txt=di[0]
                        break
            else:
                di_pos=9999999999
            if di_pos<de_pos:
                out+=ad[0]+di_txt
                dis.remove(di)
                dis.append((out,di_pos))
            elif de_pos<di_pos:
                out+=ad[0]+de_txt
                des.remove(de)
                des.append((out,de_pos))
            else:
                continue
    
    dis2=[]
    des2=[]
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
        
    print('BODY: {}\nDIS: {}\nDES: {}'.format(body, dis2, des2))