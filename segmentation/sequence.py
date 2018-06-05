# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 14:34:44 2018

@author: Administrator
"""

def dedup(ls):
    '''
    deduplicate the given SORTED list
    '''
    i, j=0, 0
    while j<len(ls):
        if ls[j]==ls[i]:
            j+=1
        else:
            i+=1
            ls[i]=ls[j]
    return ls[:i+1]

def genSubstr(string, n):
    """
    Generate all substrings of max length n for string
    """
    length=len(string)
    res=[]
    for i in range(length):
        for j in range(i+1, min(i+n+1, length+1)):
            res.append(string[i:j])
    return res

def genSubparts(string):
    """
    Partition a string into all possible two parts, e.g.
    given "abcd", generate [("a", "bcd"), ("ab", "cd"), ("abc", "d")]
    For string of length 1, return empty list
    """
    length=len(string)
    res=[]
    for i in range(1, length):
        res.append((string[:i], string[i:]))
    return res

def longestSubsequenceLength(s1, s2):
    '''
    统计s1中包含s2的最大句子长度
    Input：
        s1:list('今天你很酷')
        s2:list('你很酷呀')
    Output:
        3
    '''
    n=len(s2)+1
    cur=[0]*n
    next=[0]*n
    tmp=None
    for i in s1:
        for j in range(n):
            if j==0:
                next[j]=0
            else:
                next[j]=cur[j-1]+1 if i==s2[j-1] else max(next[j-1], cur[j])
        tmp=next
        next=cur
        cur=tmp
    return cur[n-1]

def longestSubsequence(s1, s2):
    '''
    输出s1中包含s2的最大句子
    Input：
        s1:list('今天你很酷')
        s2:list('你很酷呀')
    Output:
        '你很酷'
    '''
    n=len(s2)+1
    cur=[0]*n
    next=[0]*n
    tmp=None
    __NONE, __UP, __LEFT, __NEW=0, 1, 2, 3
    orientation=[[__NONE]*n]
    for i in s1:
        ori=[]
        for j in range(n):
            if j==0:
                next[j]=0
                ori.append(__NONE)
            else:
                next[j]=cur[j-1]+1 if i==s2[j-1] else max(next[j-1], cur[j])
                ori.append(__NEW if i==s2[j-1] else (__LEFT if next[j-1]>cur[j] else __UP))
        orientation.append(ori)
        tmp=next
        next=cur
        cur=tmp
    i, j, res=len(s1), n-1, ''
    ori=orientation[i][j]
    while ori!=__NONE:
        if ori==__UP:
            i-=1
        elif ori==__LEFT:
            j-=1
        elif ori==__NEW:
            i-=1
            j-=1
            res+=s2[j]
        ori=orientation[i][j]
    return res[::-1]
                
