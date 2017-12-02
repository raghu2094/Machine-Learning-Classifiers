#!/usr/bin/env python
##Details of the implementation are in PDF
from collections import Counter
import math
import operator
f=open("train-data.txt","rb")
lines=f.read().splitlines()
#key=File name , value=(<orientation>,[list of pixel values])
data_dic={}
index=0
for line in lines :
    s_line=line.split(" ")
    data_dic[index]=(s_line[1],s_line[2:])
    index+=1
f.close()
#print data_dic['train/10017728034.jpg']

print len(data_dic)

f1=open("test-data.txt","rb")
lines1=f1.read().splitlines()
#key=File name , value=(<orientation>,[list of pixel values])
test_dic={}
for line1 in lines1 :
    s_line1=line1.split(" ")
    test_dic[s_line1[0]]=(s_line1[1],s_line1[2:])
f1.close()

#add RGB in test data
for key,value in test_dic.iteritems() :
    count=0
    total=0
    temp=[]
    for v in value[1] :
        if count<=1 :
            total+=(int)(v)
            count+=1
        else :
            total+=(int)(v)
            temp.append(total)
            count=0
            total=0
    test_dic[key]=(value[0],temp)


#add RGB in train data
for key,value in data_dic.iteritems() :
    count=0
    total=0
    temp=[]
    for v in value[1] :
        if count<=1 :
            total+=(int)(v)
            count+=1
        else :
            total+=(int)(v)
            temp.append(total)
            count=0
            total=0
    data_dic[key]=(value[0],temp)


knn_features={}
for i in range(16) :
    knn_features[i]=0


def nearest(data_dic) :
    print "Start"
    ccount=0
    for key1 , value1 in test_dic.iteritems() :
        tp_value=value1[1]
        responses=[]
        for key , value in data_dic.iteritems() :
            orientation=value[0]
            pixel_value=value[1]
            count=0
            index=0
            pt=0
            p4=0
            total=0
            for p in range(len(pixel_value)) :
                # if count<=1 :
                #     count+=1
                #     pt+=(int)(tp_value[p])
                #     p4+=(int)(pixel_value[p])
                # else :
                #     pt+=(int)(tp_value[p])
                #     p4+=(int)(pixel_value[p])
                #     total=pow((pt-p4),2)
                #     count=0
                #     p4=0
                #     pt=0
                pt=(int)(tp_value[p])
                p4=(int)(pixel_value[p])
                total+=pow((pt-p4),2)
            distance=math.sqrt(total)
            responses.append((orientation,distance))
        responses.sort(key=operator.itemgetter(1))
        ori=[]
        for x in range(13) :
           ori.append(responses[x][0])
        result=Counter(ori)
        if result.most_common(1)[0][0]==value1[0] :
            print True
            ccount+=1
        else :
            print False
    return ccount

#print nearest(data_dic)

def learning_algo(data_dic,code,w) :
    if code==0 :
        return la1(data_dic)
    elif code==1 :
        return la2(data_dic)
    else :
        return la3(data_dic)

#Uppper Pixel Group
def la1(data_dic) :
    count=0
    h=[]
    compare=[]
    for key , value in data_dic.iteritems() :
        orientation=value[0]
        pixel_value=value[1]
        total=0
        result=0
        for p in range(4,7) :
            total+=(int)(pixel_value[p])
        if total < 1530 :
            result=0
        elif total >=1530 and total < 3060 :
            result=90
        elif total >=3060 and total < 4590 :
            result=270
        elif total >=4590 :
            result=180
        if (int)(result)==(int)(orientation) :
            count+=1
            compare.append(True)
        else :
            compare.append(False)
        h.append(result)
    return (h,compare)

#Lower Pixel Group
def la2(data_dic) :
    count=0
    h=[]
    compare=[]
    for key , value in data_dic.iteritems() :
        orientation=value[0]
        pixel_value=value[1]
        total=0
        result=0
        for p in range(58,62) :
            total+=(int)(pixel_value[p])
        if total < 1530 :
            result=180
        elif total >=1530 and total < 3060 :
            result=90
        elif total >=3060 and total < 4590 :
            result=270
        elif total >=4590 :
            result=0
        if (int)(result)==(int)(orientation) :
            count+=1
            compare.append(True)
        else :
            compare.append(False)
        h.append(result)
    return (h,compare)

#random 
def la3(data_dic) :
    count=0
    h=[]
    compare=[]
    for key , value in data_dic.iteritems() :
        orientation=value[0]
        pixel_value=value[1]
        total=0
        result=0
        for p in range(33,38) :
            total+=(int)(pixel_value[p])
        if total < 1530 :
            result=180
        elif total >=1530 and total < 3060 :
            result=90
        elif total >=3060 and total < 4590 :
            result=270
        elif total >=4590 :
            result=0
        if (int)(result)==(int)(orientation) :
            count+=1
            compare.append(True)
        else :
            compare.append(False)
        h.append(result)
    return (h,compare)


def normalize(w) :
    ma=max(w)
    mi=min(w)
    for i in range (len(w)) :
        w[i]=((float)(w[i])-(float)(mi))/((float)(ma)-(float)(mi))
    return w

def ada(data_dic):
    w=[(float)(1)/(float)(36976) for x in range(36976)]
    para=[0,0]
    for k in range (3) :
        ret=learning_algo(data_dic,k,w)
        h=ret[0]
        com=ret[1]
        error=0
        nu=0
        for j in range(len(h)) :
            if com[j]==True :
                nu+=(float)(w[j])
        error=((float)(nu)/sum(w))
        alpha=math.log((1-error)/error)+math.log(4-1)
        for j in range(len(h)) :
            if com[j]==True :
                w[j]=(w[j])*math.exp(alpha)
        w=normalize(w)
        print alpha
    

ada(data_dic)        




def rla1(data_dic) :
    count=0
    h=[]
    compare=[]
    for key , value in data_dic.iteritems() :
        orientation=value[0]
        pixel_value=value[1]
        total=0
        result=0
        for p in range(4,7) :
            total+=(int)(pixel_value[p])
        if total < 1530 :
            result=0
        elif total >=1530 and total < 3060 :
            result=90
        elif total >=3060 and total < 4590 :
            result=270
        elif total >=4590 :
            result=180
        if (int)(result)==(int)(orientation) :
            count+=1
            compare.append(True)
        else :
            compare.append(False)
        h.append(result)
    return (h,compare)

def rla2(data_dic) :
    count=0
    h=[]
    o=[]
    compare=[]
    for key , value in data_dic.iteritems() :
        orientation=value[0]
        pixel_value=value[1]
        total=0
        result=0
        for p in range(58,62) :
            total+=(int)(pixel_value[p])
        if total < 1530 :
            result=180
        elif total >=1530 and total < 3060 :
            result=90
        elif total >=3060 and total < 4590 :
            result=270
        elif total >=4590 :
            result=0
        if (int)(result)==(int)(orientation) :
            count+=1
            compare.append(True)
        else :
            compare.append(False)
        h.append(result)
        o.append(orientation)
    return (h,compare,o)

#random 
def rla3(data_dic) :
    count=0
    h=[]
    compare=[]
    for key , value in data_dic.iteritems() :
        orientation=value[0]
        pixel_value=value[1]
        total=0
        result=0
        for p in range(33,38) :
            total+=(int)(pixel_value[p])
        if total < 1530 :
            result=180
        elif total >=1530 and total < 3060 :
            result=90
        elif total >=3060 and total < 4590 :
            result=270
        elif total >=4590 :
            result=0
        if (int)(result)==(int)(orientation) :
            count+=1
            compare.append(True)
        else :
            compare.append(False)
        h.append(result)
    return (h,compare)



def adarun(test_dic) :
    h1=rla1(test_dic)[0]
    h3=rla3(test_dic)[0]
    ret=rla2(test_dic)
    h2=ret[0]
    o=ret[2]
    count=0
    K=[0,90,180,270]
    for ind in range (len(h1)) :
        maxi=-1000000000
        clas=0
        for k in K :
            su=0
            if h1[ind]==k :
                su+=2.42353118273
            if h2[ind]==k :
                su+=3.36422072407
            if h3[ind]==k :
                su+=0.434562394272
            print su
            if su>maxi :
                maxi=su
                clas=k
        print clas
        if (int)(o[ind])==clas :
            count+=1
    print count

adarun(test_dic)
