#!/usr/bin/env python
##Details of the implementation are in PDF
from collections import Counter
from decimal import Decimal
from itertools import islice
import math
import operator
import sys
import random
import os
import numpy as np
np.warnings.filterwarnings('ignore')


#get inputs
t_or_t=sys.argv[1]
filename=sys.argv[2]
modelfile=sys.argv[3]
model=sys.argv[4]

#Nnet required variables and dictionaries
Orientation=[0, 90, 180, 270]
nnet_data_dic = {}
nnet_test_dic = {}

#getting training data
if t_or_t=='train' :
    f=open(filename,"rb")
else :
    f=open('train-data.txt',"rb")
lines=f.read().splitlines()
#key=File name , value=(<orientation>,[list of pixel values])
data_dic={}
index=0
for line in lines :
    s_line=line.split(" ")
    data_dic[index]=(s_line[1],s_line[2:])
    nnet_data_dic[index] = (s_line[1],s_line[2:])
    index+=1
f.close()

#getting test data
if t_or_t=='test' :
    f1=open(filename,"rb")
    lines1=f1.read().splitlines()
    #key=File name , value=(<orientation>,[list of pixel values])
    test_dic={}
    for line1 in lines1 :
        s_line1=line1.split(" ")
        test_dic[s_line1[0]]=(s_line1[1],s_line1[2:])
        nnet_test_dic[s_line1[0]] = (s_line1[1],s_line1[2:])
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
#print len(test_dic)

#opening output text file
f3=open('output.txt',"w")

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

#K Nearest Neighbours**************************************************************************

def nearest() :
    ccount=0
    for key1 , value1 in test_dic.iteritems() :
        tp_value=value1[1]
        responses=[]
        #print value1[0]
        for key , value in data_dic.iteritems() :
            orientation=value[0]
            pixel_value=value[1]
            count=0
            index=0
            pt=0
            p4=0
            total=0
            for p in range(len(pixel_value)) :
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
            #print True
            ccount+=1
        else :
            pass
            #print False
        f3.write(key1+" "+result.most_common(1)[0][0]+"\n")
    print "Accurancy for k nearest neighbour :"
    print ((float)(ccount)/(float)(len(test_dic)))*100
    return ccount

#print nearest()

#ADABOOST********************************************************************************

#Function called for training
def la1t(data_dic,pix) :
    count=0
    h=[]
    o=[]
    compare=[]
    result={0:[],1:[],2:[],3:[]}
    totals=[]
    for key , value in data_dic.iteritems() :
        orientation=value[0]
        pixel_value=value[1]
        total=0
        for p in pix :
            total+=(int)(pixel_value[p])
        totals.append(total)
        if total < 1530 :
            result[0].append(orientation)
        elif total >=1530 and total < 3060 :
            result[1].append(orientation)
        elif total >=3060 and total < 4590 :
            result[2].append(orientation)
        elif total >=4590 :
            result[3].append(orientation)
        o.append(orientation)
    for i  in range (4) :
        r=Counter(result[i])
        result[i]=r.most_common(1)[0][0]
    for j in range (len(totals)) :
        if totals[j] < 1530 :
            h.append(result[0])
            if o[j]==result[0] :
                compare.append(True)
                count+=1
            else :
                compare.append(False)
        elif totals[j] >=1530 and totals[j] < 3060 :
            h.append(result[1])
            if o[j]==result[1] :
                compare.append(True)
                count+=1
            else :
                compare.append(False)
        elif totals[j] >=3060 and totals[j] < 4590 :
            h.append(result[2])
            if o[j]==result[2] :
                compare.append(True)
                count+=1
            else :
                compare.append(False)
        elif totals[j] >=4590 :
            h.append(result[3])
            if o[j]==result[3] :
                compare.append(True)
                count+=1
            else :
                compare.append(False)
    return (h,compare,o,result)



#Function called while runnning on test data. Arguments : Dictionary of data(either test/training data),-
#pix : List of pixels which are used for classification. Result : Classes that a particular-
#classifier identifies.
def mla1t(data_dic,pix,result) :
    count=0
    h=[]
    o=[]
    image_name=[]
    compare=[]
    totals=[]
    for key , value in data_dic.iteritems() :
        orientation=value[0]
        pixel_value=value[1]
        total=0
        for p in pix :
            total+=(int)(pixel_value[p])
        totals.append(total)
        if total < 1530 :
            h.append(result[0])
        elif total >=1530 and total < 3060 :
            h.append(result[1])
        elif total >=3060 and total < 4590 :
            h.append(result[2])
        elif total >=4590 :
            h.append(result[3])
        image_name.append(key)
        o.append(orientation)
    return (h,compare,o,image_name)

#Function to normalize values
def normalize(w) :
    ma=max(w)
    mi=min(w)
    for i in range (len(w)) :
        w[i]=((float)(w[i])-(float)(mi))/((float)(ma)-(float)(mi))
    return w

#Main function for training
def ada(data_dic,f4):
    w=[(float)(1)/(float)(36976) for x in range(36976)]
    K=[[7,15,23,31,39,47,55,63],[0,8,16,24,32,40,48,56],[0,1,2,3,4,5,6,7],[56,57,58,59,60,61,62,63],[48,49,50,51,52,53,54,55],[32,43,12,54,24,63,34,15],[8,9,10,11,12,13,14,15],[0,9,18,27,28,21,14,7],[56,49,42,35,36,45,54,63],[0,1,2,3,8,9,10,11],[4,5,6,7,12,13,14,15],[48,49,50,51,56,57,58,59],[52,53,54,55,60,61,62,63],[48,49,56,57,54,55,62,63],[16,17,24,25,32,33,40,41],[42,43,58,59,44,45,60,61],[0,9,2,11,4,13,6,15],[48,57,50,59,52,61,54,63],[0,1,8,9,16,17,24,25],[32,33,40,41,48,49,56,57],[38,39,46,47,54,55,62,63],[3,4,11,12,19,20,27,28]]
    for k in K :
        ret=la1t(data_dic,k)
        com=ret[1]
        ori=ret[2]
        f4.write(" ".join(ret[3].values())+"\n")
        error=0
        nu=0
        for j in range(len(com)) :
            if com[j]!=True :
                nu+=(float)(w[j])
        error=((float)(nu)/sum(w))
        alpha=math.log((1-error)/error)+math.log(4-1)
        for j in range(len(com)) :
            if com[j]!=True :
                w[j]=(w[j])*math.exp(alpha)
        w=normalize(w)
        f4.write("Weight "+(str)(alpha)+"\n")
        #print alpha

#ada(data_dic)


#Called at runtime for classifying test data
def adarun(test_dic,f4) :
    lines=f4.read().splitlines()
    weights=[]
    dic_list=[]
    for l in lines :
        temp=l.split(" ")
        if temp[0]=="Weight" :
            weights.append((float)(temp[1]))
        else :
            temp_dic={}
            index=0
            for t in temp :
                temp_dic[index]=t
                index+=1
            dic_list.append(temp_dic)
    h1=mla1t(test_dic,[7,15,23,31,39,47,55,63],dic_list[0])[0]
    h2=mla1t(test_dic,[0,8,16,24,32,40,48,56],dic_list[1])[0]
    h3=mla1t(test_dic,[0,1,2,3,4,5,6,7],dic_list[2])[0]
    ret=mla1t(test_dic,[56,57,58,59,60,61,62,63],dic_list[3])
    h4=ret[0]
    o=ret[2]
    i=ret[3]
    h5=mla1t(test_dic,[48,49,50,51,52,53,54,55],dic_list[4])[0]
    h6=mla1t(test_dic,[32,43,12,54,24,63,34,15],dic_list[5])[0]
    h7=mla1t(test_dic,[8,7,9,10,11,12,13,14,15],dic_list[6])[0]
    h8=mla1t(test_dic,[0,9,18,27,28,21,14,7],dic_list[7])[0]
    h9=mla1t(test_dic,[56,49,42,35,36,45,54,63],dic_list[8])[0]
    h10=mla1t(test_dic,[0,1,2,3,8,9,10,11],dic_list[9])[0]
    h11=mla1t(test_dic,[4,5,6,7,12,13,14,15],dic_list[10])[0]
    h12=mla1t(test_dic,[48,49,50,51,56,57,58,59],dic_list[11])[0]
    h13=mla1t(test_dic,[52,53,54,55,60,61,62,63],dic_list[12])[0]
    h14=mla1t(test_dic,[48,49,56,57,54,55,62,63],dic_list[13])[0]
    h15=mla1t(test_dic,[16,17,24,25,32,33,40,41],dic_list[14])[0]
    h16=mla1t(test_dic,[42,43,58,59,44,45,60,61],dic_list[15])[0]
    h17=mla1t(test_dic,[0,9,2,11,4,13,6,15],dic_list[16])[0]
    h18=mla1t(test_dic,[48,57,50,59,52,61,54,63],dic_list[17])[0]
    h19=mla1t(test_dic,[0,1,8,9,16,17,24,25],dic_list[18])[0]
    h20=mla1t(test_dic,[32,33,40,41,48,49,56,57],dic_list[19])[0]
    count=0
    K=['0','90','180','270']
    for ind in range (len(h1)) :
        maxi=-1000000000
        clas=0
        ss=[]
        for k in K :
            su=0
            if h1[ind]==k :
                su+=weights[0]
            if h2[ind]==k :
                su+=weights[1]
            if h3[ind]==k :
                su+=weights[2]
            if h4[ind]==k :
                su+=weights[3]
            if h5[ind]==k :
                su+=weights[4]
            if h6[ind]==k :
                su+=weights[5]
            if h7[ind]==k :
                su+=weights[6]
            if h8[ind]==k :
                su+=weights[7]
            if h9[ind]==k :
                su+=weights[8]
            if h10[ind]==k :
                su+=weights[9]
            if h11[ind]==k :
                su+=weights[10]
            if h12[ind]==k :
                su+=weights[11]
            if h13[ind]==k :
                su+=weights[12]
            if h14[ind]==k :
                su+=weights[13]
            if h15[ind]==k :
                su+=weights[14]
            if h16[ind]==k :
                su+=weights[15]
            if h17[ind]==k :
                su+=weights[16]
            if h18[ind]==k :
                su+=weights[17]
            if h19[ind]==k :
                su+=weights[18]
            if h20[ind]==k :
                su+=weights[19]
            ss.append(su)
            if su>maxi :
                maxi=su
                clas=k
        f3.write(i[ind]+" "+clas+"\n")
        if (int)(o[ind])==(int)(clas) :
            count+=1
    print "Accuracy for Adaboost :"
    print ((float)(count)/len(o))*100

#adarun(test_dic)

#Neural Net starts here
def forward_sigmoid(x): return 1 /(1 + (np.exp(-x)))

def dsigmoid(y): return y * (1.0 - y)

def nnet_train():
    print "Training funtion starts here.."
    # Initially generating random weights to begin with the weights.
    initial_wi_ip_to_hidden = np.random.uniform(-1, 1, [192, 20])
    wi_hidden_to_op = np.random.uniform(-1, 1, [20, 4])

    for i in range(3):
        items = nnet_data_dic.items()
        random.shuffle(items)
        for j in range(500):
            y = np.zeros(4)
            y[Orientation.index(int(items[j][1][0]))] = 1
            image_pixels_np_array = np.array(items[j][1][1]).astype(float)

            #Feed Forward
            hidden_layer_output_np_array = forward_sigmoid(np.dot(image_pixels_np_array, initial_wi_ip_to_hidden))
            output_layer_np_array = forward_sigmoid(np.dot(hidden_layer_output_np_array, wi_hidden_to_op))

            #BackPropogation
                #Delta_j calcuation for output layer
            delta_j = (y - output_layer_np_array) * dsigmoid(output_layer_np_array)
                #Delta_i calculation for hidden layer
            delta_i = dsigmoid(hidden_layer_output_np_array) * np.dot(delta_j, wi_hidden_to_op.T)
                #Wi adjustment for hidden to output
            wi_hidden_to_op += np.multiply(hidden_layer_output_np_array[np.newaxis].T, delta_j) * 0.01
                #Wi adjustment for input to hidden
            initial_wi_ip_to_hidden += np.multiply(image_pixels_np_array[np.newaxis].T, delta_i) * 0.01

    # Writing the model file
    with open(modelfile, 'w') as f1:
        np.savetxt(f1, initial_wi_ip_to_hidden, fmt='%10.8f')
        np.savetxt(f1, wi_hidden_to_op, fmt='%10.8f')

    print "Training is ended now!"

def nnet_test():
    print "Starting the Test function..."
    accuracy_count = 0.0
    accuracy_percentage = 0.0
    output=[]

    # Fetching the data from model file
    with open(modelfile, 'r') as lines:
        test_wi_ip_to_hidden = np.genfromtxt(islice(lines, 0, 192))
        lines.seek(0)
        test_wi_hidden_to_op = np.genfromtxt(islice(lines, 192, 212))

    # Performing the Test function
    for key in nnet_test_dic:
        test_image_pixels = nnet_test_dic[key]
        test_image_pixels_np_array = np.array(test_image_pixels[1]).astype(float)

        # Forward propogation
        test_hidden_layer_output = forward_sigmoid(np.dot(test_image_pixels_np_array, test_wi_ip_to_hidden))
        test_output_layer = forward_sigmoid(np.dot(test_hidden_layer_output, test_wi_hidden_to_op))

        predicted_orientation = Orientation[np.argmax(test_output_layer)]
        if predicted_orientation == int(test_image_pixels[0]):
            accuracy_count += 1
        output_line= key+" "+str(predicted_orientation)
        #print output_line
        output= np.append(output,output_line)

        
    accuracy_percentage = (accuracy_count / len(nnet_test_dic)) * 100
    print "Accuracy is: {0}% " .format(accuracy_percentage)
    with open('output.txt', 'w') as op:
        # op.write(key+" "+"%d"%(predicted_orientation) + '\n')
        np.savetxt(op, output, fmt="%s")
#Neural Net ends here

def main_function() :
    if t_or_t == 'train' :
        if model=='nearest' :
            nearest()
        if model=='adaboost' :
            f4=open(modelfile,"w")
            ada(data_dic,f4)
            f4.close()
        if model=='nnet' :
            nnet_train()
        if model=='best' :
            pass
    elif t_or_t=='test'  :
        if model=='nearest' :
            nearest()
        if model=='adaboost' :
            f4=open(modelfile,"r")
            adarun(test_dic,f4)
            f4.close()
        if model=='nnet' :
            nnet_test()
        if model=='best' :
            pass

main_function()
f3.close()

