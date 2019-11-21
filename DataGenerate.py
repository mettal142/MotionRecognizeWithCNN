import tensorflow as tf
import numpy as np
import copy as cp
import serial


Mode=0 # 0:TrainData, 1:TestData, 2:ReadData, 3:CombineData

MotionIndex=2



def HyperSampling(data,Lable):
    temp1=[]
    temp2=[]
    while True:
        it= len(data)+2
        for i in range(0,it,2):
            data=np.insert(data,i+1,(np.array(data[i])+np.array(data[i+1]))/2,axis=0)
            if len(data)>=160:
                break
        if len(data)>=160:
            break

    for i in range(160):
        temp1.extend(data[i])
    temp2.append(temp1)
    temp2.append(Lable)
    
    return temp2


def GenerateData(Mode,MotionIndex):

    ser = serial.Serial(
    port='COM4',
    baudrate=115200,
)

    FileName=''
    Lable = tf.one_hot([0,1,2,3,4,5,6,7,8,9,10],depth=11).eval(session=tf.Session())
    temp=[]
    save=[]
    data=[]
    IMU=[]
    InitializedData=[]
    Iterator=1
    StateChecker=0

    if Mode==0:
        FileName="Train"+str(MotionIndex)
        while Iterator<=11:
            if ser.readable():
                res = ser.readline()
        ##        print(res.decode()[:len(res)-1])
                IMU=list(map(float,res.decode()[1:len(res)-1].split(',')[1:]))
                if StateChecker==0 and IMU[0]==1:
                    InitializedData=cp.copy(IMU[1:])
                    StateChecker=1
                elif StateChecker==1 and IMU[0]==1:
                    data.extend(cp.copy(np.array(IMU[1:])-np.array(InitializedData)))
                elif StateChecker==1 and IMU[0]==0:
                    save.append(HyperSampling(np.array(data).reshape(-1,6),Lable[MotionIndex-1]))
                    data.clear()
                    print(Iterator)
                    Iterator+=1
                    StateChecker=0

        np.save(FileName,save[1:],True)
    elif Mode==1:
        FileName="Test"+str(MotionIndex)
        while Iterator<=10:
            if ser.readable():
                res = ser.readline()
        ##        print(res.decode()[:len(res)-1])
                IMU=list(map(float,res.decode()[1:len(res)-1].split(',')[1:]))
                if StateChecker==0 and IMU[0]==1:
                    InitializedData=cp.copy(IMU[1:])
                    StateChecker=1
                elif StateChecker==1 and IMU[0]==1:
                    data.extend(cp.copy(np.array(IMU[1:])-np.array(InitializedData)))
                elif StateChecker==1 and IMU[0]==0:
                    save.append(HyperSampling(np.array(data).reshape(-1,6),Lable[MotionIndex-1]))
                    data.clear()
                    print(Iterator)
                    Iterator+=1
                    StateChecker=0

            np.save(FileName,save[1:],True)

    elif Mode==2:
        LoadData=np.load('CombinedMotionDatajh1600.npy',allow_pickle=True)
        print(len(np.array(LoadData)))

    elif Mode==3:
        savetemp=[]
        Motion1=np.load('CombinedMotionData1000.npy',allow_pickle=True)
        Motion2=np.load('CombinedMotionDatamj45.npy',allow_pickle=True)
        #Motion3=np.load('CombinedMotionDataming45.npy',allow_pickle=True)
        savetemp.extend(Motion1) 
        savetemp.extend(Motion2) 
        #savetemp.extend(Motion3)
        np.random.shuffle(savetemp)
        np.save("CombinedMotionDatajh1600",savetemp,True)
        print("Saved")


GenerateData(Mode,MotionIndex)
#te=[[2,2,2,2,2,2],[4,4,4,4,4,4],[6,6,6,6,6,6],[8,8,8,8,8,8],[10,10,10,10,10,10]]
#HyperSampling(te,Lable[MotionIndex]-1)
#for i in range(0,len(te)+2,2):
#    te=np.insert(te,i+1,(np.array(te[i])+np.array(te[i+1]))/2,axis=0)

#print(te)
