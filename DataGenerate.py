import tensorflow as tf
import numpy as np
import copy as cp
import serial


Mode=3 #0:TrainData, 1:Test, 2:ReadData, 3:CombineData, 4:Dummy

MotionIndex=3




def HyperSampling(data,Lable):
    temp1=[]
    temp2=[]
    while True:
        it= len(data)+2
        for i in range(0,it,2):
            data=np.insert(data,i+1,(np.array(data[i])+np.array(data[i+1]))/2,axis=0)
            if len(data)>=80:
                break
        if len(data)>=80:
            break

    for i in range(80):
        temp1.extend(data[i])
    temp2.append(temp1)
    temp2.append(Lable)
    
    return temp2


def GenerateData(Mode,MotionIndex):

    ser = serial.Serial(
    port='COM8',
    baudrate=115200,
)

    
    FileName=''
    Lable = tf.one_hot([0,1,2,3,4,5,6,7,8,9,10],depth=11).eval(session=tf.Session())
    temp=[]
    save=[]
    data=[]
    IMU=[]
    InitializedData=[]
    Iterator=0
    StateChecker=0

    if Mode==0 or Mode==1:
        while True:
            if ser.readable():
                res = ser.readline()
            if Mode==0:
                FileName="Train"+str(MotionIndex)
                if ser.readable():
                    try:
                        IMU=list(map(float,res.decode()[1:len(res)-1].split(',')[1:]))
                    except:
                        ser.read_all()
                        continue
                    if StateChecker==0 and IMU[0]==1:
                        ser.read_all()
                        InitializedData=cp.copy(IMU[1:])
                        StateChecker=1
                    elif StateChecker==1 and IMU[0]==1:
                        data.extend(cp.copy(np.array(IMU[1:])-np.array(InitializedData)))
                    elif StateChecker==1 and IMU[0]==0:
                        if len(data)<=60:
                            data.clear()
                            StateChecker=0
                            continue
                        save.append(HyperSampling(np.array(data).reshape(-1,6),Lable[MotionIndex-1]))
                        data.clear()
                        print(Iterator)
                        if Iterator>=100:
                            np.save("./Data/"+FileName,save[1:],True)
                            #np.save("./Data/test1",save[1:],True)
                            break
                        Iterator+=1
                        StateChecker=0

            elif Mode==1:
                IMU=list(map(float,res.decode()[1:len(res)-1].split(',')[1:]))
                if StateChecker==0 and IMU[0]==1:
                    InitializedData=cp.copy(IMU[1:])
                    StateChecker=1
                elif StateChecker==1 and IMU[0]==1:
                    data.extend(cp.copy(np.array(IMU[1:])-np.array(InitializedData)))
                elif StateChecker==1 and IMU[0]==0:
                    li=[]
                    li.append(HyperSampling(np.array(data).reshape(-1,6),Lable[MotionIndex-1])[0])
                    ser.close()
                    return li

    elif Mode==2:
        print("Read Data Mode")
        LoadData=np.load('./Data/600.npy',allow_pickle=True)
        for i in range(len(LoadData)):
            print(np.array(LoadData[i][1]))
        print(len(LoadData[0][0]))
        print(len(LoadData[0][1]))


    elif Mode==3:
        print("Combine Mode")
        savetemp=[]
        Motion1=np.load('./Data/300.npy',allow_pickle=True)
        Motion2=np.load('./Data/300_.npy',allow_pickle=True)
        #Motion3=np.load('./Data/Train3.npy',allow_pickle=True)
        #Motion3=np.load('CombinedMotionDataming45.npy',allow_pickle=True)
        savetemp.extend(Motion1) 
        savetemp.extend(Motion2) 
        #savetemp.extend(Motion3) 
        np.random.shuffle(savetemp)
        np.save("./Data/"+str(len(savetemp)),savetemp,True)
        print(len(savetemp),"Saved")

    elif Mode==4:
        Motion1=np.load('./Data/Dummy.npy',allow_pickle=True)
        for i in range(len(Motion1)):
            Motion1[i][1]=Lable[10]
        np.save("./Data/Dymmy_c",Motion1,True)
            

#GenerateData(Mode,MotionIndex)
