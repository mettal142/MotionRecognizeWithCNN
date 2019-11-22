import tensorflow as tf
import numpy as np
import DataGenerate
import serial
import copy as cp
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/",one_hot= True)

#cnn 모델 정의
def build_CNN_classifier(x):
    x_image = tf.reshape(x,[-1,40,6,1])

    W_conv1= tf.Variable(tf.truncated_normal(shape = [3,3,1,64],stddev=5e-2))
    b_conv1= tf.Variable(tf.constant(0.1,shape=[64]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W_conv1,strides= [1,1,1,1],padding = 'SAME')+b_conv1)

    h_pool1= tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    print(h_pool1)
    W_conv2 = tf.Variable(tf.truncated_normal(shape=[3,3,64,64],stddev = 5e-2))
    b_conv2= tf.Variable(tf.constant(0.1,shape=[64]))
    h_conv2= tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)

    h_pool2=tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    print(h_pool2)
    W_conv3 = tf.Variable(tf.truncated_normal(shape=[3,3,64,128],stddev = 5e-2))
    b_conv3= tf.Variable(tf.constant(0.1,shape=[128]))
    h_conv3= tf.nn.relu(tf.nn.conv2d(h_pool2,W_conv3,strides=[1,1,1,1],padding='SAME')+b_conv3)

    W_conv4 = tf.Variable(tf.truncated_normal(shape=[3,3,128,128],stddev = 5e-2))
    b_conv4= tf.Variable(tf.constant(0.1,shape=[128]))
    h_conv4= tf.nn.relu(tf.nn.conv2d(h_conv3,W_conv4,strides=[1,1,1,1],padding='SAME')+b_conv4)
    
    W_conv5 = tf.Variable(tf.truncated_normal(shape=[3,3,128,128],stddev = 5e-2))
    b_conv5= tf.Variable(tf.constant(0.1,shape=[128]))
    h_conv5= tf.nn.relu(tf.nn.conv2d(h_conv4,W_conv5,strides=[1,1,1,1],padding='SAME')+b_conv5)

    W_fc1= tf.Variable(tf.truncated_normal(shape=[10*2*128,1024],stddev=5e-2))
    b_fc1= tf.Variable(tf.constant(0.1,shape=[1024]))

    h_conv5_flat = tf.reshape(h_conv5,[-1,10*2*128])
    h_fc1= tf.nn.relu(tf.matmul(h_conv5_flat,W_fc1)+b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)


    W_output= tf.Variable(tf.truncated_normal(shape=[1024,11],stddev=5e-2))
    b_output= tf.Variable(tf.constant(0.1,shape=[11]))
    logits = tf.matmul(h_fc1_drop,W_output)+b_output

    y_pred= tf.nn.softmax(logits)
    return y_pred,logits



x=tf.placeholder(tf.float32,shape=[None,240])
y=tf.placeholder(tf.float32,shape=[None,11])
keep_prob=tf.placeholder(tf.float32)

TrainData=np.load('./Data/300.npy',allow_pickle=True)
#np.random.shuffle(TrainData)

bx=[]
by=[]
tx=[]
ty=[]
ttx=[]
StateChecker=0
data=[]

for i in range(len(TrainData)):
    if i<=int(len(TrainData)*0.90):
        bx.append(TrainData[i][0][:-240])
        by.append(TrainData[i][1])
    else:
        tx.append(TrainData[i][0][:-240])
        ty.append(TrainData[i][1])

y_pred,logits=build_CNN_classifier(x)
loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits))
train_step=tf.train.AdamOptimizer(1e-5).minimize(loss)
currect_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(currect_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        train_accuracy = accuracy.eval(feed_dict = {x:bx,y:by,keep_prob:1.0})
        print("반복 : %d, 정확도 : %f"%(i,train_accuracy))
        sess.run([train_step],feed_dict={x:bx,y:by,keep_prob:0.8})
        if train_accuracy==1:
            break
    print("테스트 데이터 정확도 : %f"%accuracy.eval(feed_dict={x:tx, y:ty,keep_prob:1.0}))
    #for i in range(len(ttx)):
    ser = serial.Serial(
port='COM4',
baudrate=115200,
)
    while True:

        if ser.readable():
            res = ser.readline()
        IMU=list(map(float,res.decode()[1:len(res)-1].split(',')[1:]))
        if StateChecker==0 and IMU[0]==1:
            ser.read_all()
            InitializedData=cp.copy(IMU[1:])
            StateChecker=1
        elif StateChecker==1 and IMU[0]==1:
            data.extend(cp.copy(np.array(IMU[1:])-np.array(InitializedData)))
        elif StateChecker==1 and IMU[0]==0:
            ttx=[]
            ttx.append(DataGenerate.HyperSampling(np.array(data).reshape(-1,6),[])[0][:-240])
            print("모션 정확도",max(y_pred.eval(feed_dict={x:ttx,keep_prob:1.0})[0]),"모션 : ",np.array(np.where(y_pred.eval(feed_dict={x:ttx,keep_prob:1.0})[0]==max(y_pred.eval(feed_dict={x:ttx,keep_prob:1.0})[0])))[0][0]+1)
            StateChecker=0
            res=[]
            data=[]
