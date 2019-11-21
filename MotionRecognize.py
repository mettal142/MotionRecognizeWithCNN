import tensorflow as tf
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/",one_hot= True)

#cnn 모델 정의
def build_CNN_classifier(x):
    x_image = tf.reshape(x,[-1,160,6,1])

    W_conv1= tf.Variable(tf.truncated_normal(shape = [3,3,1,32],stddev=5e-2))
    b_conv1= tf.Variable(tf.constant(0.1,shape=[32]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W_conv1,strides= [1,1,1,1],padding = 'SAME')+b_conv1)

    h_pool1= tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W_conv2 = tf.Variable(tf.truncated_normal(shape=[3,3,32,64],stddev = 5e-2))
    b_conv2= tf.Variable(tf.constant(0.1,shape=[64]))
    h_conv2= tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)

    h_pool2=tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W_fc1= tf.Variable(tf.truncated_normal(shape=[40*2*64,1024],stddev=5e-2))
    b_fc1= tf.Variable(tf.constant(0.1,shape=[1024]))
    h_pool2_flat = tf.reshape(h_pool2,[-1,40*2*64])
    h_fc1= tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

    W_output= tf.Variable(tf.truncated_normal(shape=[1024,11],stddev=5e-2))
    b_output= tf.Variable(tf.constant(0.1,shape=[11]))
    logits = tf.matmul(h_fc1,W_output)+b_output

    y_pred= tf.nn.softmax(logits)
    return y_pred,logits

x=tf.placeholder(tf.float32,shape=[None,960])
y=tf.placeholder(tf.float32,shape=[None,11])

TrainData=np.load('CombinedMotionDatajh1600.npy',allow_pickle=True)
bx=[]
by=[]
tx=[]
ty=[]
for i in range(len(TrainData)):
    if i<=int(len(TrainData)*0.9):
        bx.append(TrainData[i][0])
        by.append(TrainData[i][1])
    else:
        tx.append(TrainData[i][0])
        ty.append(TrainData[i][1])
    

#TestData=np.load('CombinedTestData.npy',allow_pickle=True)
#tx=[]
#ty=[]
#for i in range(len(TestData)):
#    tx.append(TestData[i][0])
#    ty.append(TestData[i][1])   
#     

y_pred,logits=build_CNN_classifier(x)

loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits))
train_step=tf.train.AdamOptimizer(1e-5).minimize(loss)
currect_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(currect_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(200):
        train_accuracy = accuracy.eval(feed_dict = {x:bx,y:by})
        print("반복 : %d, 정확도 : %f"%(i,train_accuracy))
        sess.run([train_step],feed_dict={x:bx,y:by})
    print("테스트 데이터 정확도 : %f"%accuracy.eval(feed_dict={x:tx, y:ty}))