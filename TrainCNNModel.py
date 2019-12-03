import tensorflow as tf
import numpy as np
import DataGenerate
import copy as cp
import socket

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/",one_hot= True)


size = 1024
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 5050))
s.listen(5)

#cnn 모델 정의

def build_CNN_classifier(x):
    x_image = tf.reshape(x,[-1,40,6,1])

    W_conv1= tf.Variable(tf.truncated_normal(shape = [3,3,1,64],stddev=5e-2))
    b_conv1= tf.Variable(tf.constant(0.1,shape=[64]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W_conv1,strides= [1,1,1,1],padding = 'SAME')+b_conv1)

    h_pool1= tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W_conv2 = tf.Variable(tf.truncated_normal(shape=[3,3,64,64],stddev = 5e-2))
    b_conv2= tf.Variable(tf.constant(0.1,shape=[64]))
    h_conv2= tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)

    h_pool2=tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

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
output_node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
TrainData=np.load('./Data/600.npy',allow_pickle=True)
#np.random.shuffle(TrainData)

bx=[]
by=[]
tx=[]
ty=[]
ttx=[]
StateChecker=0
data=[]

for i in range(len(TrainData)):
    if i<=int(len(TrainData)*0.9):
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
    for i in range(500):
        train_accuracy = accuracy.eval(feed_dict = {x:bx,y:by,keep_prob:1.0})
        print("반복 : %d, 정확도 : %f"%(i,train_accuracy))
        sess.run([train_step],feed_dict={x:bx,y:by,keep_prob:0.8})
        #if train_accuracy==1:
        #    break
    print("테스트 데이터 정확도 : %f"%accuracy.eval(feed_dict={x:tx, y:ty,keep_prob:1.0}))
    #for i in range(len(ttx)):
  
    tf.compat.v1.train.Saver().save(sess,'./Model/my_test_model')
    tf.compat.v1.train.Saver().save(sess,'./Model/check.ckpt')
   # Restore the graph
    saver = tf.compat.v1.train.import_meta_graph('./Model/my_test_model.meta')

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('Model'))

    # Freeze the graph
    frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('Graph/output_graph.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())

 

