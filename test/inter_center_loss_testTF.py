# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import unittest
import tensorflow as tf
import numpy as np
import sys
sys.path.append('../src')
import facenet
import matplotlib.pyplot as plt
class CenterLossTest(unittest.TestCase):
  


    def testCenterLoss(self):
        batch_size = 5#16
        nrof_features = 2
        nrof_classes = 3#16
        alfa = 0.5
        
        with tf.Graph().as_default():
        


            # Define center loss
            #loss, update_centers, one_hot, delta1, delta2, centers_delta = facenet.center_loss(features, labels, alfa, nrof_classes)
            #loss = facenet.center_loss(features, labels, alfa, nrof_classes)
            #lbls = np.random.randint(low=0, high=nrof_classes, size=(batch_size,))
            #lbls = np.random.randint(low=0, high=nrof_classes, size=(batch_size,))
            lbls = [0,0,0,1,2,2,2,1,1]
            #lbls = [0, 1, 2, 1, 1]
            feats = np.array([[-3,-3],[-3-0.5,-3],[-3,-3+0.5],  [-3,-1],  [-3,1], [-3+0.5,1], [-3,1-0.5],  [-3,-1+0.5],  [-3+0.5,-1]])
            #feats = create_features(label_to_center, batch_size, nrof_features, lbls)
            nrof_classes_batch = len(set(lbls))
            #loss = facenet.center_inter_loss_tf(features, batch_size, labels, alfa, nrof_classes, nrof_classes_batch)
            beta=0.5
            #loss = facenet.class_level_triplet_loss_tf(features, batch_size, labels, alfa, nrof_classes, beta)
            #loss = facenet.center_loss(features, labels, alfa, nrof_classes)
            #loss = facenet.center_loss_similarity(features, labels, alfa, nrof_classes)
            features = tf.placeholder(tf.float32, shape=(len(lbls), nrof_features), name='features')
            labels = tf.placeholder(tf.int64, shape=(len(lbls),), name='labels')
            loss = facenet.class_level_triplet_loss_similarity_tf(features, len(lbls), labels, nrof_classes, beta)


            label_to_center = np.array( [ 
                 [-3,-3],  [-3,-1],  [-3,1],  [-3,3],
                 [-1,-3],  [-1,-1],  [-1,1],  [-1,3],
                 [ 1,-3],  [ 1,-1],  [ 1,1],  [ 1,3],
                 [ 3,-3],  [ 3,-1],  [ 3,1],  [ 3,3] 
                 ])
                
            sess = tf.Session()
            with sess.as_default():
                sess.run(tf.global_variables_initializer())
                np.random.seed(seed=666)
                
                for i in range(0,100):
                    # Create array of random labels
                    #lbls = np.random.randint(low=0, high=nrof_classes, size=(batch_size,))
                    #feats = create_features(label_to_center, batch_size, nrof_features, lbls)
                    #nrof_classes_batch = len(set(lbls))

                    #loss = facenet.center_inter_loss_tf(features, batch_size, labels, alfa, nrof_classes, nrof_classes_batch)
                    

                    #sess.run(tf.global_variables_initializer())
                    
                    #center_loss_, centers_, diff_, centers_batch_ = sess.run([center_loss, centers, diff, centers_batch], feed_dict={features:feats, labels:lbls})
                    #loss_, update_centers_, one_hot_, delta1_, delta2_, centers_delta_ = sess.run([loss, update_centers, one_hot, delta1, delta2, centers_delta], feed_dict={features: feats, labels: lbls})
                    #loss_, centers_, label_, centers_batch_, diff_, centers_cts_, centers_cts_batch_, diff_mean_, center_cts_clear_ = sess.run(loss, feed_dict={features:feats, labels:lbls})
                    #loss_, centers_, diff_, centers_cts_batch_ = sess.run(loss, feed_dict={features:feats, labels:lbls})
                    #loss_, centers_, label_, centers_batch, diff_, centers_cts_, centers_cts_batch_, diff_mean_,center_cts_clear_, centers_cts_batch_reshape = sess.run(loss, feed_dict={features:feats, labels:lbls})

                    #loss_, centers_, centers_1D  = sess.run(loss, feed_dict={features:feats, labels:lbls})
		            #loss_, centers_, centers_1D, centers_2D, centers_3D, features_3D, dist_inter_centers, dist_inter_centers_sum_dim, dist_inter_centers_sum_all, dist_inter_centers_sum, loss_inter_centers, loss_center, centers_batch = sess.run(loss, feed_dict={features:feats, labels:lbls})
		            #loss_, centers_, class_sum_, dist_within_center_, dist_inter_centers_sum_all_,dist_inter_centers_sum_dim_, dist_inter_centers_, features_3D,centers_3D_,centers_1D_, dist_centers_sum_,centers_list_,features_, nrof_centers_batch_, centers_batch_,label_unique_ = sess.run(loss, feed_dict={features:feats, labels:lbls})
                    #loss_, centers_, loss_x_, similarity_all_, similarity_self_ = sess.run(loss, feed_dict={features:feats, labels:lbls})
                    #loss, loss_real_mtr, loss_real_mtr, pre_loss_mtr, similarity_self_mn_beta, similarity_self_mn, similarity_self, similarity_all, centers,centers_norm_,features_,center_num_, self_index_, a_ = sess.run(loss, feed_dict={features:feats, labels:lbls})
                    loss_reg, loss_mtr, loss_sum, loss_sum_real, loss_mean, pre_loss_mtr, similarity_self_mn, similarity_self, similarity_all, centers, features, nrof_centers_batch = sess.run(
                        loss, feed_dict={features: feats, labels: lbls})
                    #similarity_all, self_index, similarity_self = sess.run(loss, feed_dict={features:feats, labels:lbls})
                    ##mzh
                    figure = plt.figure()
                    x = feats[:, 0]
                    y = feats[:, 1]
                    z = lbls
                    plt.scatter(x, y, c=z, s=100, cmap=plt.cm.cool, edgecolors='None', alpha=0.75)
                    x = centers_[:, 0]
                    y = centers_[:, 1]
                    z = lbls
                    plt.scatter(x, y, c=z, s=100, marker='x', cmap=plt.cm.cool, edgecolors='None', alpha=0.75)
               
                    plt.colorbar()
                    plt.text(0, -3, 'loss: %f, iter: %d'%(loss_,i))
                    plt.show()
                    #raw_input("Press Enter to continue...")
                    plt.close()
                    
                    lbls = np.random.randint(low=0, high=nrof_classes, size=(batch_size,))
                    #feats = create_features(label_to_center, batch_size, nrof_features, lbls)
                    nrof_classes_batch1 = len(set(lbls))
 
                # After a large number of updates the estimated centers should be close to the true ones
                np.testing.assert_almost_equal(centers_, label_to_center, decimal=5, err_msg='Incorrect estimated centers')
                np.testing.assert_almost_equal(loss, 0.0, decimal=5, err_msg='Incorrect center loss')
                


def create_features(label_to_center, batch_size, nrof_features, labels):
    # Map label to center
#     label_to_center_dict = { 
#          0:(-3,-3),  1:(-3,-1),  2:(-3,1),  3:(-3,3),
#          4:(-1,-3),  5:(-1,-1),  6:(-1,1),  7:(-1,3),
#          8:( 1,-3),  9:( 1,-1), 10:( 1,1), 11:( 1,3),
#         12:( 3,-3), 13:( 3,-1), 14:( 3,1), 15:( 3,3),
#         }
    # Create array of features corresponding to the labels
    feats = np.zeros((batch_size, nrof_features))
    for i in range(batch_size):
        cntr =  label_to_center[labels[i]]
        for j in range(nrof_features):
            feats[i,j] = cntr[j]
    return feats
                      
if __name__ == "__main__":
    unittest.main()
