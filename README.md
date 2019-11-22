# Class wise triplet loss (center-triplet loss) for face recognition

This is the training/validating script for the Inception-RsNet based deep CNNs implemented in Tensorflow (1.00) and python 2.7 for the  face verfication. 

This training algorithm of the deep CNNs for face verfication is  based on the classwise-triplet loss loss algorithm described in the paper:
["Simple Triplet Loss Based on Intra/Inter-class Metric Learning for Face Verification"](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w23/Ming_Simple_Triplet_Loss_ICCV_2017_paper.pdf) published on the ICCV workshop 2017.

The accuracy measured on the LFW test set is 98.5% when trained on CASIA-WebFace or 99.4% when trained on MsCeleb1M

### Class-wise triplet loss
Class-wise triplet loss reduce greatly the computation comparing to the classic triplet loss by using the centers instead of the elements of the class to calculate the triplet loss as shown in ![Fig.1](https://github.com/zuhengming/facenet_class_wise_triplet_loss/blob/master/figs/intra_inter_loss_cropped.png) ![Fig.2](https://github.com/zuhengming/facenet_class_wise_triplet_loss/blob/master/figs/computation_reduce.png).  
Fig. 2 shows the the comparison of the softmax, center loss and proposed class-wise triplet loss on the dataset MNIST. The rows from top to bottom are corresponding to the softmax, center loss and proposed class-wise triplet loss respectively.
<div align=center>
   <img src="https://github.com/zuhengming/facenet_class_wise_triplet_loss/blob/master/figs/comparison.png">
</div>


### 1. Training data / Validating data

### Training data
- [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset have been used for training. This training set consists of total of 494,414 images over 10,575 identities.
- [MsCeleb1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/) dataset have been used for training. This training set consists of total of ~10,000,000 images over 100,000 identities (~ 100 images/person). Note that, not all images of MsCeleb1M was used for training the current model.

### Training data organization (this is important when user using his own dataset for training/fine-tuning the model):
Training data is/should be saved in the two-level folders structure:
- Training_data_folder 
   - Person1
      - image1
      - image2
      - image3
      - image4
   - Person2
      - image1
      - image2      
   - Person3
      - image1
      - image2
      - image3
   - Person4
      .....  

### Validating data: 
- [LFW](http://vis-www.cs.umass.edu/lfw/) dataset have been used for training. This training set consists of total of 12,000 images over 6000 positive/negative pairs of images. LFW was used for the "real-time" validating during the training of the model. 
- [Youtube Face](https://www.cs.tau.ac.il/~wolf/ytfaces/) dataset have been used for training. This training set consists of total of 3,425 videos over 1,595 identities, and the average length of a video clip is 181.3 frames.

### 2. Pre-processing

### Face detection/alignment using MTCNN
Before the training of the face verification system, the face region is detected and aligned by the MTCNN.
[MTCNN: K. Zhang, Z. Zhang, Z. Li, and Y. Qiao. Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10):1499¨C1503,2016. 6]


### 3. Running training for face verification from scrath:

### Enviroment requirements:  
Linux Ubuntu 16.04 /64bits, Tensorflow 1.00, Python 2.7 and opencv 3.0. Requiring CUDA toolkit 8.0 and CuDNN v5.
- 1) Tensorflow 1.00 (GPU Enabled):https://www.tensorflow.org/versions/r1.0/install/install_windows
- 2) Python 2.7 with packages 'scipy','Pillow', 'matplotlib', 'sklearn', 'h5py', 'pandas'
- 3) OpenCV 3.0

### Training and evaluation scripts:
facenet_train_classtripletloss.py is used for training and evaluate_face_verification.py is used for evaluation.

### Parameters instructions: 
- --logs_base_dir : Directory where to write event logs.
- --models_base_dir: Directory where to write trained models and checkpoints.
- --data_dir: Directory to the data directory containing aligned face patches. Multiple directories are separated with colon when using multiple datasets for training.
- --image_size : The size of the input image patch of the  networks;
- --model_def :Directory containing the definition of the inference graph of the networks;
- --lfw_dir : The file containing the pairs of images in LFW using for validation.
- --optimizer : The optimization algorithm to use for updating the parameters of networks;
- --learning_rate: Initial learning rate. If set to -1,  a learning rate schedule file "learning_rate_schedule.txt" is used for setting learning rate;
- --max_nrof_epochs : Number of epochs to run;
- --keep_probability : Keep probability of dropout for the fully connected layer(s).
- --random_crop : Performs random cropping of training images. If false, the center image_size pixels from the training images are used.
- --random_flip : Performs random horizontal flipping of training images.
- --learning_rate_schedule_file : File containing the learning rate schedule which is used when learning_rate is set to to -1.
- --weight_decay : L2 weight regularization rate.
- --center_loss_factor : Center loss weight in the total loss function.
- --center_loss_alfa: Center update decay rate for center loss.
- --gpu_memory_fraction : Upper bound on the amount of GPU memory that will be used by the process.
- --epoch_size : Number of batches per epoch.
- --trainset_start / --trainset_end : The number of the identities between --trainset_start and --trainset_end as the training set. This is used for very large scale dataset for training model.  
- --batch_size : Numer of images per batch.  

