“ CNN Based Lung Sound Classification for Health Monitoring”

ABSTRACT: 

Respiratory conditions can impair lung function, disrupting the breathing process and the exchange of oxygen and carbon dioxide in the body. The typical approach researchers use to develop classification systems for breathing sounds involves two main steps: feature extraction and pattern classification. Recently, there has been a growing interest in the field of breathing sound categorization, especially in leveraging deep neural networks, which have shown effectiveness in handling large datasets. Advanced technologies play a crucial role in advancing medical care. Specifically, extensive research conducted in collaboration with researchers, healthcare professionals, and patients is essential for establishing precise and personalized treatment options for a wide range of diseases. This research devised an attentional deep learning model with inverse transform sampling to categorize respiratory disorders using audio data. Accurate and efficient models were developed to automatically detect respiratory diseases from lung sound recordings, utilizing the Respiratory Sound Database containing information on various respiratory illnesses. The main objective of this study is to create robust models capable of precisely classifying lung sounds and recognizing respiratory illnesses. CNN, VGG16, and ResNet50 architectures were utilized as potent deep learning frameworks for extracting features and categorizing data. Incorporating pre-trained models like VGG16 and ResNet50 significantly improves the model's capability to identify critical features within complex spectral images. This research introduces a novel element by employing inverse transfer sampling, a technique designed to equalize class distribution by transferring insights from minority to majority classes. This strategy is tailored to address the prevalent issue of class imbalance in medical datasets, where certain respiratory conditions might be less represented. The findings from this study indicate that the proposed approach is highly effective, achieving an impressive accuracy of 98% with the CNN model, 83% with the VGG16 model, and 95% with the ResNet50 model. Additionally, a CRNN and LSTM model were incorporated into the study, providing further insights into the classification of respiratory disorders.

INTRODUCTION:

As per the World Health Organization (WHO), respiratory illnesses including lung cancer, tuberculosis (TB), asthma, lower respiratory tract infections (LRTI), and chronic obstructive pulmonary disease (COPD) contribute significantly to global mortality rates. These diseases are among the most prevalent and perilous worldwide, often characterized by severe and occasionally contagious symptoms. Prompt diagnosis and timely treatment are crucial in mitigating the rising incidence of respiratory-related illnesses and fatalities.

Lung disease (LD) refers to any ailment that disrupts the proper functioning of the respiratory system. Breath sounds are produced by the lung structure during breathing. Physicians employ auscultation with a stethoscope to evaluate and diagnose lung disorders. The stethoscope is the primary tool for listening to lung sounds, enabling physicians to identify alterations such as reduced or absent breath sounds, irregularities, or regular patterns. Through auscultation, contemporary physicians can promptly and cost-effectively deduce various disease conditions affecting the pulmonary and cardiovascular systems, resulting in enhanced diagnosis and treatment. Auscultation is considered a cost-effective, user-friendly, and reliable diagnostic method with rapid turnaround time. It is most effective in a calm, well-lit, and comfortable environment. This examination provides abundant information regarding lung diseases.

Respiratory system diseases encompass a wide range of conditions, from acute infections to chronic disorders impacting lung function. Asthma is characterized by inflammation and narrowing of the airways, resulting in wheezing and breathing difficulties. Pneumonia and bronchitis fall within the category of lower respiratory tract infections (LTRIs), often caused by viral or bacterial agents that damage the airways and lungs. Chronic obstructive pulmonary disease (COPD) is marked by airflow limitation, typically resulting from prolonged exposure to irritating gases or particulate matter, most commonly through smoking. Sore throats are among the symptoms of common colds and other upper respiratory tract infections (URTIs), affecting the nose, throat, and sinuses. Pneumonia, an inflammation of lung tissue, is commonly caused by bacteria or viruses and varies in severity from mild to severe. Bronchiectasis refers to the chronic dilation of airways due to inflammation or injury, leading to mucus buildup and recurrent infections. Bronchiolitis, which narrows and inflames small airways in the lungs, primarily affects infants and young children. Knowledge of these conditions is essential for maintaining respiratory health and differentiating between respiratory disorders.

 A DL architecture merges CNN and LSTM for classifying normal and adventitious lung sounds. CNN extracts deep features and reduces dimensionality, while LSTM identifies temporal dependencies. The model applies focal loss (FL) to reduce prediction errors and address data imbalance. Training and evaluation use lung sounds from the Respiratory Sound Database.

Stethoscope-based diagnosis has certain limitations, including the subjective nature of interpretations leading to variability among different listeners and the necessity for trained medical professionals to assess auscultation sounds. These limitations are further exacerbated by the shortage of trained medical personnel in developing countries and during pandemics. In addition to enabling remote patient monitoring by non-medical staff, such as community health workers, automated analysis of respiratory sounds can help alleviate these constraints, particularly with the emergence of digital stethoscopes.

Therefore, machine learning plays a crucial role in classifying various types of sounds in numerous ways. Deep learning (DL), a subset of machine learning, facilitates the identification of respiratory conditions through auscultation. These learning techniques are currently at the forefront of rapidly expanding fields such as machine translation, image and signal recognition, and related disciplines. The continuous progress and refinement of image processing algorithms within the medical domain have elevated deep learning to a pivotal area of research. The spectrogram is a valuable tool employed in the analysis of lung sounds in the study of respiratory disorders. Compared to most emerging economies globally, healthcare costs are rising at twice the rate in the United States. Spectrophotograms could potentially be utilized for classifying lung disorders, thereby reducing the expenses associated with CT scan treatments.

To enhance the classification of data pertaining to respiratory issue diagnosis, this study employed deep learning techniques. The proposed method leverages the widely used Convolutional Neural Network (CNN), a deep learning architecture along with the pretrained models vgg16 and ResNet50. Specifically, we introduced several advanced preprocessing methods, including inverse transform sampling, to facilitate effective lung sound classification. Conventional categorization outcomes were found to be inconsistent due to ambient noise interference in the audio recordings. Existing CNN techniques vary in structure as they solely depend on audio feature algorithms. An open-source public dataset was utilized as the experimental dataset source.
n this work, we compared the lung sound categorization capabilities of CNN, VGG16, and ResNet50 models. CNN outperformed the other models in terms of accuracy and robustness, which makes it a good option for automated respiratory ailment classification based on lung sound analysis, even if the findings from all models were encouraging.

Related work

Respiratory conditions, including lung cancer, asthma, tuberculosis, lower respiratory tract infections (LRTI), upper respiratory tract infections (URTI), and chronic obstructive pulmonary disease (COPD), rank as significant contributors to mortality and morbidity across the globe. The prevalence of these illnesses is increasing annually, leading to their expansion in recent times. The World Health Organization (WHO) has identified the "big-five" respiratory diseases responsible for the majority of deaths worldwide. Proper diagnosis, treatment, and monitoring can alleviate the suffering of affected individuals and, in many cases, treat or prevent these diseases [1] Thoracic auscultation stands as the primary technique for diagnosing lung conditions. Medical professionals leverage their knowledge and skills to identify normal respiratory states and potential abnormalities based on various lung sounds heard through a stethoscope. However, relying solely on auscultation is not entirely reliable for clinical decision-making, as the human ear cannot adequately detect low-frequency sounds. To overcome this limitation and improve the rates of early detection, there is an initiative to transform lung sounds into digital information. This digital data will then feed into a proposed deep learning model, aiming to create an effective system for recognizing lung sounds.[2] Within medical disciplines, deep learning is presently employed for analyzing diverse signals and images. Published studies have investigated the utilization of artificial intelligence (AI) to aid in the auscultation of lung and heart sounds[3]

Furthermore, the development of new technologies and their increasing deployment in healthcare are generating vast amounts of data, which are challenging to manage and interpret manually. Additionally, variability in the interpretation of lung sounds during auscultation can result in incorrect diagnoses and treatment plans. To address these challenges, automating the lung auscultation process has become crucial. Research efforts have focused on automating this process, with the primary goal of these computational models being to categorize respiratory audio data into normal (healthy) and abnormal (diseased) groups. These models employ techniques such as logistic regression, support vector machines (SVM), and K-nearest neighbor classifiers (KNN).[4] Numerous investigations have been conducted to automate lung auscultation, aiming to diagnose abnormal sounds detected in respiratory audio recordings and classify patients into healthy and unhealthy categories. The predominant focus of research has been on anomaly-driven prediction, involving the identification of crackles, wheezes, or both, employing neural networks (NN), convolutional neural networks (CNN), recurrent neural networks (RNN), and other artificial intelligence (AI) methodologies.[5]

The greatest challenge in this research lies in distinguishing between lung and heart sounds due to the spectral and temporal overlap between these signals. A method known as frequency decomposition, or modulation frequency (obtained in the modulating frequency domain), allows for the differentiation of heart and lung sounds. This distinction is achieved through the use of specifically crafted Bandpass and Bandstop modulation filters. The analysis of the signal is finalized by segmenting it into consecutive overlapping frames and applying Fourier transformation.[6] Moreover, for years, researchers have utilized various features from audio data, including spectrograms, Mel spectrograms, and Mel-frequency cepstral coefficients (MFCCs), to distinguish between patients with normal and abnormal respiratory conditions. These features convey essential auditory information and are closely linked to visual cues. To categorize respiratory audio data, convolutional neural networks (CNNs), a type of deep learning technique that excels in image classification, have been employed.[7] The research detailed in [8] introduced a hybrid deep learning model, combining convolutional neural networks (CNN) and recurrent neural networks (RNN), for the classification of respiratory sounds using Mel spectrogram features.

For the study , a straightforward CNN model utilizing Mel-Spectrogram was created, achieving a score of 68.5% due to adept padding, augmentation through concatenation, fine-tuning tailored to the device, and pruning of blank regions[9] In the research work [10], the issue of class imbalance was tackled by employing a pre-trained ResNet model with co-tuning, vanilla fine-tuning, stochastic normalization, and data augmentation. The resulting model achieved a rating of 49.27%. Moreover, the authors of the research paper introduced a hybrid neural model aimed at mitigating the class imbalance issue through the implementation of the focal loss (FL) function. By extracting features from the STFT spectrogram and feeding them into an LSTM network, a classification score of 68.52% was attained[11].


METHODOLOGY:

	DATASET:

The Respiratory Sound Database resulted from a collaboration between research teams from Portugal and Greece, comprising 920 annotated recordings spanning durations from 10 to 90 seconds. These recordings were obtained from 126 patients, totaling 5.5 hours of data, and encompassing 6898 respiratory cycles. Among these cycles, 1864 contain crackles, 886 contain wheezes, and 506 exhibit both crackles and wheezes. The dataset incorporates clean respiratory sounds alongside noisy recordings simulating real-world scenarios. Patients of various age groups, including children, adults, and the elderly, are represented within the dataset. 

The collection consists of 920 wav files featuring recordings of respiratory cycles from patients. These recordings capture a range of respiratory sounds, encompassing both normal breath sounds and abnormal ones like crackles and wheezes. Accompanying each audio file are 920 annotation.txt files, essential for understanding the specific timing and characteristics of the respiratory events captured in the recordings. These annotations likely detail the timing, length, and type of respiratory sounds present, such as crackles and wheezes.

Each patient's diagnosis is documented in a text file, which is pivotal for correlating the recorded respiratory sounds with particular respiratory ailments or conditions. This file bridges the gap between the audio recordings and clinical diagnoses, enabling researchers to explore the link between sound patterns and health issues.

Additionally, the dataset contains a text file that provides demographic details for each patient, including aspects like age, gender, and other pertinent attributes. Such demographic information is vital for analyzing the variation in respiratory conditions across different age groups and populations.

When combined, these files can provide respiratory sound data users and researchers with an extensive amount of information. The sound files provide the raw audio data, the annotations files provide temporal information about respiratory events, the diagnosis file links the data to clinical conditions and the demographic data provides patient context.

	PREPROCESSING

Handling missing values:

In the patient database, a specific strategy is adopted to address the absence of data in the 'Adult BMI (kg/m2)' column by leveraging existing information on a child's weight and height. This method calculates the missing adult BMI values using the conventional BMI formula, which involves dividing the weight in kilograms by the square of the height in meters. Whenever there is a missing value for an adult's BMI but the data for a child's weight (in kilograms) and height (in centimeters) is present, these metrics are utilized to compute the BMI. The calculation involves converting the child's height into meters to ensure the units are consistent for the BMI formula.
 Subsequently, the calculated BMI value is precisely rounded to two decimal places to enhance accuracy before it is incorporated back into the dataset, effectively substituting the missing 'Adult BMI (kg/m2)' entries. This technique significantly augments the dataset's integrity, facilitating a more precise evaluation of patient information by including estimated BMI figures in cases where direct measurements are missing.

Label encoding and one Hot encoding:

When preparing categorical data for machine learning models, the labels within the 'Diagnosis' column undergo two significant transformations: label encoding and one-hot encoding. This process involves assigning a unique numerical label to each diagnosis category in the 'Diagnosis' column, with consecutive integers representing each category. After label encoding, categorical data undergoes one-hot encoding using the 'to categorical' function from the 'keras. utils' module. This step converts the integer-encoded labels into a binary matrix representation, known as one-hot encoding. Each diagnosis category is then represented by a binary vector, where only one element is set to one to indicate the presence of the category, while all other elements are set to zero.


	AUGMENTATION
 
Inverse transform sampling, a technique in probability and statistics, generates random samples from a specified probability distribution. It finds extensive application in various deep learning tasks, including generating synthetic data and sampling from probability distributions. This method is particularly prevalent in generative models such as Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs). The fundamental concept of inverse transform sampling involves drawing random samples from a uniform distribution and then applying the inverse cumulative distribution function (CDF) of the desired distribution to transform them.

	FEATURE EXTRACTION
 
MFCC (Mel Frequency Cepstrum Coefficients) is a sound processing technique that mimics the sensitivity of the human auditory system. It prioritizes low-frequency components (below 1 KHz) over high frequencies (above 1 KHz). The Mel scale is used to represent frequency as   
            mel(f) = 2595.log (1+0)                                                                            ……(1)                                                                                  
The process of calculating MFCC involves several steps. Firstly, framing is performed         to handle non-stationary signals like lung sounds, which are considered stationary within short time intervals. Frames are created by segmenting the lung signal using sliding windows. Determining the appropriate frame size and shift interval requires further research. In typical audio transmissions, each frame consists of 256 samples, with an overlap of 128 samples. Adjusting the frame size and overlap between frames can enhance performance. For instance, a window size of 100ms and an overlap of 20ms are commonly used, while a window size of 0.5s with 0.1s overlap is also effective. Additionally, the Hamming window is applied to each frame obtained during framing. The Hamming window is defined as follows:

H(n)=0.54-0.46~cos ({2\π×n} {N-1})                                                      ……(2)
To enhance the speed of filtering, the Fast Fourier Transform (FFT) is employed to convert lung sounds from the time domain to the frequency domain. Creating a triangular window bank involves distributing a uniform triangular filter bank across the Mel-Warped spectrum. The Mel-spaced filter bank is generated based on the following relationships:

∆f_{mel}={mel(f_{max})}/{L}                                                             ……(3)

            c(l) = 700[   10^l∆ mel/2595-1]              l=1,2...L                                    ……(4)
            
In above equations, in Hertz .c(l) is the central frequency of the triangle filter. The output energy of each filter is calculated by:

            m(l) = ∑_{k=o(l)} ^{k(l)} W_{l}(k) |X(k)|,       l = 1,2….L                  …….(5)


In this study, we utilize a custom triangle filter instead of the Mel filter bank, specifying parameters such as the filter order, low frequency value, and high frequency value for the third triangle filter. To compute the MFCC, the Discrete Cosine Transformation (DCT) is then applied to the logarithmic energy of the output.

            C_{mfcc}(i)=\∑_{l=1} ^{L}log~m(l)cos\ {(l-\ {1}{2}) \{i\π} {L}\}   ……(6)

MODELS

CNN

![image](https://github.com/sandeepsai15634/CNN-BASED-LUNG-SOUND-CLASSIFICATION/assets/119305751/0fbfd52b-f450-459e-a323-451d1ed1027e)

Fig 1. Data flow diagram of CNN for lung sound classification

Convolution is an operation between two functions or signals. It is denoted with an asterisk and its one-dimensional version is written as:

X s(t) = (x∗w) (t) = a=−∞ x(a)w (t – a)                                               ……(7)

In this study of a convolutional neural network (CNN), the first signal, denoted as x, serves as the input, while the second signal, represented by w, corresponds to the filter. In this one-dimensional scenario, t represents the time index, and a signifies the time shift value. The resulting output of the convolutional layer is referred to as the feature map. In the case of images, it is more common to employ a two-dimensional convolution, which takes a three-dimensional matrix (comprising width, height, and color channels) as input and produces a corresponding three-dimensional matrix as output. The parameters of the convolutional layer consist of a series of trainable filters, each of which is convolved across the width and height of the input, thereby generating a two-dimensional feature map. The network will acquire filters that become activated upon detecting specific features. 

![image](https://github.com/sandeepsai15634/CNN-BASED-LUNG-SOUND-CLASSIFICATION/assets/119305751/49bda1d2-ec85-4fa7-a82b-902f5ea47607)

Fig.2.Architecure diagram of CNN for lung sound classification

During the data loading and preparation phase, patient demographics and diagnoses are imported from CSV files. Missing values in the “Adult BMI” column of the demographic data are imputed based on the child’s height and weight. Subsequently, data frames are concatenated to form a unified dataset. Audio files are loaded using the Librosa library, and filenames are leveraged to extract information about audio recordings during audio feature extraction. Mel-frequency cepstral coefficients (MFCCs) are extracted as features and associated with the corresponding diagnosis labels. Using Keras, a sequential neural network model is built with multiple dense layers, dropout for regularization, and ReLU activation functions.

The model is constructed using the Adam optimizer and categorical cross-entropy loss. Training involves using the dataset for 30 epochs with a batch size of 24, during which training progress is monitored, with loss and accuracy indicators printed. Post-training, the model is assessed on the test set, and performance metrics such as accuracy, precision, recall, and F1-score are computed using sklearn. Metrics. Additionally, a random sample prediction from the test set is selected to demonstrate the model’s prediction, including the relevant diagnosis from the dataset and both original and predicted class labels. The model achieves an accuracy rate of approximately 98%.


![image](https://github.com/sandeepsai15634/CNN-BASED-LUNG-SOUND-CLASSIFICATION/assets/119305751/5773ed24-ae68-4550-b3f5-6c720739997e)

Fig 3. Architecture diagram of CNN for respiratory lung sound classification

The mathematical rationale behind employing convolutional neural networks (CNNs) for audio files is well-founded. When dealing directly with audio data, the input typically consists of a one-dimensional signal representing the waveform’s amplitude over time. This one-dimensional signal is denoted as x(t), with (t) representing time.

Input Representation:

In general, the CNN handles each window as an individual input by partitioning the audio signal x(t) into smaller segments, or windows. Here, “s” represents the stride (step size) and “w” denotes the input window size. This approach results in multiple windows available for processing, some of which may overlap.

x_1 〖,x〗_(2,) x_(3,) x_(4……..)                                                                                           ……(8)

Convolution Operation:

One of the primary tasks of CNNs is the convolution process, where filters, also known as kernels, are applied in audio processing. Considering that k is typically smaller than w, let’s designate F as the filter with a size of k. The convolution operation is outlined as follows:

(x*F)= ∑_(a=0)^(k-1)  x(t-a).F(a)                                                              ……(9)

Pooling Operation:

Pooling is a frequently employed method following convolution to reduce the spatial dimensions of the data and highlight prominent features. Max pooling is a popular option in this regard. When employing P, the pooling operation can be defined as 
follows:
             P(y)(t)= 〖max〗_(a ) y(t-a)                                                                    ……(10)
             
Fully Connected Layers:

Following several convolutional and pooling layers, it’s common to flatten the output into a vector and input it into fully connected layers for tasks like classification or regression. If \(h\) represents the flattened output and \(W\) is the weight matrix of a fully connected layer, the operation can be depicted as follows:
              z=W.h                                                                                                ……(11)  
              
Activation Function:

An activation function (e.g., ReLU, sigmoid) is applied elementwise to the output of the fully connected layer to introduce non-linearity.


Output Layer:

The final layer, often a SoftMax layer, provides the output of the network for classification problems.

vgg16

The initial steps in the data loading and preprocessing phase involve setting up directories for the training and testing datasets and establishing a connection with Google Drive to access the stored data. Subsequently, ImageDataGenerator is employed to handle the loading and preprocessing of images from these directories. The dataset comprises lung sound images categorized into eight distinct classes, with patient diagnosis information retrieved from a CSV file linking patient IDs to their respective diagnoses. Upon initializing the VGG16 model with pre-trained weights from the ImageNet dataset, its architecture, comprising various layers and parameters, is presented. During the modification stage, the fully connected layer of VGG16, also referred to as “fc2,” is removed, and a new dense layer featuring eight output nodes, each corresponding to one of the eight classes, is added. Following this adjustment, the model undergoes compilation utilizing the Adam optimizer with…

Batch normalization is implemented on the convolutional layers to enhance training stability, while the model undergoes training using the training dataset with continuous monitoring of mean absolute error (MAE) and other performance metrics. Visualization of the training history, encompassing loss, accuracy, and MAE over epochs, provides an insightful analysis of the model’s learning performance for both the training and validation sets. Insights gleaned from the training history plots facilitate a deeper understanding of the model’s learning dynamics and aid in identifying potential issues such as overfitting or underfitting. The model consistently achieves an accuracy of approximately 83% across validation sets. Insights derived from the training history are instrumental in comprehending the model’s learning process and recognizing potential challenges such as overfitting or underfitting.

Originally designed for image classification tasks, the VGG16 architecture is a deep convolutional neural network. While spectrogram images weren’t initially intended for audio processing, they can be utilized as input. Spectrograms visually represent the frequency spectrum of a signal, demonstrating its temporal variations.


![image](https://github.com/sandeepsai15634/CNN-BASED-LUNG-SOUND-CLASSIFICATION/assets/119305751/5635cb08-3a2c-4b31-a89d-9da889db5dde)

Fig 4. Spectrogram of Healthy Lung

![image](https://github.com/sandeepsai15634/CNN-BASED-LUNG-SOUND-CLASSIFICATION/assets/119305751/fe186b09-a305-4392-874e-d8e75cb3bb35)

Fig. 5 Spectrogram of Defective Lung

Spectrogram Computation:

To transform audio files into spectrogram images, one can employ a Short-Time Fourier Transform (STFT). This process visually represents the time-varying frequency content of the audio signal through a spectrogram. The resulting image depicts time along the x-axis, frequency along the y-axis, and color intensity corresponding to the amplitude or power of the respective frequency.

![image](https://github.com/sandeepsai15634/CNN-BASED-LUNG-SOUND-CLASSIFICATION/assets/119305751/8577bfcb-7874-4b04-ba0c-ad621dd5ec82)

Fig.6. Architecture diagram of vgg16 for respiratory lung sound classification


Convolutional Layers:

 The convolutional layers perform feature extraction by applying filters to the input spectrogram.

P(y)(t)= 〖max〗_(a ) y(t-a)                                                                                    ……(12)                                                                                      

Max Pooling Layers:

Max pooling layers down-sample the spatial dimensions of the previous layer. Mathematically, for each max pooling layer:

H_i = MaxPooI(H_(i-1))                                                                                                                    ……(13)
  
H_i = MaxPooI(H_(i-1))                                                                                                                    ……(14)

Flatten Layer:

After several convolutional and max pooling layers, the output is flattened. into a vector to be fed into fully connected layers.
Flatten〖(H〗_i)                                                                                                         ……(15)


Fully Connected Layers:

The flattened vector is passed through fully connected layers for classification.
Mathematically, for    each fully connected layer:

H_i 〖=ReLU(W〗_i 〖.H〗_(i-1)+b_i)                                                                                ……(16)
                                        
Output Layer:

The final fully connected layer is the output layer, which typically uses the SoftMax activation for multi-class classification.

 SoftMax (W_output 〖 .H〗_final)                                                                             ……(17)
             

ResNet50

The ResNet50 model is initialized with pre-trained weights from ImageNet, and an overview of its architecture is displayed. Categorical diagnoses are encoded into numerical labels using Label Encoder and then converted into a one-hot encoded format. The creation of a new model involves adding dense layers to the ResNet50 architecture, with the weights of the ResNet50 layers being frozen, and only the final 25 layers are permitted to be trainable.

To construct a new model, dense layers are incorporated into the ResNet50 architecture, with the ResNet50 layers' weights being frozen and only the final 25 layers allowed to be trainable. The model is compiled using the Adam optimizer, categorical crossentropy loss, and predefined evaluation metrics. Class weights are computed, and callbacks such as model checkpointing,17 and early stopping are defined to address class imbalance in the training data. Additionally, an alternative Convolutional Neural Network (CNN) model is created using Sequential. Both models undergo a specified number of training epochs, during which their accuracy and loss are monitored and visualized. The overall accuracy of the model is approximately 95%.

![image](https://github.com/sandeepsai15634/CNN-BASED-LUNG-SOUND-CLASSIFICATION/assets/119305751/621320da-d144-450f-9298-0e1a1c2c9929)

Fig.7.  Architecture diagram of ResNet50  for respiratory lung sound classification

The ResNet50 architecture, a deep neural network consisting of 50 layers, has been widely applied in various computer vision tasks, notably image categorization. However, since ResNet models are designed and trained for visual inputs like photos, adapting ResNet50 for audio processing might not be straightforward. To utilize ResNet50 for audio tasks, modifications would likely be necessary, possibly requiring integration with architectures tailored for sequential data.

Input:

 - Let x be the input to the layer.
- For an audio file, x could represent a sequence of audio samples.

Linear Transformation (Fully Connected Layer):
z=Wx+b                                                                                                                 ……(18)

- W is the weight matrix.
- b is the bias vector.

 Activation Function 
a = max (0, z)                                                                                                            ……(19)

                        
 - a is the output after applying the activation function.

Normalization (Batch Normalization):
(a ) ̂ = (a-u)/σ                                                                                                 ……(20)


(a ) ̂ is the normalized output.
µ and σ are the mean and standard deviation ,respectively.

Output:

The output y is then passed to the next layer.


CRNN:

CRNN, or Convolutional Recurrent Neural Network, is a blend of recurrent neural networks (RNNs) and convolutional neural networks (CNNs), leveraging the strengths of both. In the original CRNN architecture, the final convolutional layer of a modified CNN is substituted with an RNN layer. This fusion enhances the CRNN's ability to process sequential data effectively, particularly capturing temporal dependencies that might pose challenges for a standalone CNN.
With the inclusion of an RNN component, the CRNN becomes adept at learning sequential patterns and subtle variations, which are crucial for tasks like sound classification. Research indicates that utilizing a CRNN instead of a conventional CNN model often results in enhanced accuracy, particularly in tasks such as sound classification.

![image](https://github.com/sandeepsai15634/CNN-BASED-LUNG-SOUND-CLASSIFICATION/assets/119305751/b7f2fe73-893e-4ea0-8e04-d7d195621264)

Fig.8. Architecture diagram of CRNN for respiratory lung sound classification

The CNN, short for Convolutional Neural Network, is a commonly employed approach for image recognition, drawing inspiration from the receptive field organization found in the human visual cortex. It comprises convolutional, pooling, and fully connected layers.

In the convolution process, the inner product of the weight filter with the input is calculated. This represents the operation conducted on an input image x by a filter spanning H × H pixels.


LSTM:

LSTM, a key architectural design in artificial neural networks (ANNs), finds extensive use in deep learning (DL) and artificial intelligence (AI) domains. Notably, LSTM integrates feedback connections, The current research introduces a hybrid 2D CNN-LSTM network that integrates the FL function to address data imbalance. This network takes respiratory cycles as input and categorizes them into five distinct classes.empowering it to proficiently process sequential data like time series and sound signals. Unique to LSTM are its specialized components known as "LSTM units", which encompass the input gate (it), output gate (Ot), and forget gate (%&). Moreover, LSTM units employ the sigmoid function σ.

σ=  1/〖1+e〗^(-x) 

Moreover, for the LSTM unit to operate effectively, it necessitates cells such as the cell state Ct, candidate state dt, and final output ht (cell output). Additionally, the formula for the tanh function is included.

"tanh = "  ((e^x-e^x))/(〖(e〗^x+e^x))

![image](https://github.com/sandeepsai15634/CNN-BASED-LUNG-SOUND-CLASSIFICATION/assets/119305751/c51665de-1b8b-455f-9a76-4efd3740a898)

Fig.9. Architecture diagram of LSTM  for respiratory lung sound classification

EXPERIMENTAL RESULTS

CNN:

The outcomes indicate that the performance of three models—CNN, VGG16, and ResNet50—fluctuates depending on the task at hand. CNN showcased its effectiveness in data recognition by achieving the highest accuracy rate of 98%. This notable accuracy underscores CNN's ability to detect subtle patterns and characteristics within the data, leading to precise predictions.
 

![image](https://github.com/sandeepsai15634/CNN-BASED-LUNG-SOUND-CLASSIFICATION/assets/119305751/78ed88eb-aebc-4c1a-bb2d-7894b23fa387)

Fig.9 .Accuracy vs epochs

![image](https://github.com/sandeepsai15634/CNN-BASED-LUNG-SOUND-CLASSIFICATION/assets/119305751/53d86670-c956-4676-8e2d-a8e82ca6d454)

Fig.10.Accuracy vs validation loss

Vgg16:

Contrastingly, VGG16 achieved an accuracy of only 83%, significantly lower than that of CNN. Despite its reputation for simplicity and effectiveness, the VGG16 model performed notably poorer in this specific task. This diminished accuracy suggests that VGG16 may have struggled to identify certain subtle nuances within the dataset or to grasp the complexity of the data.

![image](https://github.com/sandeepsai15634/CNN-BASED-LUNG-SOUND-CLASSIFICATION/assets/119305751/a4cfd580-8eb4-4ac9-a808-da70224da889)

Fig.11.Accuracy vs validation

![image](https://github.com/sandeepsai15634/CNN-BASED-LUNG-SOUND-CLASSIFICATION/assets/119305751/d1cbdbe0-ebbf-4bb7-a0ac-afd71d56abed)

Fig.12.1MAE vs Val MAE

ResNet50:

Taking all factors into consideration, the comparison underscores the critical nature of selecting the appropriate model architecture for a given task. While CNN excelled with an accuracy of 98%, VGG16 and ResNet50 also achieved respectable results of 83% and 95%, respectively. These findings underscore the significance of model selection and optimization in maximizing outcomes in machine learning endeavors.

On the contrary, VGG16's accuracy fell short at 83%, markedly lower than CNN's performance. Despite its reputation for simplicity and effectiveness, VGG16 struggled notably in this specific challenge. This reduced accuracy could suggest that VGG16 encountered difficulty in identifying certain subtle aspects within the dataset or in comprehending the complexity of the data.

![image](https://github.com/sandeepsai15634/CNN-BASED-LUNG-SOUND-CLASSIFICATION/assets/119305751/b23087f6-936b-4eb0-9a14-3379fe8a1d5d)

Fig.13.Loss vs Val loss

![image](https://github.com/sandeepsai15634/CNN-BASED-LUNG-SOUND-CLASSIFICATION/assets/119305751/4d367682-cd5e-4bc3-a079-2b2642873a3e)

fig.14.1Accuracy vs validation Accuracy

CRNN:

![image](https://github.com/sandeepsai15634/CNN-BASED-LUNG-SOUND-CLASSIFICATION/assets/119305751/0366f69e-4b2a-41fa-b271-4db8151e3b3e)

Fig.15. epochs vs Accuracy

LSTM:

![image](https://github.com/sandeepsai15634/CNN-BASED-LUNG-SOUND-CLASSIFICATION/assets/119305751/508cd459-6fff-4202-b0ef-1bb21336f559)

Fig.15. epochs vs Accuracy

Table 1: Precision, Recall and F1-Score for CNN
Model       pulmonary       precision    Recall    F1-score
                     Condition
CNN             COPD                  0.58           0.78         0.67
CNN             Bronchiolitis    0.11           0.40         0.17
CNN             Pneumonia       0.94           0.99         0.96
CNN             URTI                     0.80           0.29         0.42
CNN             Healthy               0.11           0.40         0.17


Table 2: Precision, Recall and F1-Score for RCNN
Model       pulmonary       precision    Recall    F1-score
                   Condition 
RCNN             COPD                 1.00           1.00        1.00
RCNN             Bronchiolitis    0.75           1.00        0.86
RCNN             Pneumonia       0.75           0.75        0.75  
RCNN             URTI                     1.00           0.75         0.86 
RCNN             Healthy               0.82           0.69         0.75
      
Table 3: Precision, Recall and F1-Score for LSTM
Model       pulmonary       precision    Recall    F1-score
                   Condition 
LSTM            COPD                  1.00           0.90          0.95
LSTM            Bronchiolitis     0.86          1.00          0.92
LSTM            Pneumonia        0.94          0.99          0.86  
LSTM            URTI                      1.00           0.75          0.86 
LSTM            Healthy                0.82          0.69          0.75 

Table 4
![image](https://github.com/sandeepsai15634/CNN-BASED-LUNG-SOUND-CLASSIFICATION/assets/119305751/befc8367-cf6d-4e82-97a1-989f1e624591)

Conclusion

In a study focusing on lung sound classification, convolutional neural networks (CNNs) have shown promising outcomes in accurately categorizing respiratory sounds. The CNN's capability to automatically extract features from spectrograms has proven beneficial in distinguishing between different respiratory conditions. Additionally, the incorporation of CRNN and LSTM models further enhances the classification accuracy by capturing temporal dependencies in the respiratory sound data. Looking ahead, incorporating real-time monitoring and further enhancing the model could enhance its clinical applicability. By leveraging transfer learning and diversifying datasets, CNN, CRNN, and LSTM models can become more versatile, ultimately boosting their accuracy in identifying respiratory issues. This project sets the groundwork for cutting-edge diagnostic tools in respiratory healthcare and represents a potential advancement in early detection and monitoring.

References 

[1] S. Gairola, F. Tom, N. Kwatra, and M. Jain, “RespireNet: ADeep Neural Network for accurately Detecting Abnormal Lung Sounds in Limited Data Setting” - 2020.

[2] L. Pham, H. Phan, A. Schindler, R. King, A. Mertins, and I. McLoughlin,” Inception-Based Network and Multi-Spectrogram Ensem ble Applied To Predict Respiratory Anomalies and Lung Diseases” - 2021 

[3] Kim, Y., Hyon, Y., Jung, S.S. et al. "Respiratory sound classification for crackles, wheezes, and rhonchi in the clinical field using deep learning". - 2021.

[4] G. Petmezas, G.-A. Cheimariotis, L. Stefanopoulos, B. Rocha, R. P. Paiva, A. K. Katsaggelos, and N. Maglaveras, “Automated Lung Sound Classification Using a Hybrid CNN LSTM Network and Focal Loss Function,” - 2022. 

[5] F. Cinyol, U. Baysal, D. K¨oksal, E. Babao˘glu, and S. S. Ulas¸lı,”Incorporating support vector machine to the classification of respiratory sounds by Convolutional Neural Network,”-  2023.

[6] Naqvi SZH, Choudhry MA, “An Automated System for Classification of Chronic Obstructive Pulmonary Disease and Pneumonia Patients Using Lung Sound Analysis” - 2020.

[7] J. Acharya, A. Basu, “Deep Neural Network for Respiratory Sound Classification in Wearable Devices Enabled by Patient Specific Model Tuning” - 2020.

[8] A. Saraiva, D. Santos, A. Francisco, J. Sousa, N. Ferreira, S. Soares, and A. Valente, “Classification of Respiratory Sounds with Convolutional Neural Network”- 2020.

[9] Siddhartha Gairola, Francis Tom, Nipun Kwatra, and Mohit Jain. Respirenet.” A deep neural network for accurately detecting abnormal lung sounds in limited data setting”. - 2021

[10] Truc Nguyen and Franz Pernkopf. Lung sound classification using co tuning and stochastic normalization. IEEE Transactions on Biomedical Engineering, 2022. 

[11] Georgios Petmezas, Grigorios-Aris Cheimariotis, Leandros Stefanopou los, Bruno Rocha, Rui Pedro Paiva, Aggelos K Katsaggelos, and Nicos Maglaveras. Automated lung sound classification using a hybrid CNN lstm network and focal loss function.- 2022

