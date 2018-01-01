//Auto Grading of Glioma Tumor

//Submitted By
//Usama Pervaiz and Saed Khawaldeh

//Include the DeepLearning4jPackage
package org.deeplearning4j.examples.dataexamples;

//Include all the Libraries
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.plot.Tsne;
import org.deeplearning4j.util.ModelSerializer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.*;
//For saving the Output File
import java.io.File;
import java.util.Random;

/**
 * Created by Usama Pervaiz and Saed Khawaldeh on 18/6/2018.
 */


public class BrainTumor {
    private static Logger log = LoggerFactory.getLogger(BrainTumor.class);

    public static void main(String[] args) throws Exception {

        /* paramters testing
        // image information
        // 28 * 28 grayscale
        // grayscale implies single channel
        //int nChannels =3;
        // int height = 84;
        //int width = 84;
        //int channels = 1;
        //123
       */

     //RNG SEED.... // This random-number generator applies a seed to ensure that
                    //  the same initial weights are used when training.

    //seed(rngSeed)
           //This parameter uses a specific, randomly generated weight initialization.
            //If you run an example many times, and generate new, random weights each
            //time you begin, then your net’s results -- accuracy and F1 score -- may vary a great deal,
            //because different initial weights can lead algorithms to different local minima in the errorscape.
            //Keeping the same random weights allows you isolate the effect of adjusting other hyperparameters more clearly,
            //while other conditions remain equal.
        int rngseed = 160;
        Random randNumGen = new Random(rngseed);

        //Checked with 30 40 50 60 70
        int batchSize = 50; //// How many examples to fetch with each step.
        //The batchSize and numEpochs have to be chosen based on experience; you learn what works through experimentation.
        // A larger batch size results in faster training, while more epochs, or passes through the dataset, result in better accuracy.
        //However, there are diminishing returns beyond a certain number of epochs, so there is a trade off between accuracy and training speed.
        int nEpochs = 5;//Tested for 5,10,12,15,20

      //This will be the dimension of the Input Image

        int height=160;
        int width=160;

        int channels = 1;  //We will only work on the grayscale Image
        int outputNum = 3; //The Output classes we have, In our case, we have three classes which are Low Grade,High Grade and Healthy Subjects.

        //Same as rng seed, Size equal to the Size of the Input Image
        long seed = 160;

       // Each iteration, for a neural network, is a learning step; i.e. an update of the model's weights. The network is exposed to data, makes guesses
       // about the data, and then corrects its own parameters based on how wrong its guesses were. So more iterations allow neural networks to take more
       // steps and to learn more, minimizing error.
        int iterations = 40;//Tested for 40,50,60,70,80

        // Define the File Paths
        //Load the Testing and Training Data
        //Just name the folders 1,2,3-for three classes
        File trainData = new File("H:\\deep learning\\data\\Braindata1\\Testing");
        File testData = new File("H:\\deep learning\\data\\Braindata1\\Training");

        // Define the FileSplit(PATH, ALLOWED FORMATS,random)
        //Splits up a root directory in to files.
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS,randNumGen);
        FileSplit test = new FileSplit(testData,NativeImageLoader.ALLOWED_FORMATS,randNumGen);

        // Extract the parent path as the image label
        //The Name of the folder of the Path will be taken as the labels for the data

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        //The Image will be downsampled here
        //My Inout Image size is 256*256, Downsampled to 160*160

        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);

        // Initialize the record reader
        // add a listener, to extract the name

        recordReader.initialize(train);
        //recordReader.setListeners(new LogRecordListener());

        // DataSet Iterator

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);

        // Scale pixel values to 0-1

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

      //Value of the constant
        double nonZeroBias = 1;

        //The key idea is to randomly drop units (along with their connections) from the neural
        //network during training. This prevents units from co-adapting too much. During training,
        //dropout samples from an exponential number of different “thinned” networks. At test time,
        // it is easy to approximate the effect of averaging the predictions of all these thinned networks
        //by simply using a single unthinned network that has smaller weights. This significantly
        //reduces overfitting and gives major improvements over other regularization methods
        //Referencee----https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
        double dropOut = 0.5;

        //Downsampling, or "pooling" layers are often placed after convolutional layers in a ConvNet, mainly to reduce the feature map
       //dimensionality for computational efficiency, which can in turn improve actual performance.

        //The main type of pooling layer in use today is a "max pooling" layer, where the feature map is downsampled in such a way that the
       //maximum feature response within a given sample size is retained. This is in contrast with average pooling, where you basically just
        //lower the resolution by averaging together a group of pixels. Max pooling tends to do better because it is more responsive to kernels
        //that are "lit up" or respond to patterns detected in the data.
        SubsamplingLayer.PoolingType poolingType = SubsamplingLayer.PoolingType.MAX;

        //TODO split and link kernel maps on GPUs - 2nd, 4th, 5th convolution should only connect maps on the same gpu, 3rd connects to all in 2nd
        //WeightInit.DISTRIBUTION---Sample weights from a provided distribution<br>
        //RELU ACTIVATION LAYER

        //Learning Rate---This line sets the learning rate, which is the size of the adjustments made to the weights with each iteration, the step size.
        // A high learning rate makes a net traverse the errorscape quickly, but also makes it prone to overshoot the point of minimum error.
        // A low learning rate is more likely to find the minimum, but it will do so very slowly, because it is taking small steps in adjusting the weights.

        //Momentum--Momentum is an additional factor in determining how fast an optimization algorithm converges on the optimum point. Momentum affects the direction that weights are adjusted in,
        // so in the code we consider it a weight updater.

        //  .regularization(true).l2(1e-4)
        //Regularization is a technique to prevent what’s called overfitting. Overfitting is when the model fits the training data really well,
        // but performs poorly in real life as soon as it's exposed to data it hasn’t seen before.
        //We use L2 regularization, which prevents individual weights from having too much influence on the overall results.

        MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .weightInit(WeightInit.DISTRIBUTION)
            .dist(new NormalDistribution(0.0, 0.01))
            .activation(Activation.RELU)
            .updater(Updater.NESTEROVS)
            .iterations(iterations)
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(1e-2)
            .biasLearningRate(1e-2 * 2)
            .learningRateDecayPolicy(LearningRatePolicy.Step)
            .lrPolicyDecayRate(0.1)
            .lrPolicySteps(100000)
            .regularization(true)
            .l2(5 * 1e-4)
            .momentum(0.9)
            .miniBatch(false)
            .list()
        //The list specifies the number of layers in the net; this function replicates your configuration n times and builds a layerwise configuration.
            //20 20
        //This Layer Structure is based on the modified Version of the Alex-Net which was presented by Prof. Alex on the Image net classification paper.
            .layer(0, new ConvolutionLayer.Builder(new int[]{20, 20}, new int[]{4, 4}, new int[]{3, 3})
                .name("cnn1")
                .nIn(channels)
                .nOut(96)
                .build())
            .layer(1, new LocalResponseNormalization.Builder()
                .name("lrn1")
                .build())
            .layer(2, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                .name("maxpool1")
                .build())
            .layer(3, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{2, 2})
                .name("cnn2")
                .nOut(256)
                .biasInit(nonZeroBias)
                .build())
            .layer(4, new LocalResponseNormalization.Builder()
                .name("lrn2")
                .k(2).n(5).alpha(1e-4).beta(0.75)
                .build())
            .layer(5, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                .name("maxpool2")
                .build())
            .layer(6, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                .name("cnn3")
                .nOut(384)
                .build())
            .layer(7, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                .name("cnn4")
                .nOut(384)
                .biasInit(nonZeroBias)
                .build())
            .layer(8, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                .name("cnn5")
                .nOut(256)
                .biasInit(nonZeroBias)
                .build())
            .layer(9, new SubsamplingLayer.Builder(poolingType, new int[]{3, 3}, new int[]{2, 2})
                .name("maxpool3")
                .build())
            .layer(10, new DenseLayer.Builder()
                .name("ffn1")
                .nOut(4096)
                .dist(new GaussianDistribution(0, 0.005))
                .biasInit(nonZeroBias)
                .dropOut(dropOut)
                .build())
            .layer(11, new DenseLayer.Builder()
                .name("ffn2")
                .nOut(4096)
                .dist(new GaussianDistribution(0, 0.005))
                .biasInit(nonZeroBias)
                .dropOut(dropOut)
                .build())
            .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .name("output")
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .build())

           //Run the Back propogation Algorithm
            .backprop(true)
            .pretrain(false)

            .cnnInputSize(height, width, channels);


        /*
        Regarding the .setInputType(InputType.convolutionalFlat(20,20,1)) line: This does a few things.
        (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
            and the dense layer
        (b) Does some additional configuration validation
        (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
            layer based on the size of the previous layer (but it won't override values manually set by the user)
        */

        MultiLayerNetwork model = new MultiLayerNetwork(conf.build());
        model.init();

        //Display the Output and the Score at Each Iteration

        log.info("Train model....");
        model.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<nEpochs; i++ ) {
            model.fit(dataIter);
            log.info("*** Completed epoch {} ***", i);

            log.info("******SAVE TRAINED MODEL******");
            // Details

            // Where to save model
            File locationToSave = new File("trained_cancer_model.zip");

            // boolean save Updater
            boolean saveUpdater = false;

            // ModelSerializer needs modelname, saveUpdater, Location

            ModelSerializer.writeModel(model,locationToSave,saveUpdater);

            //Evaluate the Results
               // Accuracy - The percentage of images that were correctly identified by our model.
               // Precision - The number of true positives divided by the number of true positives and false positives.
               // Recall - The number of true positives divided by the number of true positives and the number of false negatives.
               // F1 Score - Weighted average of precision and recall.

            log.info("****evaluate model****");

            recordReader.reset();
            recordReader.initialize(test);
            DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
            scaler.fit(testIter);
            testIter.setPreProcessor(scaler);

            //Creat eval object
            Evaluation eval = new Evaluation(outputNum);

            while(testIter.hasNext()){

                DataSet next = testIter.next();
                INDArray output = model.output(next.getFeatureMatrix());
                eval.eval(next.getLabels(),output);

            }
            log.info(eval.stats());



        }




    }}
