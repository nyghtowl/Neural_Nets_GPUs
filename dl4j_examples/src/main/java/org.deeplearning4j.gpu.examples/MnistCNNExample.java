package org.deeplearning4j.gpu.examples;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Created by nyghtowl on 9/18/15.
 */
public class MnistCNNExample {
    private static Logger log = LoggerFactory.getLogger(MnistCNNExample.class);

    public static void main(String[] args) throws Exception {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        final int numRows = 28;
        final int numColumns = 28;
        int numSamples = 10000;
        int batchSize = 500;
        int nChannels = 1;
        int outputNum = 10;
        int iterations = 10;
        int seed = 123;
        int splitTrainNum = (int) (batchSize*.8);
        int listenerFreq = iterations/5;
        DataSet mnist;
        SplitTestAndTrain trainTest;
        DataSet trainInput;
        List<INDArray> testInput = new ArrayList<>();
        List<INDArray> testLabels = new ArrayList<>();


        log.info("Load data....");
        DataSetIterator data = new MnistDataSetIterator(batchSize, numSamples);

        log.info("Build model....");
        MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .learningRate(0.01)
                .constrainGradientToUnitNorm(true)
                .regularization(true).l2(5 * 1e-4)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .useDropConnect(true)
                .list(6)
                .layer(0, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1})
                        .nIn(numRows * numColumns)
                        .nOut(20)
                        .activation("relu")
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new GaussianDistribution(0, .01))
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(2, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1})
                        .nOut(50)
                        .activation("relu")
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new GaussianDistribution(0, .01))
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .nOut(500)
                        .activation("relu")
                        .dropOut(0.5)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new GaussianDistribution(0, .01))
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new GaussianDistribution(0, .01))
                        .build())
                .backprop(true)
                .pretrain(false);

        new ConvolutionLayerSetup(conf,numRows,numColumns,nChannels);

        MultiLayerNetwork model = new MultiLayerNetwork(conf.build());
        model.init();

        log.info("Train model....");
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));
        while(data.hasNext()) {
            mnist = data.next();
            trainTest = mnist.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
            trainInput = trainTest.getTrain(); // get feature matrix and labels for training
            testInput.add(trainTest.getTest().getFeatureMatrix());
            testLabels.add(trainTest.getTest().getLabels());
            model.fit(trainInput);
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        for(int i = 0; i < testInput.size(); i++) {
            INDArray output = model.output(testInput.get(i));
            eval.eval(testLabels.get(i), output);
        }
        INDArray output = model.output(testInput.get(0));
        eval.eval(testLabels.get(0), output);
        log.info(eval.stats());

        log.info("****************Example finished********************");


    }

}
