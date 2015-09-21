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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by nyghtowl on 9/18/15.
 * Version matches Caffe & Theano for comparison
 */
public class MnistCompareExample {
    private static Logger log = LoggerFactory.getLogger(MnistExample.class);

    public static void main(String[] args) throws Exception {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        final int numRows = 28;
        final int numColumns = 28;
        int numSamples = 100;
        int batchSize = 50;
        int nChannels = 1;

        log.info("Load data....");
        DataSetIterator iter = new MnistDataSetIterator(batchSize, numSamples);

        MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .iterations(5)
                .learningRate(0.01)
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
                        .nOut(10)
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
        model.fit(iter);

        iter.reset();

        DataSetIterator testIter = new MnistDataSetIterator(batchSize, numSamples);
        log.info("Evaluate model....");
        Evaluation eval = new Evaluation();
        while (testIter.hasNext()) {
            DataSet test_data = testIter.next();
            INDArray predict2 = model.output(test_data.getFeatureMatrix());
            eval.eval(test_data.getLabels(), predict2);
        }
        log.info(eval.stats());
        log.info("****************Example finished********************");


    }

}
