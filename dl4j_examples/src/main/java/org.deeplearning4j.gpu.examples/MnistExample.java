package org.deeplearning4j.gpu.examples;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



public class MnistExample {
    private static Logger log = LoggerFactory.getLogger(MnistExample.class);


    public static void main(String[] args) throws Exception {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10;
        int numSamples = 100;
        int batchSize = 50;
        int iterations = 5;
        int seed = 123;

        log.info("Load data....");
        DataSetIterator iter = new MnistDataSetIterator(batchSize, numSamples);



        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .constrainGradientToUnitNorm(true)
                .iterations(iterations)
                .learningRate(1e-3f)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .list(3)
                .layer(0, new DenseLayer.Builder()
                        .nIn(numRows*numColumns)
                        .nOut(1000)
                        .activation("sigmoid")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(1000)
                        .nOut(500)
                        .activation("sigmoid")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(500)
                        .nOut(outputNum)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .backprop(true)
                .pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info("Train model....");
        model.fit(iter);

        iter.reset();

        log.info("Evaluate weights....");
        for (org.deeplearning4j.nn.api.Layer layer : model.getLayers()) {
            INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
            log.info("Weights: " + w);
        }

        DataSetIterator testIter = new MnistDataSetIterator(100, 10000);
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
