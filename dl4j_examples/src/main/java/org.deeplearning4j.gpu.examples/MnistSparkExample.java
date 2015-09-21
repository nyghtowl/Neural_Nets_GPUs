package org.deeplearning4j.gpu.examples;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.feature.StandardScaler;
import org.apache.spark.mllib.feature.StandardScalerModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.catalyst.expressions.In;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
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
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.util.MLLibUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;
import scala.tools.cmd.gen.AnyVals;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.util.List;


/**
 * Created by nyghtowl on 9/18/15.
 * Run on Spark to compare for speed
 */
public class MnistSparkExample {
    private static Logger log = LoggerFactory.getLogger(MnistExample.class);

    public static void main(String[] args) throws Exception {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10;
        int numSamples = 10;
        int batchSize = 10;
        int nChannels = 1;
        final JavaSparkContext sc = new JavaSparkContext(new SparkConf().setMaster("local[*]").setAppName("mnist"));

        log.info("Load data....");
//        DataSetIterator iter = new MnistDataSetIterator(batchSize, numSamples);
        DataSet mnist = new MnistDataSetIterator(batchSize, numSamples).next();


        JavaRDD<LabeledPoint> data = MLLibUtil.fromDataSet(sc,
                sc.parallelize(mnist.asList()));
        StandardScaler scaler = new StandardScaler(true,true);

        final StandardScalerModel scalarModel = scaler.fit(data.map(new Function<LabeledPoint, Vector>() {
            @Override
            public Vector call(LabeledPoint v1) throws Exception {
                return v1.features();
            }
        }).rdd());

        //get the trained data for the train/test split
        JavaRDD<LabeledPoint> normalizedData = data.map(new Function<LabeledPoint, LabeledPoint>() {
            @Override
            public LabeledPoint call(LabeledPoint v1) throws Exception {
                Vector features = v1.features();
                Vector normalized = scalarModel.transform(features);
                return new LabeledPoint(v1.label(), normalized);
            }
        }).cache();

        //train test split
        JavaRDD<LabeledPoint>[] trainTestSplit = normalizedData.randomSplit(new double[]{80, 20});

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
                        .nOut(outputNum)
                        .activation("softmax")
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new GaussianDistribution(0, .01))
                        .build())
                .backprop(true)
                .pretrain(false);

        new ConvolutionLayerSetup(conf,numRows,numColumns,nChannels);

        log.info("Train model....");
        SparkDl4jMultiLayer trainLayer = new SparkDl4jMultiLayer(sc.sc(),conf.build());
        MultiLayerNetwork trainedNetwork = trainLayer.fit(trainTestSplit[0],batchSize);
        final SparkDl4jMultiLayer trainedNetworkWrapper = new SparkDl4jMultiLayer(sc.sc(),trainedNetwork);

        log.info("Evaluate model....");
        // Compute raw scores on the test set.

        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = trainTestSplit[1].map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(LabeledPoint p) {
                        Vector prediction = trainedNetworkWrapper.predict(p.features());
                        double max = 0;
                        double idx = 0;
                        for(int i = 0; i < prediction.size(); i++) {
                            if(prediction.apply(i) > max) {
                                idx = i;
                                max = prediction.apply(i);
                            }
                        }

                        return new Tuple2<>((Object) idx, (Object) p.label());
                    }
                }
        );

        //Get evaluation metrics.
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        double precision = metrics.fMeasure();
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream("model.bin"));
        Nd4j.write(bos,trainedNetwork.params());
        // FileUtils.write(new File("conf.yaml"),trainedNetwork.conf().toYaml());
        System.out.println("F1 = " + precision);



//        final Evaluation eval = new Evaluation(outputNum);
//        JavaRDD<Vector> evalPreductions = predictionAndLabels.reduce(
//                new Function2<Double, Double, Double>() {
//                    public Double call(Double k, Double v) {
//                        INDArray newK = Nd4j.create(k);
//                        INDArray newV = Nd4j.create(v);
//                        eval.eval(newK, newV);
//                        return k;
//                    }
//                }
//        );
//        eval.eval();
//        log.info(eval.stats());

        log.info("****************Example finished********************");


    }

}
