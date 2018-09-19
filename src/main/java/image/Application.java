package image;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.Utils.ImageNetLabels;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class Application {
    NativeImageLoader nativeImageLoader;
    ComputationGraph vgg16;
    public static void main(String[] args) {
        Logger logger = LoggerFactory.getLogger(Application.class);

        try {
            ZooModel zooModel = new VGG16();
            ComputationGraph vgg16 = (ComputationGraph) new VGG16().initPretrained(PretrainedType.IMAGENET);
           // ComputationGraph vgg16 = VGG16
            NativeImageLoader nativeImageLoader = new NativeImageLoader(224, 224, 3);

            INDArray image = nativeImageLoader.asMatrix(new File("C:\\Users\\DL\\Desktop\\foto\\109APPLE\\IMG_9606.JPG"));
            DataNormalization scaler = new VGG16ImagePreProcessor();
            scaler.transform(image);

            INDArray[] output = vgg16.output(false, image);

            INDArray encodedPredictions = output[0];

            List<Prediction> decodedPredictions = new ArrayList<>();
            int[] top5 = new int[5];
            float[] top5Prob = new float[5];

            ArrayList<String> labels = ImageNetLabels.getLabels();
            int i = 0;

            for (INDArray currentBatch = encodedPredictions.getRow(0).dup(); i < 5; ++i) {

                top5[i] = Nd4j.argMax(currentBatch, 1).getInt(0, 0);
                top5Prob[i] = currentBatch.getFloat(0, top5[i]);
                currentBatch.putScalar(0, top5[i], 0.0D);

                decodedPredictions.add(new Prediction(labels.get(top5[i]), (top5Prob[i] * 100.0F)));

            }
            String top1 = decodedPredictions.get(0).toString();
            System.setProperty("webdriver.chrome.driver", "chromedriver.exe");
            WebDriver driver = new ChromeDriver();

            driver.navigate().to("https://search.naver.com/search.naver?where=image&sm=tab_jum&query="+top1);

            //decodedPredictions.get(0);
            System.out.println(predictionsToString(decodedPredictions));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static String predictionsToString(List<Prediction> predictions) {
        StringBuilder builder = new StringBuilder();
        for (Prediction prediction : predictions) {
            builder.append(prediction.toString());
            builder.append('\n');
        }
        return builder.toString();
    }
}
