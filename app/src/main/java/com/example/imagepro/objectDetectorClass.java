package com.example.imagepro;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Typeface;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import org.checkerframework.checker.units.qual.A;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class  objectDetectorClass {

    // this is used to load model and predict
    private Interpreter interpreter;
    private Interpreter interpreter2;
    // store all label in array
    private List<String> labelList;
    private int INPUT_SIZE;
    private int PIXEL_SIZE=3; // for RGB
    private int IMAGE_MEAN=0;
    private  float IMAGE_STD=255.0f;
    // use to initialize gpu in app
    private GpuDelegate gpuDelegate;
    private int height=0;
    private  int width=0;
    private int Classification_Input_Size = 0;
    TextView letter, combine;
    Button add;
    objectDetectorClass(AssetManager assetManager,String modelPath, String labelPath,int inputSize, String classification_model, int classification_input_size) throws IOException{
        INPUT_SIZE=inputSize;
        Classification_Input_Size = classification_input_size;
        // use to define gpu or cpu // no. of threads
        Interpreter.Options options=new Interpreter.Options();
        gpuDelegate=new GpuDelegate();
        options.addDelegate(gpuDelegate);
        options.setNumThreads(4); // set it according to your phone
        // loading model
        interpreter=new Interpreter(loadModelFile(assetManager,modelPath),options);
        // load labelmap
        labelList=loadLabelList(assetManager,labelPath);

        //load model
        Interpreter.Options options2 = new Interpreter.Options();
        //add 2 threads to this interpreter
        options2.setNumThreads(2);
        //load model
        interpreter2 = new Interpreter(loadModelFile(assetManager,classification_model),options2);

    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        // to store label
        List<String> labelList=new ArrayList<>();
        // create a new reader
        BufferedReader reader=new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        // loop through each line and store it to labelList
        while ((line=reader.readLine())!=null){
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        // use to get description of file
        AssetFileDescriptor fileDescriptor=assetManager.openFd(modelPath);
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset =fileDescriptor.getStartOffset();
        long declaredLength=fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }
    // create new Mat function
    public Mat recognizeImage(Mat mat_image, TextView letterView, TextView combineView, Button addButton){
        // Rotate original image by 90 degree get get portrait frame

        letter = letterView;
        combine = combineView;
        add = addButton;

        Mat rotated_mat_image=new Mat();

        Mat a=mat_image.t();
        Core.flip(a,rotated_mat_image,1);
        // Release mat
        a.release();

        // now convert it to bitmap
        Bitmap bitmap=null;
        bitmap=Bitmap.createBitmap(rotated_mat_image.cols(),rotated_mat_image.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rotated_mat_image,bitmap);
        // define height and width
        height=bitmap.getHeight();
        width=bitmap.getWidth();

        // scale the bitmap to input size of model
         Bitmap scaledBitmap=Bitmap.createScaledBitmap(bitmap,INPUT_SIZE,INPUT_SIZE,false);

         // convert bitmap to bytebuffer as model input should be in it
        ByteBuffer byteBuffer=convertBitmapToByteBuffer(scaledBitmap);

        // defining output
        // 10: top 10 object detected
        // 4: there coordinate in image
        //  float[][][]result=new float[1][10][4];
        Object[] input=new Object[1];
        input[0]=byteBuffer;

        Map<Integer,Object> output_map=new TreeMap<>();
        // we are not going to use this method of output
        // instead we create treemap of three array (boxes,score,classes)

        float[][][]boxes =new float[1][10][4];
        // 10: top 10 object detected
        // 4: there coordinate in image
        float[][] scores=new float[1][10];
        // stores scores of 10 object
        float[][] classes=new float[1][10];
        // stores class of object

        // add it to object_map;
        output_map.put(0,boxes);
        output_map.put(1,classes);
        output_map.put(2,scores);

        // now predict
        interpreter.runForMultipleInputsOutputs(input,output_map);

        Object value=output_map.get(0);
        Object Object_class=output_map.get(1);
        Object score=output_map.get(2);

        // loop through each object
        // as output has only 10 boxes
        for (int i=0;i<10;i++){
            //looping for each hand detected
            float class_value=(float) Array.get(Array.get(Object_class,0),i);
            float score_value=(float) Array.get(Array.get(score,0),i);

            // Now we will do some change to improve app
            if(score_value>0.5){
                Object box1=Array.get(Array.get(value,0),i);
                // we are multiplying it with Original height and width of frame

                float y1=(float) Array.get(box1,0)*height;
                float x1=(float) Array.get(box1,1)*width;
                float y2=(float) Array.get(box1,2)*height;
                float x2=(float) Array.get(box1,3)*width;

                //set boundary limit
                if(y1<0){
                    y1=0;
                }
                if(x1<0){
                    x1=0;
                }
                if(y2>height){
                    y2=height;
                }
                if(x2>width){
                    x2=width;
                }

                //set height and width of box
                float w1=x2-x1;
                float h1=y2-y1;

                //crop hand image from original frame
                Rect cropped_roi = new Rect((int)x1,(int)y1,(int)w1,(int)h1);
                Mat cropped = new Mat(rotated_mat_image,cropped_roi).clone();

                //convert cropped mat to Bitmap
                Bitmap bitmap1 = null;
                bitmap1 = Bitmap.createBitmap(cropped.cols(),cropped.rows(),Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(cropped,bitmap1);

                //resize bitmap1 to classification input size 96
                Bitmap scaledBitmap1 = Bitmap.createScaledBitmap(bitmap1,Classification_Input_Size, Classification_Input_Size, false);

                //convert scaled bitmap to byte buffer
                ByteBuffer byteBuffer1 = convertBitmapToByteBuffer2(scaledBitmap1);

                //array for interpreter2 output
                float [][] output_class_value = new float[1][1];

                //predict output for byteBuffer1
                interpreter2.run(byteBuffer1,output_class_value);
                Log.d("ObjectDetectionClass","output_class_value: "+output_class_value[0][0]);

                //convert output_class_value to alphabets
                String sign_val = get_alphabets(output_class_value[0][0]);
                Log.d("ObjectDetectionClass","letter: "+ sign_val);

                letter.setText(sign_val);

                add.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        if(sign_val!=null){
                            String text = combine.getText().toString()+sign_val;
                            combine.setText(text);
                        }
                    }
                });

                //add class name in image
                Imgproc.putText(rotated_mat_image,"",new Point(x1+10,y1+40),3,1.5,new Scalar(255, 255, 255, 255),2);
                Imgproc.rectangle(rotated_mat_image,new Point(x1,y1),new Point(x2,y2),new Scalar(0, 255, 0, 255),2);

            }

        }

        Mat b=rotated_mat_image.t();
        Core.flip(b,mat_image,0);
        b.release();

        return mat_image;
    }

    private String get_alphabets(float sig_v) {
        String val = "";

        if(sig_v >= -0.2 && sig_v < 0.2){
            val = "ط";
        } else if(sig_v >= 0.8 && sig_v < 1.1){
            val = "ح";
        } else if(sig_v >= 2.9 && sig_v < 3.1){
            val = "ال";
        } else if(sig_v >= 3.8 && sig_v < 5){
            val = "ا";
        } else if(sig_v >= 5 && sig_v < 6){
            val = "ب";
        } else if(sig_v >= 6.5 && sig_v < 7.5){
            val = "د";
        } else if(sig_v >= 7.5 && sig_v < 8.7){
            val = "ذ";
        } else if(sig_v >= 8.7 && sig_v < 9.8){
            val = "ف";
        } else if(sig_v >= 9.8 && sig_v < 11.5){
            val = "ج";
        } else if(sig_v >= 12 && sig_v < 13.8){
            val = "ه";
        } else if(sig_v >= 13.8 && sig_v <14.2){
            val = "ك";
        } else if(sig_v >= 14.2 && sig_v < 15.8){
            val = "خ";
        } else if(sig_v >= 15.8 && sig_v < 16.8){
            val = "لا";
        } else if(sig_v >= 16.8 && sig_v < 17.2){
            val = "ل";
        } else if(sig_v >= 17.2 && sig_v < 17.65){
            val = "م";
        } else if(sig_v >= 17.65 && sig_v < 19){
            val = "و";
        } else if(sig_v >= 19 && sig_v < 19.5){
            val = "ز";
        } else if(sig_v >= 19.5 && sig_v < 20.5){
            val = "ر";
        } else if(sig_v >= 20.5 && sig_v < 21){
            val = "و";
        } else if(sig_v >= 21 && sig_v < 21.9){
            val = "س";
        } else if(sig_v >= 21.9 && sig_v < 23){
            val = "ش";
        } else if(sig_v >= 23.5 && sig_v < 24.5){
            val = "ت";
        } else if(sig_v >= 24.5 && sig_v < 25.5){
            val = "ة";
        } else if(sig_v >= 25.5 && sig_v < 26){
            val = "ظ";
        } else if(sig_v >= 26 && sig_v < 27){
            val = "ث";
        } else if(sig_v >= 27 && sig_v < 27.9){
            val = "لا";
        } else if(sig_v >= 27.9 && sig_v < 28.5){
            val = "ع";
        } else if(sig_v >= 28.5 && sig_v < 28.9){
            val = "م";
        } else if(sig_v >= 28.9 && sig_v < 29.2){
            val = "ي";
        } else if(sig_v >= 29.2 && sig_v < 30.5){
            val = "ئ";
        } else{
            val = " ";
        }

        return val;
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;

        int quant=1;
        int size_images=INPUT_SIZE;
        if(quant==0){
            byteBuffer=ByteBuffer.allocateDirect(1*size_images*size_images*3);
        }
        else {
            byteBuffer=ByteBuffer.allocateDirect(4*1*size_images*size_images*3);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues=new int[size_images*size_images];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pixel=0;

        for (int i=0;i<size_images;++i){
            for (int j=0;j<size_images;++j){
                final  int val=intValues[pixel++];
                if(quant==0){
                    byteBuffer.put((byte) ((val>>16)&0xFF));
                    byteBuffer.put((byte) ((val>>8)&0xFF));
                    byteBuffer.put((byte) (val&0xFF));
                }
                else {
                    byteBuffer.putFloat((((val >> 16) & 0xFF))/255.0f);
                    byteBuffer.putFloat((((val >> 8) & 0xFF))/255.0f);
                    byteBuffer.putFloat((((val) & 0xFF))/255.0f);
                }
            }
        }
    return byteBuffer;
    }

    private ByteBuffer convertBitmapToByteBuffer2(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        int quant=1;
        int size_images=Classification_Input_Size;
        if(quant==0){
            byteBuffer=ByteBuffer.allocateDirect(1*size_images*size_images*3);
        }
        else {
            byteBuffer=ByteBuffer.allocateDirect(4*1*size_images*size_images*3);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues=new int[size_images*size_images];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pixel=0;
        for (int i=0;i<size_images;++i){
            for (int j=0;j<size_images;++j){
                final  int val=intValues[pixel++];
                if(quant==0){
                    byteBuffer.put((byte) ((val>>16)&0xFF));
                    byteBuffer.put((byte) ((val>>8)&0xFF));
                    byteBuffer.put((byte) (val&0xFF));
                }
                else {
                    byteBuffer.putFloat((((val >> 16) & 0xFF)));
                    byteBuffer.putFloat((((val >> 8) & 0xFF)));
                    byteBuffer.putFloat((((val) & 0xFF)));
                }
            }
        }
        return byteBuffer;
    }
}