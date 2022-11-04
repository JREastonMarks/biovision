package edu.uh.compsci.eastonmark.jeremy.sandbox;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;

public class Test01 {

	public static void main(String[] args) {
		System.out.println("hello world");
	}

	public void detectObjectOnVideo() throws FileNotFoundException
    // load the COCO class labels our YOLO model was trained on
    Scanner scan = new Scanner(new FileReader("d:\\Eclipse_2019_06\\myWork\\OpenCV_Demo\\yolo-coco\\coco.names"));
    List<String> cocoLabels = new ArrayList<String>();
    while(scan.hasNextLine()) {
        cocoLabels.add(scan.nextLine());        
    }
    
    // load our YOLO object detector trained on COCO dataset
    Net dnnNet = Dnn.readNetFromDarknet("d:\\Eclipse_2019_06\\myWork\\OpenCV_Demo\\yolo-coco\\yolov3.cfg",
                                            "d:\\Eclipse_2019_06\\myWork\\OpenCV_Demo\\yolo-coco\\yolov3.weights");
    // YOLO on GPU:
        dnnNet.setPreferableBackend(Dnn.DNN_BACKEND_CUDA);
        dnnNet.setPreferableTarget(Dnn.DNN_TARGET_CUDA);
        
        // generate radnom color in order to draw bounding boxes
        Random random = new Random();
        ArrayList<Scalar> colors = new ArrayList<Scalar>();
        for (int i= 0; i < cocoLabels.size(); i++) {
             colors.add(new Scalar( new double[] {random.nextInt(255), random.nextInt(255), random.nextInt(255)}));
        }
        
        List<String> layerNames = dnnNet.getLayerNames();
        List<String> outputLayers = new ArrayList<String>();
        for (Integer i : dnnNet.getUnconnectedOutLayers().toList()) {
            outputLayers.add(layerNames.get(i - 1));
        }

        // load our video:
        VideoCapture capture = new VideoCapture();
        capture.open("d:\\Eclipse_2019_06\\myWork\\OpenCV_Demo\\videos\\car_chase_01.mp4");
        double frmCount = 0;
        if( capture.isOpened()){
            frmCount = capture.get(Videoio.CAP_PROP_FRAME_COUNT);
            System.out.println("frmCount = " + frmCount);
            // CAP_PROP_FRAME_COUNT
        }else {
            System.out.println("Capture is not opened!");
            return;
        }
        
        // # loop over frames from the video file stream
        Mat img = new Mat();
        double fps = capture.get(Videoio.CAP_PROP_FPS);
        Size size = new Size(capture.get(Videoio.CAP_PROP_FRAME_WIDTH), capture.get(Videoio.CAP_PROP_FRAME_HEIGHT));
        VideoWriter writer = new VideoWriter("d:\\Eclipse_2019_06\\myWork\\OpenCV_Demo\\videos\\outputs\\video_out.mp4",
                                      VideoWriter.fourcc('m','p','g','4'),
                                              fps, size, true);
        while (true){
            capture.read(img);
            if (img.empty())
                break;
            HashMap<String, List>  result = forwardImageOverNetwork(img, dnnNet, outputLayers);
            
            ArrayList<Rect2d> boxes = (ArrayList<Rect2d>)result.get("boxes");
            ArrayList<Float> confidences = (ArrayList<Float>) result.get("confidences");
            ArrayList<Integer> class_ids = (ArrayList<Integer>)result.get("class_ids");
            
            // -- Now , do so-called “non-maxima suppression”
            //Non-maximum suppression is performed on the boxes whose confidence is equal to or greater than the threshold.
            // This will reduce the number of overlapping boxes:
            MatOfInt indices =  getBBoxIndicesFromNonMaximumSuppression(boxes,
                                                                        confidences);
            //-- Finally, go over indices in order to draw bounding boxes on the image:
            img =  drawBoxesOnTheImage(img,
                                       indices,
                                       boxes,
                                       cocoLabels,
                                       class_ids,
                                       colors);
            writer.write(img);
            
        }// of main while()
        capture.release();  
        writer.release();
}
}
