package edu.uh.compsci.eastonmark.jeremy.sandbox;

import java.util.List;


import org.bytedeco.javacv.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_imgproc.*;
import org.bytedeco.opencv.opencv_calib3d.*;
import org.bytedeco.opencv.opencv_objdetect.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_calib3d.*;
import static org.bytedeco.opencv.global.opencv_objdetect.*;


class ObjectDetection2 {
    public void detectAndDisplay(Mat frame, CascadeClassifier carCascade) {
        Mat frameGray = new Mat();
        Imgproc.cvtColor_1(frame, frameGray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(frameGray, frameGray);

        // -- Detect Cars
        MatOfRect cars = new MatOfRect();
        carCascade.detectMultiScale(frameGray, cars);

        List<Rect> listOfCars = cars.toList();
        for (Rect car : listOfCars) {
            Point center = new Point(car.x + car.width / 2, car.y + car.height / 2);
            Imgproc.ellipse(frame, center, new Size(car.width / 2, car.height / 2), 0, 0, 360,
                    new Scalar(255, 0, 255));
        }

        //-- Show what you got
        HighGui.imshow("Car Detection", frame );
    }

    public void run(String[] args) {
        String carCascadeFile = "/Users/jreastonmarks/Downloads/haarcascade_car.xml";

        CascadeClassifier carCascade = new CascadeClassifier();

        if (!carCascade.load(carCascadeFile)) {
            System.err.println("--(!)Error loading car cascade: " + carCascade);
            System.exit(0);
        }
        

        String filename = args[0];
        VideoCapture capture = new VideoCapture(filename);
        if (!capture.isOpened()) {
        	System.out.println("Unable to open file!");
            System.exit(-1);
        }

        Mat frame = new Mat();
        while (capture.read(frame)) {
            if (frame.empty()) {
                System.err.println("--(!) No captured frame -- Break!");
                break;
            }

            //-- 3. Apply the classifier to the frame
            detectAndDisplay(frame, carCascade);

            if (HighGui.waitKey(10) == 27) {
                break;// escape
            }
        }

        System.exit(0);
    }
}

public class ObjectDetectionDemo2 {
	static {
//		nu.pattern.OpenCV.loadShared();
	}
	
    public static void main(String[] args) {
        new ObjectDetection().run(args);
    }
}