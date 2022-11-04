package edu.uh.compsci.eastonmark.jeremy.sandbox;

import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Range;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

class ObjectDetection {
	boolean useFoveal = true;

	public void detectAndDisplay(Mat frame, CascadeClassifier carCascade) {
		int fovealWidthStart = 0;
		int fovealHeightStart = 0;

		Mat frameGray = new Mat();

		if (useFoveal) {
			int width = frame.width();
			int height = frame.height();

			// Parse to smaller frame
			int fovealWidth = (int) (width * .3);
			int fovealHeight = (int) (height * .3);

			fovealWidthStart = (width / 2) - (fovealWidth / 2);
			fovealHeightStart = (height / 2) - (fovealHeight / 2);

			Mat fovealFrame = frame.submat(new Range(fovealHeightStart, fovealHeightStart + fovealHeight),
					new Range(fovealWidthStart, fovealWidthStart + fovealWidth));

			Imgproc.cvtColor(fovealFrame, frameGray, Imgproc.COLOR_BGR2GRAY);
			Imgproc.equalizeHist(frameGray, frameGray);
			
			Rect fovealRect = new Rect(new Point(fovealWidthStart, fovealHeightStart), new Size(fovealWidth, fovealHeight));
			Imgproc.rectangle(frame, fovealRect, new Scalar(0, 0, 0));
			
			
		} else {
			Imgproc.cvtColor(frame, frameGray, Imgproc.COLOR_BGR2GRAY);
			Imgproc.equalizeHist(frameGray, frameGray);
		}

		// -- Detect Cars
		MatOfRect cars = new MatOfRect();
		carCascade.detectMultiScale(frameGray, cars);

		List<Rect> listOfCars = cars.toList();
		for (Rect car : listOfCars) {
			Point center = new Point(fovealWidthStart + car.x + car.width / 2,
					fovealHeightStart + car.y + car.height / 2);
			Imgproc.ellipse(frame, center, new Size(car.width / 2, car.height / 2), 0, 0, 360, new Scalar(255, 0, 255));
		}

		// -- Show what you got
		HighGui.imshow("Car Detection", frame);
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

			// -- 3. Apply the classifier to the frame
			detectAndDisplay(frame, carCascade);

			if (HighGui.waitKey(10) == 27) {
				break;// escape
			}
		}

		System.exit(0);
	}
}

public class ObjectDetectionDemo {
	static {
		nu.pattern.OpenCV.loadShared();
	}

	public static void main(String[] args) {
		new ObjectDetection().run(args);
	}
}