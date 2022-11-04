package edu.uh.compsci.eastonmark.jeremy.biovision;

import java.util.ArrayList;
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

public class BioVision {
	static {
		nu.pattern.OpenCV.loadShared();
	}
	private boolean useVisionSystem = true;
	private CascadeClassifier classifier;

	// Foveal
	private double fovealPercentage = .3;
	private List<Rect> fovealObjects = new ArrayList<Rect>();
	private Scalar fovealColor = new Scalar(0, 0, 255);

	// Near-peripheral
	private double nearPeripheralPercentage = .3;
	private int nearPeripheralResizeSize = 3;
	private int nearPeripheralPriority = 3;
	private List<Rect> topNearPeripheralObjects = new ArrayList<Rect>();
	private List<Rect> leftNearPeripheralObjects = new ArrayList<Rect>();
	private List<Rect> rightNearPeripheralObjects = new ArrayList<Rect>();
	private List<Rect> bottomeNearPeripheralObjects = new ArrayList<Rect>();
	private Scalar nearPeripheralColor = new Scalar(0, 255, 0);

	// Far-peripheral
	private double farPeripheralPercentage = .4;
	private int farPeripheralResizeSize = 9;
	private int farPeripheralPriority = 7;
	private List<Rect> topFarPeripheralObjects = new ArrayList<Rect>();
	private List<Rect> leftFarPeripheralObjects = new ArrayList<Rect>();
	private List<Rect> rightFarPeripheralObjects = new ArrayList<Rect>();
	private List<Rect> bottomeFarPeripheralObjects = new ArrayList<Rect>();
	private Scalar farPeripheralColor = new Scalar(255, 0, 0);

	private List<Rect> detectFrameSection(Mat frame, int top, int height, int left, int width, int resizeRatio) {

		Mat detectFrame = frame.submat(new Range(top, top + height), new Range(left, left + width));

		if (resizeRatio > 1) {
			Size newSize = new Size(width / resizeRatio, height / resizeRatio);
			Imgproc.resize(detectFrame, detectFrame, newSize);
		}
		Imgproc.cvtColor(detectFrame, detectFrame, Imgproc.COLOR_BGR2GRAY);
		Imgproc.equalizeHist(detectFrame, detectFrame);

		MatOfRect detectedObjects = new MatOfRect();
		classifier.detectMultiScale(detectFrame, detectedObjects);
		return detectedObjects.toList();
	}

	private Mat displayFrameSection(Mat frame, List<Rect> detectedObjects, int top, int height, int left, int width,
			int resizeRatio, Scalar color) {
		Rect fovealRect = new Rect(new Point(left, top), new Size(width, height));
		Imgproc.rectangle(frame, fovealRect, color);

		for (Rect detectedObject : detectedObjects) {
			detectedObject.x = (detectedObject.x * resizeRatio) + left;
			detectedObject.y = (detectedObject.y * resizeRatio) + top;
			Imgproc.rectangle(frame, detectedObject, color);
		}

		return frame;
	}

	private Mat parseFrame(Mat frame, int frameNumber, int width, int height) {
		// Create Foveal Area
		int fovealWidth = (int) (width * fovealPercentage);
		int fovealHeight = (int) (height * fovealPercentage);
		int fovealLeft = (width / 2) - (fovealWidth / 2);
		int fovealTop = (height / 2) - (fovealHeight / 2);

		// Create Near-Peripheral Area
		double totalNearPeripheralPercentage = fovealPercentage + nearPeripheralPercentage;
		int nearPeripheralWidth = (int) (width * totalNearPeripheralPercentage);
		int nearPeripheralHeight = (int) (height * totalNearPeripheralPercentage);
		int nearPeripheralLeft = (width / 2) - (nearPeripheralWidth / 2);
		int nearPeripheralTop = (height / 2) - (nearPeripheralHeight / 2);

		// Run Detection Foveal
		this.fovealObjects = detectFrameSection(frame, fovealTop, fovealHeight, fovealLeft, fovealWidth, 1);
		displayFrameSection(frame, fovealObjects, fovealTop, fovealHeight, fovealLeft, fovealWidth, 1, fovealColor);

		// Run Detection Near-Peripheral
		int nearPeripheralColumnWidth = (nearPeripheralWidth - fovealWidth) / 2;
		int nearPeripheralRowHeight = (nearPeripheralHeight - fovealHeight) / 2;

		if ((frameNumber % nearPeripheralPriority) == 0) {
			this.leftNearPeripheralObjects = detectFrameSection(frame, nearPeripheralTop, nearPeripheralHeight,
					nearPeripheralLeft, nearPeripheralColumnWidth, nearPeripheralResizeSize);
			this.topNearPeripheralObjects = detectFrameSection(frame, nearPeripheralTop, nearPeripheralRowHeight,
					fovealLeft, fovealWidth, nearPeripheralResizeSize);
			this.rightNearPeripheralObjects = detectFrameSection(frame, nearPeripheralTop, nearPeripheralHeight,
					nearPeripheralLeft + nearPeripheralColumnWidth + fovealWidth, nearPeripheralColumnWidth,
					nearPeripheralResizeSize);
			this.bottomeNearPeripheralObjects = detectFrameSection(frame, fovealTop + fovealHeight,
					nearPeripheralRowHeight, fovealLeft, fovealWidth, nearPeripheralResizeSize);
		}

		displayFrameSection(frame, leftNearPeripheralObjects, nearPeripheralTop, nearPeripheralHeight,
				nearPeripheralLeft, nearPeripheralColumnWidth, nearPeripheralResizeSize, nearPeripheralColor);
		displayFrameSection(frame, topNearPeripheralObjects, nearPeripheralTop, nearPeripheralRowHeight, fovealLeft,
				fovealWidth, nearPeripheralResizeSize, nearPeripheralColor);
		displayFrameSection(frame, rightNearPeripheralObjects, nearPeripheralTop, nearPeripheralHeight,
				nearPeripheralLeft + nearPeripheralColumnWidth + fovealWidth, nearPeripheralColumnWidth,
				nearPeripheralResizeSize, nearPeripheralColor);
		displayFrameSection(frame, bottomeNearPeripheralObjects, fovealTop + fovealHeight, nearPeripheralRowHeight,
				fovealLeft, fovealWidth, nearPeripheralResizeSize, nearPeripheralColor);

		// Run Detection Far-Peripheral
		int farPeripheralColumnWidth = (width - nearPeripheralWidth) / 2;
		int farPeripheralRowHeight = (height - nearPeripheralHeight) / 2;

		if ((frameNumber % farPeripheralPriority) == 0) {
			this.leftFarPeripheralObjects = detectFrameSection(frame, 0, height, 0, farPeripheralColumnWidth,
					farPeripheralResizeSize);
			this.topFarPeripheralObjects = detectFrameSection(frame, 0, farPeripheralRowHeight, nearPeripheralLeft,
					nearPeripheralWidth, farPeripheralResizeSize);
			this.rightFarPeripheralObjects = detectFrameSection(frame, 0, height,
					farPeripheralColumnWidth + nearPeripheralWidth, farPeripheralColumnWidth, farPeripheralResizeSize);
			this.bottomeFarPeripheralObjects = detectFrameSection(frame, nearPeripheralTop + nearPeripheralHeight,
					farPeripheralRowHeight, nearPeripheralLeft, nearPeripheralWidth, farPeripheralResizeSize);
		}

		displayFrameSection(frame, leftFarPeripheralObjects, 0, height, 0, farPeripheralColumnWidth,
				farPeripheralResizeSize, farPeripheralColor);
		displayFrameSection(frame, topFarPeripheralObjects, 0, farPeripheralRowHeight, nearPeripheralLeft,
				nearPeripheralWidth, farPeripheralResizeSize, farPeripheralColor);
		displayFrameSection(frame, rightFarPeripheralObjects, 0, height, farPeripheralColumnWidth + nearPeripheralWidth,
				farPeripheralColumnWidth, farPeripheralResizeSize, farPeripheralColor);
		displayFrameSection(frame, bottomeFarPeripheralObjects, nearPeripheralTop + nearPeripheralHeight,
				farPeripheralRowHeight, nearPeripheralLeft, nearPeripheralWidth, farPeripheralResizeSize,
				farPeripheralColor);

		return frame;
	}

	private void detectAndDisplay(Mat frame, int frameNumber, double frameRate) {

		if (useVisionSystem) {
			int width = frame.width();
			int height = frame.height();
			frame = parseFrame(frame, frameNumber, width, height);

		} else {
			Mat detectFrame = new Mat();
			Imgproc.cvtColor(frame, detectFrame, Imgproc.COLOR_BGR2GRAY);
			Imgproc.equalizeHist(detectFrame, detectFrame);

			// Run Detector
			MatOfRect detectedObjects = new MatOfRect();
			classifier.detectMultiScale(detectFrame, detectedObjects);

			List<Rect> listOfDetectedObjects = detectedObjects.toList();
			for (Rect detectedObject : listOfDetectedObjects) {
				Imgproc.rectangle(frame, detectedObject, new Scalar(0, 0, 255));
			}

		}
		Imgproc.putText(frame, "Frame: " + frameNumber + " Frame Rate: " + frameRate, new Point(10, 20),
				Imgproc.FONT_HERSHEY_PLAIN, 1, new Scalar(0, 0, 0));

		HighGui.imshow("Classifier Detection", frame);
	}

	public void run(String cascadeFile, String videoFile) {

		// Load Classifier
		classifier = new CascadeClassifier();
		if (!classifier.load(cascadeFile)) {
			System.err.println("Error loading cascadeFile: " + cascadeFile);
			System.exit(-1);
		}

		// Load Capture
		VideoCapture videoCapture = new VideoCapture(videoFile);
		if (!videoCapture.isOpened()) {
			System.err.println("Error loading videoFile: " + videoFile);
			System.exit(-1);
		}
		int frameNumber = 0;
		long startTime = System.currentTimeMillis();
		// Detect and Display
		Mat frame = new Mat();
		while (videoCapture.read(frame)) {
			if (frame.empty()) {
				System.err.println("Empty frame detected");
				break;
			}
			frameNumber++;

			long currentTime = System.currentTimeMillis();
			long elapsedTime = (currentTime / 1000) - (startTime / 1000);
			if (elapsedTime == 0) {
				elapsedTime = 1;
			}
			double frameRate = frameNumber / (elapsedTime);

			detectAndDisplay(frame, frameNumber, frameRate);
			
			// Hit esc key to exit
			if (HighGui.waitKey(10) == 27) {
				break;
			}
		}

		HighGui.destroyAllWindows();

	}

	public boolean isUseVisionSystem() {
		return useVisionSystem;
	}

	public void setUseVisionSystem(boolean useVisionSystem) {
		this.useVisionSystem = useVisionSystem;
	}

	public static void main(String[] args) {
		BioVision bv = new BioVision();
		bv.setUseVisionSystem(true);
//		bv.setUseVisionSystem(false);
		bv.run(args[0], args[1]);
	}

}
