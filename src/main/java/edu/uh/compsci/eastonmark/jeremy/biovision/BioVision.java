package edu.uh.compsci.eastonmark.jeremy.biovision;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
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

	private Mat singleStreamParse(Mat frame, int width, int height) {
		// Create Foveal Area
		int fovealAreaWidth = (int) (width * fovealPercentage);
		int fovealAreaHeight = (int) (height * fovealPercentage);
		int fovealAreaWidthStart = (width / 2) - (fovealAreaWidth / 2);
		int fovealAreaHeightStart = (height / 2) - (fovealAreaHeight / 2);

		Mat fovealArea = new Mat(fovealAreaWidth, fovealAreaHeight, 16);

		// Create Near-Peripheral Area
		int nearPeripheralAreaWidth = (int) (width * (fovealPercentage + nearPeripheralPercentage));
		int nearPeripheralAreaHeight = (int) (height * (fovealPercentage + nearPeripheralPercentage));
		int nearPeripheralAreaWidthStart = (width / 2) - (nearPeripheralAreaWidth / 2);
		int nearPeripheralAreaHeightStart = (height / 2) - (nearPeripheralAreaHeight / 2);

		Mat nearPeripheralArea = new Mat(nearPeripheralAreaWidth, nearPeripheralAreaHeight, 16);

		// Create Far-Peripheral Area
		Mat farPeripheralArea = frame.clone();

		// Stream to different areas

		for (int frameY = 0; frameY < height; frameY++) {
			for (int frameX = 0; frameX < width; frameX++) {
				int nearPeripheralX = frameX - nearPeripheralAreaWidthStart;
				int nearPeripheralY = frameY - nearPeripheralAreaHeightStart;

				int fovealX = frameX - fovealAreaWidthStart;
				int fovealY = frameY - fovealAreaHeightStart;

				double[] frameValue = frame.get(frameX, frameY);
				if (frameValue == null) {
					continue;
				}
				if ((frameX >= fovealAreaWidthStart) && (frameX < (fovealAreaWidthStart + fovealAreaWidth))
						&& (frameY >= fovealAreaHeightStart) && (frameY < (fovealAreaHeightStart + fovealAreaHeight))) {

					farPeripheralArea.put(frameX, frameY, 0, 0, 0);
					nearPeripheralArea.put(nearPeripheralX, nearPeripheralY, 0, 0, 0);
					fovealArea.put(fovealX, fovealY, frameValue[0], frameValue[1], frameValue[2]);
				} else if ((frameX >= nearPeripheralAreaWidthStart)
						&& (frameX < (nearPeripheralAreaWidthStart + nearPeripheralAreaWidth))
						&& (frameY >= nearPeripheralAreaHeightStart)
						&& (frameY < (nearPeripheralAreaHeightStart + nearPeripheralAreaHeight))) {
					farPeripheralArea.put(frameX, frameY, 0, 0, 0);
					nearPeripheralArea.put(nearPeripheralX, nearPeripheralY, frameValue[0], frameValue[1],
							frameValue[2]);
				} else {
//					nearPeripheralArea.put(nearPeripheralX, nearPeripheralY, 0, 0, 0);
//					fovealArea.put(fovealX, fovealY, 0, 0, 0);
				}
			}
		}

		// Run Detection Foveal
		Imgproc.cvtColor(fovealArea, fovealArea, Imgproc.COLOR_BGR2GRAY);
		Imgproc.equalizeHist(fovealArea, fovealArea);

		MatOfRect fovealDetectedObjects = new MatOfRect();
		classifier.detectMultiScale(fovealArea, fovealDetectedObjects);
		fovealObjects = fovealDetectedObjects.toList();

		// Run Detection Near-Peripheral
		Imgproc.cvtColor(nearPeripheralArea, nearPeripheralArea, Imgproc.COLOR_BGR2GRAY);
		Imgproc.equalizeHist(nearPeripheralArea, nearPeripheralArea);

		Imgproc.dilate(nearPeripheralArea, nearPeripheralArea, Mat.ones(3, 3, 1));
		MatOfRect nearPeripheralDetectedObjects = new MatOfRect();
		classifier.detectMultiScale(nearPeripheralArea, nearPeripheralDetectedObjects);
		nearPeripheralObjects = nearPeripheralDetectedObjects.toList();

		// Run Detection Far-Peripheral
		Imgproc.cvtColor(farPeripheralArea, farPeripheralArea, Imgproc.COLOR_BGR2GRAY);
		Imgproc.equalizeHist(farPeripheralArea, farPeripheralArea);

		Imgproc.dilate(farPeripheralArea, farPeripheralArea, Mat.ones(9, 9, 1));
		MatOfRect farPeripheralDetectedObjects = new MatOfRect();
		classifier.detectMultiScale(farPeripheralArea, farPeripheralDetectedObjects);
		farPeripheralObjects = farPeripheralDetectedObjects.toList();

		// Display Objects Foveal
		Rect fovealRect = new Rect(new Point(fovealAreaWidthStart, fovealAreaHeightStart),
				new Size(fovealAreaWidth, fovealAreaHeight));
		Imgproc.rectangle(frame, fovealRect, fovealColor);

		for (Rect detectedObject : fovealObjects) {
			detectedObject.x = detectedObject.x + fovealAreaWidthStart;
			detectedObject.y = detectedObject.y + fovealAreaHeightStart;
			Imgproc.rectangle(frame, detectedObject, fovealColor);
		}

		// Display Objects Near-Peripheral
		Rect nearPeripheralRect = new Rect(new Point(nearPeripheralAreaWidthStart, nearPeripheralAreaHeightStart),
				new Size(nearPeripheralAreaWidth, nearPeripheralAreaHeight));
		Imgproc.rectangle(frame, nearPeripheralRect, nearPeripheralColor);

		for (Rect detectedObject : nearPeripheralObjects) {
			detectedObject.x = detectedObject.x + nearPeripheralAreaWidthStart;
			detectedObject.y = detectedObject.y + nearPeripheralAreaHeightStart;
			Imgproc.rectangle(frame, detectedObject, nearPeripheralColor);
		}

		// Display Objects Far-Peripheral
		Rect farPeripheralRect = new Rect(new Point(0, 0), new Size(width, height));
		Imgproc.rectangle(frame, farPeripheralRect, farPeripheralColor);

		for (Rect detectedObject : farPeripheralObjects) {
			detectedObject.x = detectedObject.x;
			detectedObject.y = detectedObject.y;
			Imgproc.rectangle(frame, detectedObject, farPeripheralColor);
		}

		return frame;
	}

	private Mat analyzeFovealArea(Mat frame, int width, int height) {
		int areaWidth = (int) (width * fovealPercentage);
		int areaHeight = (int) (height * fovealPercentage);

		int areaWidthStart = (width / 2) - (areaWidth / 2);
		int areaHeightStart = (height / 2) - (areaHeight / 2);

		Mat areaFrame = frame.submat(new Range(areaHeightStart, areaHeightStart + areaHeight),
				new Range(areaWidthStart, areaWidthStart + areaWidth));

		Imgproc.cvtColor(areaFrame, areaFrame, Imgproc.COLOR_BGR2GRAY);
		Imgproc.equalizeHist(areaFrame, areaFrame);

		MatOfRect detectedObjects = new MatOfRect();
		classifier.detectMultiScale(areaFrame, detectedObjects);
		fovealObjects = detectedObjects.toList();

		for (Rect detectedObject : fovealObjects) {
			detectedObject.x = detectedObject.x + areaWidthStart;
			detectedObject.y = detectedObject.y + areaHeightStart;
			Imgproc.rectangle(frame, detectedObject, fovealColor);
		}

		Rect rect = new Rect(new Point(areaWidthStart, areaHeightStart), new Size(areaWidth, areaHeight));
		Imgproc.rectangle(frame, rect, fovealColor);

		return frame;
	}

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
	
	private Mat displayFrameSection(Mat frame, List<Rect> detectedObjects, int top, int height, int left, int width, int resizeRatio, Scalar color) {
		Rect fovealRect = new Rect(new Point(left, top),
				new Size(width, height));
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
		
		// Create Far-Peripheral Area
		// ???

		// Run Detection Foveal
		this.fovealObjects = detectFrameSection(frame, fovealTop, fovealHeight, fovealLeft, fovealWidth, 1);
		displayFrameSection(frame, fovealObjects, fovealTop, fovealHeight, fovealLeft, fovealWidth, 1, fovealColor);
		
		// Run Detection Near-Peripheral
		int nearPeripheralColumnWidth = (nearPeripheralWidth - fovealWidth) / 2;
		int nearPeripheralRowHeight = (nearPeripheralHeight - fovealHeight) / 2;
		
		if((frameNumber % nearPeripheralPriority) == 0) {
			this.leftNearPeripheralObjects = detectFrameSection(frame, nearPeripheralTop, nearPeripheralHeight, nearPeripheralLeft, nearPeripheralColumnWidth, nearPeripheralResizeSize);
			this.topNearPeripheralObjects = detectFrameSection(frame, nearPeripheralTop, nearPeripheralRowHeight, fovealLeft, fovealWidth, nearPeripheralResizeSize);
			this.rightNearPeripheralObjects = detectFrameSection(frame, nearPeripheralTop, nearPeripheralHeight, nearPeripheralLeft + nearPeripheralColumnWidth + fovealWidth, nearPeripheralColumnWidth, nearPeripheralResizeSize);
			this.bottomeNearPeripheralObjects = detectFrameSection(frame,fovealTop + fovealHeight, nearPeripheralRowHeight, fovealLeft, fovealWidth, nearPeripheralResizeSize);
		}
//		if(leftNearPeripheralObjects.size() + topNearPeripheralObjects.size() + rightNearPeripheralObjects.size() + bottomeNearPeripheralObjects.size() > 0) {
//			System.out.println("HIT");
//		}
		
		displayFrameSection(frame, leftNearPeripheralObjects, nearPeripheralTop, nearPeripheralHeight, nearPeripheralLeft, nearPeripheralColumnWidth, nearPeripheralResizeSize, nearPeripheralColor);
		displayFrameSection(frame, topNearPeripheralObjects, nearPeripheralTop, nearPeripheralRowHeight, fovealLeft, fovealWidth, nearPeripheralResizeSize, nearPeripheralColor);
		displayFrameSection(frame, rightNearPeripheralObjects, nearPeripheralTop, nearPeripheralHeight, nearPeripheralLeft + nearPeripheralColumnWidth + fovealWidth, nearPeripheralColumnWidth, nearPeripheralResizeSize, nearPeripheralColor);
		displayFrameSection(frame, bottomeNearPeripheralObjects, fovealTop + fovealHeight, nearPeripheralRowHeight, fovealLeft, fovealWidth, nearPeripheralResizeSize, nearPeripheralColor);
		
		
		// Run Detection Far-Peripheral
		int farPeripheralColumnWidth = (width - nearPeripheralWidth) / 2;
		int farPeripheralRowHeight = (height - nearPeripheralHeight) / 2;
		
		if((frameNumber % farPeripheralPriority) == 0) {
			this.leftFarPeripheralObjects = detectFrameSection(frame, 0, height, 0, farPeripheralColumnWidth, farPeripheralResizeSize);
			this.topFarPeripheralObjects = detectFrameSection(frame, 0, farPeripheralRowHeight, nearPeripheralLeft, nearPeripheralWidth, farPeripheralResizeSize);
			this.rightFarPeripheralObjects = detectFrameSection(frame, 0, height, farPeripheralColumnWidth + nearPeripheralWidth, farPeripheralColumnWidth, farPeripheralResizeSize);
			this.bottomeFarPeripheralObjects = detectFrameSection(frame,nearPeripheralTop + nearPeripheralHeight, farPeripheralRowHeight, nearPeripheralLeft, nearPeripheralWidth, farPeripheralResizeSize);
		}
		
		
		displayFrameSection(frame, leftFarPeripheralObjects, 0, height, 0, farPeripheralColumnWidth, farPeripheralResizeSize, farPeripheralColor);
		displayFrameSection(frame, topFarPeripheralObjects, 0, farPeripheralRowHeight, nearPeripheralLeft, nearPeripheralWidth, farPeripheralResizeSize, farPeripheralColor);
		displayFrameSection(frame, rightFarPeripheralObjects, 0, height, farPeripheralColumnWidth + nearPeripheralWidth, farPeripheralColumnWidth, farPeripheralResizeSize, farPeripheralColor);
		displayFrameSection(frame, bottomeFarPeripheralObjects, nearPeripheralTop + nearPeripheralHeight, farPeripheralRowHeight, nearPeripheralLeft, nearPeripheralWidth, farPeripheralResizeSize, farPeripheralColor);

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
		Imgproc.putText(frame, "Frame: " + frameNumber + " Frame Rate: " + frameRate, new Point(10,20), Imgproc.FONT_HERSHEY_PLAIN, 1, new Scalar(0,0,0));

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
			if(elapsedTime == 0) {
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
//		bv.setUseVisionSystem(true);
		bv.setUseVisionSystem(false);
		bv.run(args[0], args[1]);
	}

}
