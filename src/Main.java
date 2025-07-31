import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;

import java.util.*;

public class Main {
    static {
        // ✅ Load OpenCV DLL
        System.load("C:\\Users\\Surya_Goud\\Downloads\\opencv\\build\\java\\x64\\opencv_java4110.dll");
    }

    public static void main(String[] args) {
        // ✅ Set correct camera index (try 0, 1, 2 to match DroidCam)
        VideoCapture cap = new VideoCapture(2); // ← Change this if needed

        if (!cap.isOpened()) {
            System.out.println("❌ Cannot open camera.");
            return;
        }

        Mat frame = new Mat();
        int operand1 = -1, operand2 = -1;
        char operator = 0;
        int inputStep = -1;
        int lastDetectedFingers = -1;
        long fingerHoldStartTime = 0;

        while (true) {
            cap.read(frame);
            if (frame.empty()) continue;

            Core.flip(frame, frame, 1); // Mirror

            long currentTime = System.currentTimeMillis();

            // 📦 ROI box
            Rect roiRect = new Rect(frame.cols() / 2 - 200, 50, 400, 400);
            Mat roi = new Mat(frame, roiRect);
            Imgproc.rectangle(frame, roiRect.tl(), roiRect.br(), new Scalar(0, 255, 0), 2);
            Imgproc.putText(frame, "Place Hand Here", new Point(roiRect.x, roiRect.y - 10),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(255, 255, 255), 2);

            // 🖐 Preprocess
            Mat gray = new Mat();
            Imgproc.cvtColor(roi, gray, Imgproc.COLOR_BGR2GRAY);
            Imgproc.GaussianBlur(gray, gray, new Size(35, 35), 0);
            Mat thresh = new Mat();
            Imgproc.threshold(gray, thresh, 70, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

            // 🔍 Find contours
            List<MatOfPoint> contours = new ArrayList<>();
            Imgproc.findContours(thresh, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
            int fingerCount = 0;

            if (!contours.isEmpty()) {
                double maxArea = 0;
                int maxIndex = -1;
                for (int i = 0; i < contours.size(); i++) {
                    double area = Imgproc.contourArea(contours.get(i));
                    if (area > maxArea) {
                        maxArea = area;
                        maxIndex = i;
                    }
                }

                if (maxIndex != -1) {
                    MatOfPoint handContour = contours.get(maxIndex);
                    Imgproc.drawContours(roi, contours, maxIndex, new Scalar(0, 255, 0), 2);

                    MatOfInt hull = new MatOfInt();
                    Imgproc.convexHull(handContour, hull);

                    Point[] contourArray = handContour.toArray();
                    List<Point> hullPoints = new ArrayList<>();
                    for (int i = 0; i < hull.rows(); i++) {
                        int index = (int) hull.get(i, 0)[0];
                        hullPoints.add(contourArray[index]);
                    }

                    MatOfPoint hullMat = new MatOfPoint();
                    hullMat.fromList(hullPoints);
                    Imgproc.drawContours(roi, Arrays.asList(hullMat), 0, new Scalar(255, 0, 0), 2);

                    MatOfInt4 defects = new MatOfInt4();
                    Imgproc.convexityDefects(handContour, hull, defects);
                    if (defects.rows() > 0) {
                        for (int i = 0; i < defects.rows(); i++) {
                            double[] defect = defects.get(i, 0);
                            Point far = contourArray[(int) defect[2]];
                            double depth = defect[3] / 256.0;
                            if (depth > 10) {
                                fingerCount++;
                                Imgproc.circle(roi, far, 5, new Scalar(0, 0, 255), -1);
                            }
                        }
                        fingerCount = Math.min(fingerCount + 1, 5);
                    }
                }
            }

            // ⏱️ Finger hold detection
            if (fingerCount != lastDetectedFingers) {
                lastDetectedFingers = fingerCount;
                fingerHoldStartTime = currentTime;
            }

            if (currentTime - fingerHoldStartTime > 1500) {
                if (inputStep == -1 && fingerCount == 2) {
                    inputStep = 0;
                    System.out.println("✅ Start");
                } else if (inputStep == 0 && fingerCount >= 1 && fingerCount <= 5) {
                    operand1 = fingerCount;
                    inputStep = 1;
                    System.out.println("📥 Operand1: " + operand1);
                } else if (inputStep == 1) {
                    switch (fingerCount) {
                        case 1: operator = '+'; break;
                        case 2: operator = '-'; break;
                        case 3: operator = '*'; break;
                        case 4: operator = '/'; break;
                    }
                    if (operator != 0) {
                        inputStep = 2;
                        System.out.println("🔣 Operator: " + operator);
                    }
                } else if (inputStep == 2 && fingerCount >= 1 && fingerCount <= 5) {
                    operand2 = fingerCount;
                    inputStep = 3;
                    System.out.println("📥 Operand2: " + operand2);
                } else if (inputStep == 3 && fingerCount == 5) {
                    int result = 0;
                    switch (operator) {
                        case '+': result = operand1 + operand2; break;
                        case '-': result = operand1 - operand2; break;
                        case '*': result = operand1 * operand2; break;
                        case '/': result = operand2 != 0 ? operand1 / operand2 : 0; break;
                    }
                    inputStep = 4;
                    System.out.println("🧮 " + operand1 + " " + operator + " " + operand2 + " = " + result);
                    Imgproc.putText(frame, "Result: " + result, new Point(20, 180),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 1.2, new Scalar(0, 255, 255), 3);
                } else if (inputStep == 4 && fingerCount == 1) {
                    inputStep = -1;
                    operand1 = -1;
                    operand2 = -1;
                    operator = 0;
                    System.out.println("🔄 Reset");
                }
                fingerHoldStartTime = currentTime + 1000;
            }

            // ℹ️ Text overlay
            Imgproc.putText(frame, "Fingers: " + fingerCount, new Point(20, 50),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 255), 2);
            Imgproc.putText(frame, "Step: " + inputStep, new Point(20, 90),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(0, 255, 255), 2);

            if (inputStep == -1) {
                Imgproc.putText(frame, "✌️ Show 2 fingers to Start", new Point(20, 130),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 255, 0), 2);
            } else if (inputStep == 0) {
                Imgproc.putText(frame, "🖐 Show (1-5) = Operand1", new Point(20, 130),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 0), 2);
            } else if (inputStep == 1) {
                Imgproc.putText(frame, "☝=+, ✌=-, 🤘=*, ✋=/", new Point(20, 130),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 255, 0), 2);
            } else if (inputStep == 2) {
                Imgproc.putText(frame, "🖐 Show (1-5) = Operand2", new Point(20, 130),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 0), 2);
            } else if (inputStep == 3) {
                Imgproc.putText(frame, "✋ Show 5 = Calculate", new Point(20, 130),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 255, 0), 2);
            } else if (inputStep == 4) {
                Imgproc.putText(frame, "☝ Show 1 = Reset", new Point(20, 130),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 200, 255), 2);
            }

            HighGui.imshow("Hand Gesture Calculator", frame);
            if (HighGui.waitKey(1) == 27) break; // ESC to quit
        }

        cap.release();
        HighGui.destroyAllWindows();
    }
}
