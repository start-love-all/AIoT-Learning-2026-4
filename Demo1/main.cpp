#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    // 1. 加载人脸检测器 记得把 XML 文件重新放回项目目录
    CascadeClassifier face_cascade;
    // 建议这里先用绝对路径，确保能跑通
    if (!face_cascade.load("E:/opencv/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml")) {
        cout << "错误：无法加载 XML 特征文件！" << endl;
        return -1;
    }
    //
    // 2. 打开摄像头
    VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    Mat frame, gray;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));

        for (auto& face : faces) {
            rectangle(frame, face, Scalar(255, 0, 0), 2);
            putText(frame, "Face Detected", Point(face.x, face.y - 10),
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
        }

        imshow("Face Recognition - Recovery Success", frame);

        if (waitKey(30) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}